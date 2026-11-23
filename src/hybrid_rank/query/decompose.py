# decompose.py
# LLM query decomposition: a search request -> {semantic_query, filters} over the dataset's filter fields.
#
# Every result is cached, one JSONL line per query, in results/decompositions/<dataset>/<model>_<hash>.jsonl,
# where the hash covers the prompt, schema and settings. So each query is paid for once, a re-run resumes
# where it stopped, the committed file reproduces the paper's numbers without an API key, and eval never
# calls the API. Changing the prompt starts a new file instead of mixing old and new answers.

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Sequence

from tqdm import tqdm

from hybrid_rank.query.prompts import output_schema, system_prompt
from hybrid_rank.schema import FilterField

if TYPE_CHECKING:
    from hybrid_rank.filtering.hard_filter import FilterIndex

DEFAULT_MODEL = "claude-haiku-5-5"  # a short extraction per query; --model claude-opus-5-5 to compare
DEFAULT_EFFORT = "low"  # a short extraction; current models always think, and effort bounds how much
MAX_TOKENS = 4096       # room for thinking plus a ~100-token JSON answer
MAX_RETRIES = 6         # the SDK retries 429 / 5xx / connection errors with backoff
# list prices, USD per 1M (input, output) tokens -- for the cost estimate only
PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50)}


@dataclass
class Decomposition:
    query: str
    semantic_query: str
    filters: Dict[str, Any] = field(default_factory=dict)  # only constrained fields: text -> [values],
                                                            # number -> {"min": x, "max": y} (either may be absent)
    error: Optional[str] = None  # refusal / max_tokens / bad output: plain query, no filters
    input_tokens: int = 0
    output_tokens: int = 0


def decompositions_dir(dataset: str) -> Path:
    return Path("results") / "decompositions" / dataset


def clean_filters(raw: Mapping[str, Any], fields: Sequence[FilterField]) -> Dict[str, Any]:
    """Keep only real constraints on declared fields: no empty lists, blank strings or null bounds."""
    out: Dict[str, Any] = {}
    for f in fields:
        value = raw.get(f.name)
        if f.kind == "number" and isinstance(value, dict):
            bounds = {
                k: float(value[k]) for k in ("min", "max")
                if isinstance(value.get(k), (int, float)) and not isinstance(value.get(k), bool)
            }
            if bounds:
                out[f.name] = bounds
        elif f.kind == "text" and isinstance(value, list):
            values = list(dict.fromkeys(v.strip() for v in value if isinstance(v, str) and v.strip()))
            if values:
                out[f.name] = values
    return out


class Decomposer:
    def __init__(
        self,
        dataset: str,
        catalog: str,
        fields: Sequence[FilterField],
        notes: Mapping[str, str],
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        cache_dir: Optional[Path] = None,
        client: Any = None,  # anything with .messages.create(); tests pass a fake, None = anthropic.Anthropic()
    ):
        self.fields = list(fields)
        self.model, self.effort = model, effort
        self.system = system_prompt(catalog, self.fields, notes)
        self.schema = output_schema(self.fields)
        settings = json.dumps(
            {"system": self.system, "schema": self.schema, "effort": effort, "max_tokens": MAX_TOKENS}, sort_keys=True
        )
        key = hashlib.sha1(settings.encode()).hexdigest()[:10]
        self.cache_path = (cache_dir or decompositions_dir(dataset)) / f"{model}_{key}.jsonl"
        self.client = client
        self.cache: Dict[str, Decomposition] = self._load()

    @classmethod
    def for_dataset(cls, dataset: str, index: "FilterIndex", model: str = DEFAULT_MODEL,
                    effort: str = DEFAULT_EFFORT, **kwargs: Any) -> "Decomposer":
        """The decomposer for a dataset: its adapter's catalog description and fields, its catalog's values."""
        from hybrid_rank.ingest.registry import ADAPTERS

        adapter = ADAPTERS[dataset]
        return cls(dataset, adapter.catalog_description, adapter.filter_fields, index.catalog_notes(),
                   model=model, effort=effort, **kwargs)

    def _load(self) -> Dict[str, Decomposition]:
        cache: Dict[str, Decomposition] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                try:
                    d = Decomposition(**json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    continue  # a line cut off when a run was killed mid-write; that query is just redone
                cache[d.query] = d
        return cache

    def _call(self, text: str) -> Decomposition:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=self.system,
            messages=[{"role": "user", "content": f"Search request: {text}"}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": self.schema}},
        )
        tokens = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        # a refusal or a cut-off answer may not match the schema: fall back to the plain query, no filters
        if response.stop_reason != "end_turn":
            return Decomposition(text, text, error=str(response.stop_reason), **tokens)
        try:
            data = json.loads(next(b.text for b in response.content if b.type == "text"))
        except (StopIteration, json.JSONDecodeError):
            return Decomposition(text, text, error="unparseable output", **tokens)
        semantic = data.get("semantic_query") if isinstance(data.get("semantic_query"), str) else ""
        filters = clean_filters(data.get("filters") or {}, self.fields)
        return Decomposition(text, semantic.strip() or text, filters, **tokens)

    def missing(self, texts: Iterable[str]) -> list:
        return [t for t in dict.fromkeys(texts) if t not in self.cache]

    def decompose_many(self, texts: Sequence[str], workers: int = 8) -> Dict[str, Decomposition]:
        """Decompose every text not already cached (in parallel), appending each result as it arrives."""
        todo = self.missing(texts)
        if todo:
            if self.client is None:
                import anthropic  # only the decomposition step needs the SDK and an API key

                self.client = anthropic.Anthropic(max_retries=MAX_RETRIES)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            pool = ThreadPoolExecutor(max_workers=workers)
            try:
                with open(self.cache_path, "a") as f, tqdm(total=len(todo), desc=f"Decomposing ({self.model})") as bar:
                    futures = [pool.submit(self._call, t) for t in todo]
                    for future in as_completed(futures):
                        d = future.result()  # an API error that outlived the retries stops the run; re-run resumes
                        self.cache[d.query] = d
                        f.write(json.dumps(asdict(d)) + "\n")
                        f.flush()
                        bar.update()
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
        return {t: self.cache[t] for t in texts}

    def cost(self, decompositions: Iterable[Decomposition]) -> Optional[float]:
        """Estimated USD at list price, or None for a model without a known price."""
        if self.model not in PRICES:
            return None
        price_in, price_out = PRICES[self.model]
        ds = list(decompositions)
        return (sum(d.input_tokens for d in ds) * price_in + sum(d.output_tokens for d in ds) * price_out) / 1e6
