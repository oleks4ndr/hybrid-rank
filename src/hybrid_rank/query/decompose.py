# decompose.py
# LLM query decomposition: a search request -> {semantic_query, filters} over the dataset's filter fields.
#
# Every result is cached, one JSONL line per query, in results/decompositions/<dataset>/<model>_<hash>.jsonl,
# where the hash covers the prompt, schema and settings. So each query is paid for once, a re-run resumes
# where it stopped, the committed file reproduces the paper's numbers without an API key, and eval never
# calls the API. Changing the prompt starts a new file instead of mixing old and new answers.

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hybrid_rank.llm import (
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    MAX_TOKENS,
    append_jsonl,
    call_json,
    estimate_cost,
    make_client,
    read_jsonl,
    run_parallel,
)
from hybrid_rank.query.prompts import (
    DEFAULT_PROMPT,
    GROUNDED,
    grounded_request,
    output_schema,
    system_prompt,
    verdict_schema,
)
from hybrid_rank.schema import FilterField

if TYPE_CHECKING:
    from hybrid_rank.filtering.hard_filter import FilterIndex

MAX_MENTIONS = 20   # catalog values shown per field by a grounded prompt
LOOKUP_VERSION = 2  # bump when FilterIndex.mentions changes: cached verdicts answer the old list


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
        prompt: str = DEFAULT_PROMPT,
        index: Optional["FilterIndex"] = None,  # grounded prompts look up catalog values here
        client: Any = None,  # anything with .messages.create(); tests pass a fake, None = anthropic.Anthropic()
    ):
        # a grounded prompt covers names and numbers only: on STaRK train, categories it judged to be required
        # kept just 79% of a query's answers (sellers file the same kind of product under different ones)
        self.grounded = prompt in GROUNDED
        self.fields = [f for f in fields if not self.grounded or f.kind == "number" or f.match == "prefix"]
        self.model, self.effort = model, effort
        self.system = system_prompt(catalog, self.fields, {} if self.grounded else notes, prompt)
        self.index = index
        self.schema = verdict_schema(self.fields) if self.grounded else output_schema(self.fields)
        settings: Any = {"system": self.system, "schema": self.schema, "effort": effort, "max_tokens": MAX_TOKENS}
        if self.grounded:
            # the request also depends on the lookup and the catalog, which the system prompt doesn't show
            settings.update(lookup=LOOKUP_VERSION, max_mentions=MAX_MENTIONS, notes=dict(notes))
        settings = json.dumps(settings, sort_keys=True)
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
                   model=model, effort=effort, index=index, **kwargs)

    def _load(self) -> Dict[str, Decomposition]:
        cache: Dict[str, Decomposition] = {}
        for record in read_jsonl(self.cache_path):
            try:
                d = Decomposition(**record)
            except TypeError:
                continue
            cache[d.query] = d
        return cache

    def mentions(self, text: str) -> List[Tuple[str, str, int]]:
        assert self.index is not None, "a grounded prompt needs the filter index"
        found = [(f.name, value, count) for f in self.fields if f.kind == "text"
                 for value, count in self.index.mentions(f.name, text)[:MAX_MENTIONS]]
        return found

    def _call_grounded(self, text: str) -> Decomposition:
        mentions = self.mentions(text)
        reply = call_json(self.client, self.model, self.effort, self.system, grounded_request(text, mentions), self.schema)
        tokens = {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens}
        if reply.data is None:
            return Decomposition(text, text, error=reply.error, **tokens)
        raw: Dict[str, Any] = dict(reply.data.get("numbers") or {})
        for verdict in reply.data.get("verdicts") or []:
            i, role = verdict.get("id"), verdict.get("role")
            if role == "requirement" and isinstance(i, int) and 1 <= i <= len(mentions):
                name, value, _ = mentions[i - 1]
                raw.setdefault(name, []).append(value)
        return Decomposition(text, text, clean_filters(raw, self.fields), **tokens)

    def _call(self, text: str) -> Decomposition:
        if self.grounded:
            return self._call_grounded(text)
        reply = call_json(self.client, self.model, self.effort, self.system, f"Search request: {text}", self.schema)
        tokens = {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens}
        if reply.data is None:
            return Decomposition(text, text, error=reply.error, **tokens)  # the plain query, no filters
        semantic = reply.data.get("semantic_query")
        semantic = semantic.strip() if isinstance(semantic, str) else ""
        filters = clean_filters(reply.data.get("filters") or {}, self.fields)
        return Decomposition(text, semantic or text, filters, **tokens)

    def missing(self, texts: Iterable[str]) -> list:
        return [t for t in dict.fromkeys(texts) if t not in self.cache]

    def decompose_many(self, texts: Sequence[str], workers: int = 8) -> Dict[str, Decomposition]:
        """Decompose every text not already cached (in parallel), appending each result as it arrives."""
        todo = self.missing(texts)
        if todo:
            if self.client is None:
                self.client = make_client()

            def results():
                for d in run_parallel(self._call, todo, workers, f"Decomposing ({self.model})"):
                    self.cache[d.query] = d
                    yield asdict(d)

            append_jsonl(self.cache_path, results())
        return {t: self.cache[t] for t in texts}

    def cost(self, decompositions: Iterable[Decomposition]) -> Optional[float]:
        return estimate_cost(self.model, decompositions)
