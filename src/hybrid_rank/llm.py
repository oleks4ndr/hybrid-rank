# llm.py
# What every LLM step (query decomposition, LLM reranking) shares: one structured-output call, a resumable
# JSONL cache, parallel execution and a cost estimate. Each result is paid for once; evaluation only reads
# the caches.

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, TypeVar

from tqdm import tqdm

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_EFFORT = "low"  # current models always think, and effort bounds how much
MAX_TOKENS = 4096       # room for thinking plus a short JSON answer
MAX_RETRIES = 6         # the SDK retries 429 / 5xx / connection errors with backoff
# list prices, USD per 1M (input, output) tokens -- for the cost estimate only
PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50)}

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class Reply:
    data: Optional[Dict[str, Any]]  # the parsed JSON answer, or None with the reason in error
    error: Optional[str]
    input_tokens: int
    output_tokens: int


def make_client() -> Any:
    import anthropic  # only the steps that call the API need the SDK and a key

    return anthropic.Anthropic(max_retries=MAX_RETRIES)


def call_json(client: Any, model: str, effort: str, system: str, user: str, schema: Dict[str, Any]) -> Reply:
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
    )
    tokens = (response.usage.input_tokens, response.usage.output_tokens)
    # a refusal or a cut-off answer may not match the schema; callers fall back to doing nothing
    if response.stop_reason != "end_turn":
        return Reply(None, str(response.stop_reason), *tokens)
    try:
        data = json.loads(next(b.text for b in response.content if b.type == "text"))
    except (StopIteration, json.JSONDecodeError):
        return Reply(None, "unparseable output", *tokens)
    if not isinstance(data, dict):
        return Reply(None, "unparseable output", *tokens)
    return Reply(data, None, *tokens)


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut off when a run was killed mid-write; that item is just redone
        if isinstance(record, dict):
            yield record


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Appends as records arrive. The file appears with the first record, so a run that fails before any
    result leaves nothing behind."""
    f = None
    try:
        for record in records:
            if f is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                cut_off = path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
                f = open(path, "a")
                if cut_off:
                    f.write("\n")  # don't glue this record onto a half-written line
            f.write(json.dumps(record) + "\n")
            f.flush()
    finally:
        if f is not None:
            f.close()


def run_parallel(fn: Callable[[T], R], items: Sequence[T], workers: int, desc: str) -> Iterator[R]:
    """Results as they finish. An API error that outlived the SDK's retries stops the run; re-running resumes."""
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(fn, item) for item in items]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            yield future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def estimate_cost(model: str, records: Iterable[Any]) -> Optional[float]:
    """USD at list price for records with input_tokens / output_tokens; None if the price isn't known."""
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    rs: List[Any] = list(records)
    return (sum(r.input_tokens for r in rs) * price_in + sum(r.output_tokens for r in rs) * price_out) / 1e6


def usage_line(model: str, records: Sequence[Any]) -> str:
    cost = estimate_cost(model, records)
    line = (f"{len(records)} calls, {sum(r.input_tokens for r in records):,} input + "
            f"{sum(r.output_tokens for r in records):,} output tokens")
    return line + (f", ~${cost:.{2 if cost >= 0.01 else 4}f} at list price" if cost is not None else "")
