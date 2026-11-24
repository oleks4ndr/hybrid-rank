# decompose.py
# CLI: run the LLM query decomposition over a split's queries and cache the results. This is the only step
# that calls a paid API -- eval.py only reads the cache -- and a re-run skips queries already cached.
# Reads ANTHROPIC_API_KEY from the repo's .env (git-ignored); an exported variable takes precedence.
#
# usage: python scripts/decompose.py stark-amazon --split dev --limit 5    # smoke test: 5 queries
#        python scripts/decompose.py stark-amazon --split dev
#        python scripts/decompose.py wands --split dev --model claude-opus-5-5   # default: claude-haiku-5-5
#        python scripts/decompose.py stark-amazon --split train --sample 300 --prompt v2
#        python scripts/decompose.py stark-amazon --print-prompt           # the prompt, no API calls

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from hybrid_rank.eval.run_eval import load_listing_filters, load_queries, sample_queries
from hybrid_rank.filtering.hard_filter import FilterIndex
from hybrid_rank.ingest.registry import ADAPTERS
from hybrid_rank.llm import usage_line
from hybrid_rank.query.decompose import DEFAULT_EFFORT, DEFAULT_MODEL, Decomposer, Decomposition
from hybrid_rank.query.prompts import DEFAULT_PROMPT, PROMPTS, grounded_request

DATASETS = [name for name, adapter in ADAPTERS.items() if adapter.filter_fields]


def report(results: Dict[str, Decomposition], index: FilterIndex, decomposer: Decomposer, new: List[Decomposition],
           show: int) -> None:
    ds = list(results.values())
    n = len(ds)
    errors = Counter(d.error for d in ds if d.error)
    print(f"\n{n} queries decomposed" + (f"; fell back to the plain query for {dict(errors)}" if errors else ""))
    print(f"rewritten query differs from the original for {sum(d.semantic_query != d.query for d in ds) / n:.0%}")

    print("\nfilters extracted (unresolved = matches no catalog value, so it won't be applied):")
    for name, f in index.fields.items():
        with_value = [d for d in ds if name in d.filters]
        line = f"  {name:<14} {len(with_value):>5} queries ({len(with_value) / n:.0%})"
        if f.kind == "text":
            values = [v for d in with_value for v in d.filters[name]]
            unresolved = sum(not index.resolve(name, v) for v in values)
            top = Counter(v.lower() for v in values).most_common(8)
            line += f", {unresolved}/{len(values)} values unresolved; top: {', '.join(f'{v} ({c})' for v, c in top)}"
        print(line)

    if new:
        print(f"\nthis run: {usage_line(decomposer.model, new)}")
    print(f"cache: {decomposer.cache_path}")

    examples = [d for d in ds if d.filters][:show]
    if examples:
        print(f"\n{len(examples)} examples with filters:")
        for d in examples:
            print(f"  {d.query!r}\n    -> {d.semantic_query!r}  {d.filters}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--split", default="dev", help="a split in data/splits/<dataset>.json, or 'all'")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--prompt", choices=list(PROMPTS), default=DEFAULT_PROMPT, help="prompt version")
    parser.add_argument("--sample", type=int, help="a fixed random sample of N of the split's queries")
    parser.add_argument("--limit", type=int, help="only the split's first N queries (smoke test)")
    parser.add_argument("--workers", type=int, default=8, help="parallel API requests")
    parser.add_argument("--show", type=int, default=5, help="example decompositions to print")
    parser.add_argument("--print-prompt", action="store_true", help="print the system prompt and exit")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # the repo root's .env, wherever this runs from

    print(f"Indexing {args.dataset} filter fields ...")
    index = FilterIndex(load_listing_filters(args.dataset), ADAPTERS[args.dataset].filter_fields)
    print("  coverage: " + ", ".join(f"{name} {index.coverage(name):.0%}" for name in index.fields))
    decomposer = Decomposer.for_dataset(args.dataset, index, model=args.model, effort=args.effort,
                                        prompt=args.prompt)
    if args.print_prompt:
        print(f"\n{decomposer.system}")
        if decomposer.grounded:
            for q in load_queries(args.dataset, "train" if args.dataset == "stark-amazon" else args.split)[:3]:
                print("\n--- example message\n" + grounded_request(q.text, decomposer.mentions(q.text)))
        return

    queries = sample_queries(load_queries(args.dataset, args.split), args.sample)[: args.limit]
    texts = [q.text for q in queries]
    todo = decomposer.missing(texts)
    print(f"{len(texts)} {args.split} queries: {len(texts) - len(todo)} already cached, "
          f"calling {args.model} (effort {args.effort}, prompt {args.prompt}) for {len(todo)}")
    results = decomposer.decompose_many(texts, workers=args.workers)
    report(results, index, decomposer, [results[t] for t in todo], args.show)


if __name__ == "__main__":
    main()
