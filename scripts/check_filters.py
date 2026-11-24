# check_filters.py
# CLI: compare decomposition prompts on one text field (default brand), from cached decompositions -- no API calls.
#
# "Names an answer's value" is a string match, so it is a proxy: themes such as NCAA and context mentions that
# happen to share the answers' brand also count as missed. Use it to compare prompts, not as true recall.
# retention = share of a filtered query's answers the filter keeps (listings without a value are kept).
#
# usage: python scripts/check_filters.py stark-amazon --split train --sample 300 --prompts v1 v2
#        python scripts/check_filters.py stark-amazon --split dev --prompts v1 v2 --show 10

import argparse

import numpy as np
import pandas as pd

from hybrid_rank.eval.config import EVAL_CONFIGS
from hybrid_rank.eval.run_eval import load_listing_filters, load_queries, processed_dir, sample_queries
from hybrid_rank.filtering.hard_filter import FilterIndex, normalize
from hybrid_rank.ingest.registry import ADAPTERS
from hybrid_rank.query.decompose import DEFAULT_MODEL, Decomposer
from hybrid_rank.query.prompts import DEFAULT_PROMPT, PROMPTS


def values(filters, field):
    raw = filters.get(field)
    return [" ".join(normalize(v)) for v in (raw if isinstance(raw, list) else [raw] if raw else [])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(EVAL_CONFIGS))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--sample", type=int, help="the same fixed sample decompose.py --sample uses")
    parser.add_argument("--field", default="brand")
    parser.add_argument("--prompts", nargs="+", choices=list(PROMPTS), default=[DEFAULT_PROMPT])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--show", type=int, default=5, help="examples of missed and answer-losing queries")
    args = parser.parse_args()

    fields = ADAPTERS[args.dataset].filter_fields
    filters = load_listing_filters(args.dataset)
    index = FilterIndex(filters, fields)
    ids = pd.read_parquet(processed_dir(args.dataset) / "listings.parquet", columns=["id"])["id"]
    pos = {lid: i for i, lid in enumerate(ids)}
    gain = EVAL_CONFIGS[args.dataset].gain
    top = [label for label, g in gain.items() if g == max(gain.values())]
    df = pd.read_parquet(processed_dir(args.dataset) / "judgments.parquet")
    answers = df[df["label"].isin(top)].groupby("query_id")["listing_id"].apply(list).to_dict()
    queries = [q for q in sample_queries(load_queries(args.dataset, args.split), args.sample) if q.id in answers]

    same, mixed = set(), set()  # queries naming an answer's value: every answer that has one shares it / some differ
    for q in queries:
        text = f" {' '.join(normalize(q.text))} "
        per_answer = [v for v in (values(filters[pos[a]], args.field) for a in answers[q.id]) if v]
        named = {v for vs in per_answer for v in vs if len(v) > 1 and f" {v} " in text}
        if named:
            (same if all(named & set(vs) for vs in per_answer) else mixed).add(q.id)
    print(f"{args.dataset}/{args.split}: {len(queries)} queries; {len(same) + len(mixed)} name an answer's "
          f"{args.field} ({len(same)} where every answer shares it, {len(mixed)} where some differ)")

    for prompt in args.prompts:
        decomposer = Decomposer.for_dataset(args.dataset, index, model=args.model, prompt=prompt)
        if decomposer.missing(q.text for q in queries):
            print(f"\n{prompt}: not decomposed yet -- run scripts/decompose.py with --prompt {prompt} and the same split / sample")
            continue
        filtered, kept, missed, lossy = set(), [], [], []
        for q in queries:
            constraint = decomposer.cache[q.text].filters.get(args.field)
            result = index.apply({args.field: constraint}) if constraint else None
            if result is None or result.mask is None:
                if q.id in same:
                    missed.append(q.text)
                continue
            filtered.add(q.id)
            kept.append(float(result.mask[[pos[a] for a in answers[q.id]]].mean()))
            if kept[-1] < 1:
                lossy.append((kept[-1], constraint, q.text))
        print(f"\n{prompt} ({decomposer.cache_path.name})")
        print(f"  filtered {len(filtered)} ({len(filtered) / len(queries):.1%}); same-value queries filtered "
              f"{len(filtered & same)}/{len(same)}; mixed-value queries filtered {len(filtered & mixed)}/{len(mixed)}")
        if kept:
            print(f"  retention {np.mean(kept):.1%}; filtered queries losing an answer {len(lossy)}, "
                  f"losing every answer {sum(k == 0 for k in kept)}")
        for text in missed[: args.show]:
            print(f"  missed: {text[:140]!r}")
        for k, constraint, text in sorted(lossy, key=lambda x: x[0])[: args.show]:
            print(f"  kept {k:.0%} with {constraint}: {text[:120]!r}")


if __name__ == "__main__":
    main()
