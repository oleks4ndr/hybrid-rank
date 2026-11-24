# filter_ceiling.py
# CLI: how much could hard filters help if extraction were perfect? Filters are chosen from the labels, so these
# are upper bounds, never system results, and they aren't written to summary.jsonl. No LLM calls.
#
#   llm       the cached LLM decomposition (what we have now)
#   stated    values the query literally names and an answer has, used only if every answer survives, else none
#             -- what a facet list plus a perfect "requirement or context?" judgement could give
#   answer    every value any top-grade answer has, stated or not -- the bound for any extraction method
#   answer+   the same over every relevant answer (WANDS: Exact and Partial); same as answer for binary labels
# "or none" then picks filter / no filter per query by the headline metric, the best a perfect "none" option does.
#
# usage: python scripts/filter_ceiling.py wands --model bge-small
#        python scripts/filter_ceiling.py stark-amazon --model bge-small

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from tqdm import tqdm

from hybrid_rank.eval.config import EVAL_CONFIGS
from hybrid_rank.eval.run_eval import evaluate, load_processed
from hybrid_rank.filtering.hard_filter import FilterIndex, normalize
from hybrid_rank.ingest.registry import ADAPTERS
from hybrid_rank.query.decompose import Decomposer
from hybrid_rank.retrieval.base import top_k_ids
from hybrid_rank.retrieval.bm25 import BM25Index, bm25_dir
from hybrid_rank.retrieval.embed import ENCODERS, DenseIndex, embeddings_dir
from hybrid_rank.retrieval.rrf import DEFAULT_DEPTH, reciprocal_rank_fusion


def raw_values(filters, field):
    raw = filters.get(field)
    return [str(v) for v in (raw if isinstance(raw, list) else [] if raw is None else [raw])]


def names(query_tokens, value, match):
    key = normalize(value)
    if not key:
        return False
    if match == "prefix":  # a name: its words in order
        n = len(key)
        return any(tuple(query_tokens[i : i + n]) == key for i in range(len(query_tokens) - n + 1))
    return set(key) <= set(query_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(EVAL_CONFIGS))
    parser.add_argument("--model", choices=list(ENCODERS), default="bge-small")
    parser.add_argument("--split", default="dev", choices=["dev", "train"])  # labels pick the filters: never test
    args = parser.parse_args()

    config = EVAL_CONFIGS[args.dataset]
    headline = config.tune_metrics[0]
    fields = [f for f in ADAPTERS[args.dataset].filter_fields if f.kind == "text"]
    listings, queries, df = load_processed(args.dataset, args.split)
    filters = [l.filters for l in listings]
    ids = [l.id for l in listings]
    pos = {lid: i for i, lid in enumerate(ids)}
    index = FilterIndex(filters, ADAPTERS[args.dataset].filter_fields)
    bm25 = BM25Index(listings, cache_dir=bm25_dir(args.dataset))
    dense = DenseIndex(listings, ENCODERS[args.model], embeddings_dir(args.dataset))
    del listings

    top, relevant = defaultdict(list), defaultdict(list)
    for qid, lid, label in zip(df["query_id"], df["listing_id"], df["label"]):
        g = config.gain.get(label, 0.0)
        if g > 0 and lid in pos:
            relevant[qid].append(pos[lid])
            if g == max(config.gain.values()):
                top[qid].append(pos[lid])

    decomposer = Decomposer.for_dataset(args.dataset, index)
    llm = {} if decomposer.missing(q.text for q in queries) else decomposer.cache
    groups = [(f.name, [f]) for f in fields] + [("all", fields)]

    def constraints(kind, q, group):
        out = {}
        tokens = list(normalize(q.text))
        for f in group:
            if kind == "llm":
                values = llm[q.text].filters.get(f.name, [])
            else:
                rows = relevant[q.id] if kind == "answer+" else top[q.id]
                values = list(dict.fromkeys(v for r in rows for v in raw_values(filters[r], f.name)))
                if kind == "stated":
                    values = [v for v in values if names(tokens, v, f.match)]
            if values:
                out[f.name] = values
        return out

    kinds = (["llm"] if llm else []) + ["stated", "answer"]
    if any(len(relevant[q.id]) != len(top[q.id]) for q in queries):
        kinds.append("answer+")
    variants = [(kind, name, group) for kind in kinds for name, group in groups]
    lists = {"none": {}, **{(kind, name): {} for kind, name, _ in variants}}
    fired = {(kind, name): {} for kind, name, _ in variants}
    for q in tqdm(queries, desc="Retrieving"):
        sparse = bm25.get_scores(q.text)
        matched = sparse > 0
        vector = np.asarray(dense.model.encode(dense.spec.query_prefix + q.text, normalize_embeddings=True))
        cosine = dense.doc_emb @ vector

        def fused(mask):
            m = matched if mask is None else matched & mask
            return reciprocal_rank_fusion([top_k_ids(ids, sparse, DEFAULT_DEPTH, m),
                                           top_k_ids(ids, cosine, DEFAULT_DEPTH, mask)])[: config.depth]

        lists["none"][q.text] = fused(None)
        for kind, name, group in variants:
            mask = index.apply(constraints(kind, q, group)).mask
            if kind == "stated" and mask is not None and not mask[top[q.id]].all():
                mask = None  # the named value is context, not a requirement
            fired[kind, name][q.id] = mask is not None
            lists[kind, name][q.text] = lists["none"][q.text] if mask is None else fused(mask)

    per_query = {key: evaluate(queries, df, lambda t, k, r=ranked: r[t][:k], config, False).per_query.set_index("query_id")
                 for key, ranked in lists.items()}
    base = per_query["none"]
    shown = [m for m in config.metrics if not m.startswith(("R@50", "nDCG@50"))]
    print(f"\n{args.dataset}/{args.split}: {len(base)} queries; no filter: " + "  ".join(f"{m} {base[m].mean():.4f}" for m in shown))
    rows = [{"filter": "none", "field": "", "fired": 0.0, **base[config.metrics].mean().round(4).to_dict()}]
    for kind, name, _ in variants:
        run = per_query[kind, name]
        d = (run[headline] - base[headline]).to_numpy()
        boots = np.random.default_rng(0).choice(d, size=(10_000, len(d))).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        nonzero = d[np.abs(d) > 1e-9]
        p = wilcoxon(nonzero).pvalue if len(nonzero) else float("nan")
        share = float(np.mean([fired[kind, name][qid] for qid in base.index]))
        best = run.where(run[headline] >= base[headline], base)  # filter or none, per query
        print(f"\n{kind:<8}{name:<15} filtered {share:>4.0%}  {headline} {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f} "
              f"W/L {(d > 1e-9).sum()}/{(d < -1e-9).sum()}  | or none: {headline} {best[headline].mean() - base[headline].mean():+.4f}")
        print("   " + "  ".join(f"{m} {run[m].mean():.4f}" for m in shown))
        rows.append({"filter": kind, "field": name, "fired": round(share, 4), **run[config.metrics].mean().round(4).to_dict(),
                     f"{headline}_or_none": round(float(best[headline].mean()), 4)})
    path = Path("results") / f"filter_ceiling_{args.dataset}_{args.model}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
