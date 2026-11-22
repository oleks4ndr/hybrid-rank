# compare_runs.py
# CLI: compare runs saved by eval.py on one split -- a metric table, paired tests against a baseline,
# and how much each system's top 100 overlaps with the baseline's. Reads saved runs only, never re-retrieves.
#
# usage: python scripts/compare_runs.py wands --split dev --baseline bm25 --systems "dense-*" "rrf-*-d500"
#        python scripts/compare_runs.py wands --split dev --baseline rrf-minilm-k60-d500 --systems "rrf-*-d500" --examples 3

import argparse
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from hybrid_rank.eval.config import EVAL_CONFIGS

RUNS_DIR = Path("results/runs")
OVERLAP_DEPTH = 100
N_BOOT = 10_000
TIE = 1e-9  # per-query differences smaller than this count as ties


def find_systems(dataset: str, split: str, patterns: List[str]) -> List[str]:
    suffix = f"_{split}.per_query.parquet"
    names: List[str] = []
    for pattern in patterns:
        matches = sorted(p.name[: -len(suffix)] for p in (RUNS_DIR / dataset).glob(pattern + suffix))
        if not matches:
            raise SystemExit(f"no saved {split} runs match {pattern!r} in {RUNS_DIR / dataset}/ -- run scripts/eval.py first")
        names += [m for m in matches if m not in names]
    return names


def load_runs(dataset: str, split: str, systems: List[str], metrics: List[str]):
    per_query, rankings = {}, {}
    for system in systems:
        stem = RUNS_DIR / dataset / f"{system}_{split}"
        pq = pd.read_parquet(f"{stem}.per_query.parquet").set_index("query_id")
        missing = [m for m in metrics if m not in pq.columns]
        if missing:
            raise SystemExit(f"{system} was saved before {missing} existed -- re-run scripts/eval.py for it")
        per_query[system] = pq
        df = pd.read_parquet(f"{stem}.rankings.parquet").sort_values(["query_id", "rank"])
        rankings[system] = df.groupby("query_id")["listing_id"].apply(list).to_dict()

    query_ids = set(per_query[systems[0]].index)
    for system in systems[1:]:
        if set(per_query[system].index) != query_ids:
            raise SystemExit(f"{system} and {systems[0]} were evaluated on different queries -- re-run both")
    return per_query, rankings


def top_grade_sets(dataset: str) -> Dict[str, Set[str]]:
    gain = EVAL_CONFIGS[dataset].gain
    top_labels = [label for label, g in gain.items() if g == max(gain.values())]
    df = pd.read_parquet(Path("data/processed") / dataset / "judgments.parquet")
    return df[df["label"].isin(top_labels)].groupby("query_id")["listing_id"].apply(set).to_dict()


def paired(system: pd.Series, baseline: pd.Series) -> Dict[str, str]:
    """Mean per-query difference with a 95% bootstrap CI, Wilcoxon signed-rank p, and win/loss/tie counts."""
    d = (system - baseline).dropna().to_numpy()  # aligned on query_id; drops undefined recall
    boots = np.random.default_rng(0).choice(d, size=(N_BOOT, len(d))).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    nonzero = d[np.abs(d) > TIE]
    p = wilcoxon(nonzero).pvalue if len(nonzero) else float("nan")
    return {
        "diff": f"{d.mean():+.4f}",
        "95% CI": f"[{lo:+.4f}, {hi:+.4f}]",
        "p": f"{p:.4f}",
        "W/L/T": f"{(d > TIE).sum()}/{(d < -TIE).sum()}/{(np.abs(d) <= TIE).sum()}",
    }


def overlap(base_ranked: Dict[str, List[str]], sys_ranked: Dict[str, List[str]],
            query_ids, top_grade: Dict[str, Set[str]]) -> Dict[str, float]:
    jaccard, r_base, r_sys, r_union = [], [], [], []
    for qid in query_ids:
        b = set(base_ranked.get(qid, [])[:OVERLAP_DEPTH])
        s = set(sys_ranked.get(qid, [])[:OVERLAP_DEPTH])
        if b | s:
            jaccard.append(len(b & s) / len(b | s))
        e = top_grade.get(qid)
        if e:
            r_base.append(len(e & b) / len(e))
            r_sys.append(len(e & s) / len(e))
            r_union.append(len(e & (b | s)) / len(e))
    return {
        "Jaccard": float(np.mean(jaccard)),
        f"R@{OVERLAP_DEPTH} baseline": float(np.mean(r_base)),
        f"R@{OVERLAP_DEPTH} system": float(np.mean(r_sys)),
        f"R@{OVERLAP_DEPTH} combined": float(np.mean(r_union)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(EVAL_CONFIGS))
    parser.add_argument("--split", default="dev", help="a split in data/splits/<dataset>.json, or 'all'")
    parser.add_argument("--baseline", default="bm25")
    parser.add_argument("--systems", nargs="+", required=True, help="run names or glob patterns, e.g. 'rrf-*'")
    parser.add_argument("--examples", type=int, default=0, help="show each system's N biggest wins and losses")
    args = parser.parse_args()

    config = EVAL_CONFIGS[args.dataset]
    headline = config.tune_metrics[0]  # the metric examples are ranked by
    others = [s for s in find_systems(args.dataset, args.split, args.systems) if s != args.baseline]
    systems = [args.baseline] + others
    per_query, rankings = load_runs(args.dataset, args.split, systems, config.metrics)
    base = per_query[args.baseline]
    pd.set_option("display.width", 200)

    note = " (cnDCG@10 = nDCG@10 with unlabelled results removed; unlabelled@10 = their share of the top 10)"
    print(f"=== {args.dataset}/{args.split}: {len(base)} queries{note if config.condensed else ''}\n")
    table = pd.DataFrame({s: per_query[s][config.metrics].mean() for s in systems}).T
    print(table.to_string(float_format="{:.4f}".format))

    print(f"\n=== paired against {args.baseline}: diff = system - baseline, 95% bootstrap CI, Wilcoxon p, "
          f"per-query wins/losses/ties")
    print(f"p-values are uncorrected: across {len(others)} systems some p < 0.05 can be chance, so confirm a pick on test")
    for metric in config.tested:
        rows = {s: paired(per_query[s][metric], base[metric]) for s in others}
        print(f"\n{metric}")
        print(pd.DataFrame(rows).T.to_string())

    print(f"\n=== top-{OVERLAP_DEPTH} overlap with {args.baseline} (recall over top-grade labels; queries without one skipped)")
    top_grade = top_grade_sets(args.dataset)
    rows = {s: overlap(rankings[args.baseline], rankings[s], base.index, top_grade) for s in others}
    print(pd.DataFrame(rows).T.to_string(float_format="{:.4f}".format))

    if args.examples:
        text = pd.read_parquet(Path("data/processed") / args.dataset / "queries.parquet").set_index("id")["text"]
        for s in others:
            d = (per_query[s][headline] - base[headline]).sort_values()
            print(f"\n{s} vs {args.baseline}, {headline}")
            print("  biggest wins:  ", ", ".join(f"{text[q]!r} {d[q]:+.2f}" for q in d.index[::-1][: args.examples]))
            print("  biggest losses:", ", ".join(f"{text[q]!r} {d[q]:+.2f}" for q in d.index[: args.examples]))


if __name__ == "__main__":
    main()
