# final_tables.py
# CLI: the paper's result tables from saved runs (no retrieval, no API calls). Each row is tested against the row
# named in its last column: 95% paired bootstrap CI and Wilcoxon p (uncorrected).
#
# usage: python scripts/final_tables.py                 # test, test-0.1, human, WANDS test, WANDS all 480
#        python scripts/final_tables.py --split dev     # the same systems on dev

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from hybrid_rank.eval.config import EVAL_CONFIGS
from hybrid_rank.eval.metrics import ndcg_at_k
from hybrid_rank.eval.run_eval import processed_dir
from hybrid_rank.eval.runs import load_rankings, runs_dir
from hybrid_rank.eval.splits import load_split

RRF, CE, HAIKU = "rrf-bge-small-k60-d500", "ce.bge-base-d100.blend", "llm.claude-haiku-5-5-d20"
BRAND = f"{RRF}+f.brand"
# label, run, run it is tested against
STARK = [
    ("BM25 (Okapi idf)", "bm25-okapi", "bm25"),
    ("BM25", "bm25", None),
    ("Dense (bge-small)", "dense-bge-small", "bm25"),
    ("RRF", RRF, "bm25"),
    ("RRF + rewritten query", f"{RRF}+rw", RRF),
    ("RRF + all filters", f"{RRF}+f.brand.category.color.price", RRF),
    ("RRF + brand, strict", f"{BRAND}.strict", RRF),
    ("RRF + brand", BRAND, RRF),
    ("RRF + CE (bge)", f"{RRF}+{CE}", RRF),
    ("RRF + brand + CE (MiniLM)", f"{BRAND}+ce.minilm-d100.blend", BRAND),
    ("RRF + brand + CE (bge)", f"{BRAND}+{CE}", BRAND),
    ("RRF + brand + CE + Haiku", f"{BRAND}+{CE}+{HAIKU}", f"{BRAND}+{CE}"),
    ("  same, against BM25", f"{BRAND}+{CE}+{HAIKU}", "bm25"),
]
WANDS = [
    ("BM25 (Okapi idf)", "bm25-okapi", "bm25"),
    ("BM25", "bm25", None),
    ("Dense (bge-small)", "dense-bge-small", "bm25"),
    ("RRF", RRF, "bm25"),
    ("RRF + rewritten query", f"{RRF}+rw", RRF),
    ("RRF + all filters", f"{RRF}+f.product_class.color.style", RRF),
    ("RRF + CE (MiniLM)", f"{RRF}+ce.minilm-d20.blend", RRF),
    ("RRF + Haiku", f"{RRF}+{HAIKU}", RRF),
    ("  CE against BM25", f"{RRF}+ce.minilm-d20.blend", "bm25"),
]
# Hit@1, Hit@5, R@20, MRR in %, from Wu et al. 2024 (Tables 6, 7) -- other systems, quoted for reference
PUBLISHED = {
    "test": {"BM25": (44.94, 67.42, 53.77, 55.30), "ColBERTv2": (46.10, 66.02, 53.44, 55.51),
             "GritLM-7b": (42.08, 66.87, 56.52, 53.46), "MFAR (trained, Li et al.)": (57.4, None, 66.3, 68.1)},
    # reranker R@20 left out: the printed values differ from ada-002's, which reranking its top 20 can't change
    "test-0.1": {"BM25": (42.68, 67.07, 54.48, 54.02), "GritLM-7b": (43.29, 71.34, 56.14, 55.07),
                 "Claude 3 reranker": (45.49, 71.13, None, 55.91), "GPT-4 reranker": (44.79, 71.17, None, 55.69)},
    "human": {"multi-ada-002": (46.91, 72.84, 40.22, 58.74), "Claude 3 reranker": (53.09, 74.07, 35.46, 62.11),
              "GPT-4 reranker": (50.62, 75.31, 35.46, 61.06)},
}
LIU = {"nDCG@1": 0.7919, "nDCG@20": 0.7914, "nDCG@50": 0.7866, "nDCG@100": 0.7721}  # qq-pretrained, overall bucket


def load(dataset, run, split):
    path = runs_dir(dataset) / f"{run}_{split}.per_query.parquet"
    return pd.read_parquet(path).set_index("query_id") if path.exists() else None


def table(title, dataset, rows, metrics, frames):
    """frames: run -> per-query frame (missing runs are reported, not skipped silently)"""
    print(f"\n### {title}")
    print(f"{'system':<28}" + "".join(f"{m:>9}" for m in metrics) + f"   vs   {metrics[0]} / {metrics[-1]}: diff [95% CI] p")
    for label, run, against in rows:
        pq = frames(run)
        if pq is None:
            print(f"{label:<28}  MISSING: {run}")
            continue
        line = f"{label:<28}" + "".join(f"{pq[m].mean():>9.4f}" for m in metrics)
        base = frames(against) if against else None
        if base is not None:
            parts = []
            for m in (metrics[0], metrics[-1]):
                d = (pq[m] - base[m].reindex(pq.index)).dropna().to_numpy()
                boots = np.random.default_rng(0).choice(d, size=(5_000, len(d))).mean(axis=1)
                lo, hi = np.percentile(boots, [2.5, 97.5])
                nz = d[np.abs(d) > 1e-9]
                p = wilcoxon(nz).pvalue if len(nz) else float("nan")
                parts.append(f"{d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] " + ("p<.0001" if p < 1e-4 else f"p={p:.4f}"))
            line += f"   {against}: " + " / ".join(parts)
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["test", "dev"], default="test")
    args = parser.parse_args()
    metrics = ["Hit@1", "Hit@5", "R@20", "R@100", "MRR"]
    splits = load_split("stark-amazon")
    for split in (["test", "test-0.1", "human"] if args.split == "test" else ["dev"]):
        source = "test" if split == "test-0.1" else split

        def frames(run, keep=splits[split], source=source):
            pq = load("stark-amazon", run, source)
            return pq if pq is None else pq.reindex([q for q in keep if q in pq.index])

        table(f"STaRK-Amazon {split} ({len(splits[split])} queries)", "stark-amazon", STARK, metrics, frames)
        for name, values in PUBLISHED.get(split, {}).items():
            print(f"  published: {name:<26}" + "  ".join(f"{m} {v / 100:.4f}" for m, v in zip(["Hit@1", "Hit@5", "R@20", "MRR"], values) if v))
        for run in (BRAND, f"{BRAND}.strict"):
            path = runs_dir("stark-amazon") / f"{run}_{source}.filters.parquet"
            if path.exists():
                t = pd.read_parquet(path).set_index("query_id").reindex(splits[split])
                f = t[t["filtered"].astype(bool)]["retention"].dropna()
                print(f"  filter {run.removeprefix(RRF)}: used on {t['filtered'].astype(bool).mean():.1%}, answers kept {f.mean():.1%}, "
                      f"queries left with none {int((f == 0).sum())}")

    wm = ["nDCG@10", "nDCG@50", "P@5", "R@100", "cnDCG@10", "MRR"]
    table(f"WANDS {args.split}", "wands", WANDS, wm, lambda run: load("wands", run, args.split))

    # Liu et al. score all 480 queries (dev included, where our settings were chosen), so the union is for comparison only
    df = pd.read_parquet(processed_dir("wands") / "judgments.parquet")
    gain = EVAL_CONFIGS["wands"].gain
    relevance = {q: dict(zip(g["listing_id"], g["label"].map(gain))) for q, g in df.groupby("query_id")}
    print("\n### WANDS nDCG at other depths (unlabelled = 0)  Liu et al. 2022 qq: " + "  ".join(f"{k} {v}" for k, v in LIU.items()))
    for label, run in [("BM25", "bm25"), ("RRF", RRF), ("RRF + CE (MiniLM)", f"{RRF}+ce.minilm-d20.blend"), ("RRF + Haiku", f"{RRF}+{HAIKU}")]:
        for name, parts in (("dev", ["dev"]), ("test", ["test"]), ("all 480", ["dev", "test"])):
            if args.split == "dev" and name != "dev":
                continue
            if not all((runs_dir("wands") / f"{run}_{p}.rankings.parquet").exists() for p in parts):
                print(f"{label:<20}{name:<9}MISSING")
                continue
            ranked = {q: r for p in parts for q, r in load_rankings("wands", run, p).items()}
            print(f"{label:<20}{name:<9}n={len(ranked):<4}" + "  ".join(
                f"nDCG@{k} {np.mean([ndcg_at_k(r, relevance[q], k) for q, r in ranked.items()]):.4f}" for k in (1, 20, 50, 100)))


if __name__ == "__main__":
    main()
