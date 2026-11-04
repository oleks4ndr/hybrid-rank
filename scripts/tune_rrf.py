# tune_rrf.py
# CLI: grid over RRF's k and fusion depth, always on the dev split so test stays untouched.
# Retrieves once at the deepest setting; each config only re-fuses, so the grid takes seconds.
# Saves the full grid to results/rrf_grid_<model>.jsonl
#
# usage: python scripts/tune_rrf.py wands --model minilm

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from hybrid_rank.eval.gain_schemes import GAIN_SCHEMES
from hybrid_rank.eval.run_eval import evaluate, load_processed
from hybrid_rank.retrieval.bm25 import BM25Index
from hybrid_rank.retrieval.embed import ENCODERS, DenseIndex, embeddings_dir
from hybrid_rank.retrieval.rrf import reciprocal_rank_fusion

K_VALUES = (1, 5, 10, 20, 40, 60, 80, 100, 200)
DEPTHS = (50, 100, 200, 500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(GAIN_SCHEMES))
    parser.add_argument("--model", choices=list(ENCODERS), default="minilm")
    args = parser.parse_args()

    listings, queries, df_judgments = load_processed(args.dataset, split="dev")
    retrievers = [
        BM25Index(listings),
        DenseIndex(listings, ENCODERS[args.model], embeddings_dir(args.dataset)),
    ]
    deepest = {
        q.text: [r.search(q.text, max(DEPTHS)) for r in retrievers]
        for q in tqdm(queries, desc="Retrieving")
    }

    def fused_search(k: int, depth: int):
        def search(query_text: str, top_k: int):
            # a shorter ranking is a prefix of a deeper one, so slicing matches a real search at that depth
            return reciprocal_rank_fusion([r[:depth] for r in deepest[query_text]], k=k)[:top_k]
        return search

    rows = []
    for depth in DEPTHS:
        for k in K_VALUES:
            result = evaluate(
                queries, df_judgments, fused_search(k, depth), GAIN_SCHEMES[args.dataset], show_progress=False
            )
            rows.append({"k": k, "depth": depth, **{m: round(v, 4) for m, v in result.summary.items()}})

    grid = pd.DataFrame(rows)
    for metric in ("nDCG@10", "R@100"):
        print(f"\n{metric} on {args.dataset}/dev (rows: depth, columns: k)")
        print(grid.pivot(index="depth", columns="k", values=metric).to_string())

    path = Path("results") / f"rrf_grid_{args.model}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"\nGrid saved to {path}")


if __name__ == "__main__":
    main()
