# usage: python scripts/eval_bm25.py wands [--split dev|test|all]

import argparse

from hybrid_rank.eval.gain_schemes import GAIN_SCHEMES
from hybrid_rank.eval.run_eval import evaluate, load_processed
from hybrid_rank.retrieval.bm25 import BM25Index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["wands"])
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    args = parser.parse_args()

    listings, queries, df_judgments = load_processed(args.dataset, split=args.split)
    print(f"Building BM25 index over {len(listings):,} listings ...")
    index = BM25Index(listings)

    result = evaluate(queries, df_judgments, search_fn=index.search, gain=GAIN_SCHEMES[args.dataset])
    print(f"\nResults ({args.split}):")
    for k, v in result.summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
