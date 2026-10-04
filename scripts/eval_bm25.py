# usage: python scripts/eval_bm25.py wands

import argparse

from hybrid_rank.eval.run_eval import evaluate, load_processed
from hybrid_rank.retrieval.bm25 import BM25Index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["wands"])
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--ndcg-k", type=int, default=10)
    args = parser.parse_args()

    listings, _, _ = load_processed(args.dataset)
    print(f"Building BM25 index over {len(listings):,} listings ...")
    index = BM25Index(listings)

    results = evaluate(args.dataset, search_fn=index.search, top_k=args.top_k, ndcg_k=args.ndcg_k)
    print("\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()