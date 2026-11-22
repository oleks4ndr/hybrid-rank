# query.py
# CLI: BM25, dense and RRF top-k for one or more queries, for manual inspection.
# If a query exactly matches a judged query, each result shows its label (E/P/I, - = unjudged).
#
# usage: python scripts/query.py wands "salon chair" "outdoor lounge cushions" --model minilm

import argparse

from hybrid_rank.eval.run_eval import load_processed
from hybrid_rank.retrieval.bm25 import BM25Index, bm25_dir
from hybrid_rank.retrieval.embed import ENCODERS, DenseIndex, embeddings_dir
from hybrid_rank.retrieval.rrf import RRFRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("queries", nargs="+")
    parser.add_argument("--model", choices=list(ENCODERS), default="minilm")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    listings, queries, df_judgments = load_processed(args.dataset)
    text_by_id = {l.id: l.text for l in listings}
    query_id_by_text = {q.text: q.id for q in queries}
    labels = dict(zip(zip(df_judgments["query_id"], df_judgments["listing_id"]), df_judgments["label"]))

    bm25 = BM25Index(listings, cache_dir=bm25_dir(args.dataset))
    dense = DenseIndex(listings, ENCODERS[args.model], embeddings_dir(args.dataset))
    retrievers = {"bm25": bm25, f"dense-{args.model}": dense, f"rrf-{args.model}": RRFRetriever([bm25, dense])}

    for query_text in args.queries:
        qid = query_id_by_text.get(query_text)
        print(f"\n=== {query_text!r}" + (f"  (query {qid})" if qid else "  (not a judged query)"))
        for name, retriever in retrievers.items():
            print(f"--- {name}")
            for rank, lid in enumerate(retriever.search(query_text, args.k), start=1):
                label = labels.get((qid, lid), "-")[0] if qid else " "
                print(f"{rank:>3} {label} {' '.join(text_by_id[lid].split())[:90]}")  # STaRK text has newlines


if __name__ == "__main__":
    main()
