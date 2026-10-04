# run_eval.py
# CLI: run pipeline over a domain's test split, report metrics

from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd
from tqdm import tqdm

from hybrid_rank.eval.gain_schemes import GAIN_SCHEMES
from hybrid_rank.eval.metrics import mrr, ndcg_at_k, precision_at_k
from hybrid_rank.schema import Listing, Query


def load_processed(dataset: str):
    processed_dir = Path("data/processed") / dataset
    df_listings = pd.read_parquet(processed_dir / "listings.parquet")
    df_queries = pd.read_parquet(processed_dir / "queries.parquet")
    df_judgments = pd.read_parquet(processed_dir / "judgments.parquet")

    listings = [Listing.from_record(r) for r in df_listings.to_dict("records")]
    queries = [Query.from_record(r) for r in df_queries.to_dict("records")]
    return listings, queries, df_judgments


def evaluate(
    dataset: str,
    search_fn: Callable[[str, int], List[str]],
    top_k: int = 50,
    ndcg_k: int = 10,
    p_k: int = 5,
) -> Dict[str, float]:
    """Anything judged relevant but never retrieved by search_fn scores 0 --
    this is what makes it a real full-corpus eval, not a rerank-a-pool eval."""
    listings, queries, df_judgments = load_processed(dataset)
    gain = GAIN_SCHEMES[dataset]
    judgments_by_query = df_judgments.groupby("query_id")

    ndcgs, precisions, mrrs = [], [], []
    for query in tqdm(queries, desc=f"Evaluating on {dataset}"):
        if query.id not in judgments_by_query.groups:
            continue
        q_judgments = judgments_by_query.get_group(query.id)
        relevance = {row.listing_id: gain.get(row.label, 0.0) for row in q_judgments.itertuples()}
        relevant_ids = {lid for lid, g in relevance.items() if g > 0}

        ranked = search_fn(query.text, top_k)

        ndcgs.append(ndcg_at_k(ranked, relevance, k=ndcg_k))
        precisions.append(precision_at_k(ranked, relevant_ids, k=p_k))
        mrrs.append(mrr(ranked, relevant_ids))

    return {
        f"nDCG@{ndcg_k}": sum(ndcgs) / len(ndcgs),
        f"P@{p_k}": sum(precisions) / len(precisions),
        "MRR": sum(mrrs) / len(mrrs),
        "n_queries": len(ndcgs),
    }