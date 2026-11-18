# run_eval.py
# run any search_fn over a dataset's queries, report metrics

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, cast

import pandas as pd
from tqdm import tqdm

from hybrid_rank.eval.metrics import (
    condensed_ndcg_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    unlabelled_at_k,
)
from hybrid_rank.eval.splits import load_split
from hybrid_rank.schema import Listing, Query


def load_processed(dataset: str, split: str = "all"):
    """Listings are always the full corpus; only the queries are restricted to the split."""
    processed_dir = Path("data/processed") / dataset
    df_listings = pd.read_parquet(processed_dir / "listings.parquet")
    df_queries = pd.read_parquet(processed_dir / "queries.parquet")
    df_judgments = pd.read_parquet(processed_dir / "judgments.parquet")

    listings = [Listing.from_record(cast(Dict[str, Any], r)) for r in df_listings.to_dict("records")]
    queries = [Query.from_record(cast(Dict[str, Any], r)) for r in df_queries.to_dict("records")]
    if split != "all":
        keep = set(load_split(dataset)[split])
        queries = [q for q in queries if q.id in keep]
    return listings, queries, df_judgments


@dataclass
class EvalResult:
    summary: Dict[str, float]
    per_query: pd.DataFrame          # one row per query: id, category, every metric
    rankings: Dict[str, List[str]]   # query_id -> ranked listing ids


def evaluate(
    queries: List[Query],
    df_judgments: pd.DataFrame,
    search_fn: Callable[[str, int], List[str]],
    gain: Dict[str, float],
    ndcg_ks: Sequence[int] = (10, 50),
    p_k: int = 5,
    recall_ks: Sequence[int] = (50, 100),
    show_progress: bool = True,
) -> EvalResult:
    """Full-corpus eval: anything judged relevant but never retrieved scores 0.

    nDCG uses the graded gains. P@k and MRR count any gain > 0 as relevant.
    Recall@k counts only top-grade items (Exact on WANDS): queries average ~300 Partial
    labels, so recall over Partial would be capped far below 1. Queries with no top-grade
    item have undefined recall and are left out of its mean (see n_queries_recall).

    Plain nDCG counts unlabelled results as irrelevant, which penalizes systems that surface
    listings the label pool never saw. cnDCG@k ignores them instead, and unlabelled@k says how
    much of the top k that affects -- plain and condensed nDCG bound the truth from both sides.
    """
    top_k = max(*ndcg_ks, p_k, *recall_ks)
    top_gain = max(gain.values())
    judgments_by_query = df_judgments.groupby("query_id")

    rows, rankings = [], {}
    for query in tqdm(queries, desc="Evaluating", disable=not show_progress):
        if query.id not in judgments_by_query.groups:
            continue
        q_judgments = judgments_by_query.get_group(query.id)
        relevance = {
            str(lid): gain.get(str(label), 0.0)
            for lid, label in zip(q_judgments["listing_id"], q_judgments["label"])
        }
        relevant_ids = {lid for lid, g in relevance.items() if g > 0}
        top_grade_ids = {lid for lid, g in relevance.items() if g == top_gain}

        ranked = search_fn(query.text, top_k)
        rankings[query.id] = ranked

        row = {"query_id": query.id, "category": query.category}
        for k in ndcg_ks:
            row[f"nDCG@{k}"] = ndcg_at_k(ranked, relevance, k=k)
        row[f"P@{p_k}"] = precision_at_k(ranked, relevant_ids, k=p_k)
        row["MRR"] = mrr(ranked, relevant_ids)
        for k in recall_ks:
            row[f"R@{k}"] = recall_at_k(ranked, top_grade_ids, k=k)
        row[f"cnDCG@{ndcg_ks[0]}"] = condensed_ndcg_at_k(ranked, relevance, k=ndcg_ks[0])
        row[f"unlabelled@{ndcg_ks[0]}"] = unlabelled_at_k(ranked, relevance, k=ndcg_ks[0])
        rows.append(row)

    per_query = pd.DataFrame(rows)
    metrics = per_query.drop(columns=["query_id", "category"])
    # mean skips nan, i.e. queries with undefined recall
    summary: Dict[str, float] = {str(k): float(v) for k, v in metrics.mean().items()}
    summary["n_queries"] = len(per_query)
    summary["n_queries_recall"] = int(metrics[f"R@{recall_ks[0]}"].notna().sum())
    return EvalResult(summary=summary, per_query=per_query, rankings=rankings)
