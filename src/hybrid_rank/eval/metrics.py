# metrics.py
# ndcg@k, precision@k, mrr, recall@k, plus two views on unlabelled results — gain scheme is a parameter

import math
from typing import Dict, List, Set


def ndcg_at_k(ranked_ids: List[str], relevance: Dict[str, float], k: int = 10) -> float:
    def dcg(ids):
        return sum(relevance.get(pid, 0.0) / math.log2(i + 2) for i, pid in enumerate(ids[:k]))

    ideal_order = sorted(relevance, key=lambda pid: relevance[pid], reverse=True)
    idcg = dcg(ideal_order)
    return dcg(ranked_ids) / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    if k == 0:
        return 0.0
    return sum(1 for pid in ranked_ids[:k] if pid in relevant_ids) / k


def mrr(ranked_ids: List[str], relevant_ids: Set[str]) -> float:
    for i, pid in enumerate(ranked_ids, start=1):
        if pid in relevant_ids:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked_ids: List[str], relevant_ids: Set[str], k: int = 100) -> float:
    # undefined when nothing is relevant -- nan so the caller's mean skips the query
    if not relevant_ids:
        return float("nan")
    return len(relevant_ids.intersection(ranked_ids[:k])) / len(relevant_ids)


def condensed_ndcg_at_k(ranked_ids: List[str], relevance: Dict[str, float], k: int = 10) -> float:
    # condensed-list nDCG (Sakai, 2007): drop unlabelled results, then score what's left.
    # Plain nDCG counts unlabelled results as irrelevant; this ignores them -- report both as bounds
    return ndcg_at_k([pid for pid in ranked_ids if pid in relevance], relevance, k=k)


def unlabelled_at_k(ranked_ids: List[str], relevance: Dict[str, float], k: int = 10) -> float:
    # share of the returned top-k that has no label at all (labelled-irrelevant counts as labelled)
    top = ranked_ids[:k]
    if not top:
        return float("nan")
    return sum(1 for pid in top if pid not in relevance) / len(top)