# rrf.py
# reciprocal rank fusion (Cormack et al., 2009) -- merges rankings by rank alone,
# so BM25 and cosine scores never need to share a scale

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from hybrid_rank.retrieval.base import Retriever

DEFAULT_K = 60  # the paper's value
DEFAULT_DEPTH = 500  # candidates from each retriever; both already score every listing, so depth is cheap


def reciprocal_rank_fusion(rankings: Sequence[List[str]], k: int = DEFAULT_K) -> List[str]:
    """Best-first ids, scored by the sum of 1 / (k + rank) over the rankings each id appears in.
    Ties keep first-seen order, so earlier rankings win them."""
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, lid in enumerate(ranking, start=1):
            scores[lid] += 1.0 / (k + rank)
    return sorted(scores, key=lambda lid: scores[lid], reverse=True)


class RRFRetriever:
    def __init__(self, retrievers: Sequence[Retriever], k: int = DEFAULT_K, depth: int = DEFAULT_DEPTH):
        self.retrievers = retrievers
        self.k = k
        self.depth = depth  # taken from each retriever regardless of the top_k asked for

    def search(self, query_text: str, top_k: int = 50, mask: Optional[np.ndarray] = None) -> List[str]:
        rankings = [r.search(query_text, self.depth, mask) for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, self.k)[:top_k]
