# base.py
# Reranking as one more stage behind the same search() interface: reorder the top `depth` results of any
# retriever and leave the rest alone. blend=True fuses the reranked and original orders with RRF instead of
# trusting the reranker outright, since an off-the-shelf model may be weaker than the fused list.

from typing import Dict, List, Mapping, Optional, Protocol

import numpy as np

from hybrid_rank.retrieval.base import Retriever
from hybrid_rank.retrieval.rrf import reciprocal_rank_fusion


class Reranker(Protocol):
    def rerank(self, query_text: str, listing_ids: List[str]) -> List[str]:
        """The same ids, best first."""
        ...


class RerankRetriever:
    def __init__(self, retriever: Retriever, reranker: Reranker, depth: int = 20, blend: bool = False):
        self.retriever = retriever
        self.reranker = reranker
        self.depth = depth
        self.blend = blend

    def search(self, query_text: str, top_k: int = 50, mask: Optional[np.ndarray] = None) -> List[str]:
        candidates = self.retriever.search(query_text, max(top_k, self.depth), mask)
        head, tail = candidates[: self.depth], candidates[self.depth :]
        if len(head) > 1:
            reranked = self.reranker.rerank(query_text, head)
            head = reciprocal_rank_fusion([head, reranked]) if self.blend else reranked
        return (head + tail)[:top_k]


class SavedRun:
    """A saved run's rankings as a retriever, so reranking needs no indexes in memory."""

    def __init__(self, rankings_by_query_text: Mapping[str, List[str]]):
        self.rankings: Dict[str, List[str]] = dict(rankings_by_query_text)

    def search(self, query_text: str, top_k: int = 50, mask: Optional[np.ndarray] = None) -> List[str]:
        if mask is not None:
            raise ValueError("a saved run is already filtered; it can't take a mask")
        return self.rankings.get(query_text, [])[:top_k]
