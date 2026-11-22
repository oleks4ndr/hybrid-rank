# base.py
# what every retriever looks like, plus the shared exact top-k selection

from typing import List, Optional, Protocol

import numpy as np


class Retriever(Protocol):
    def search(self, query_text: str, top_k: int = 50, mask: Optional[np.ndarray] = None) -> List[str]:
        """Best-first listing ids. mask: one bool per listing, False = excluded (hard filters)."""
        ...


def top_k_ids(ids: List[str], scores: np.ndarray, top_k: int, mask: Optional[np.ndarray] = None) -> List[str]:
    """Exact top-k over every listing, ties broken by corpus order.
    Excluded listings are dropped, so a selective mask can return fewer than top_k."""
    if top_k <= 0:
        return []
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    if top_k < len(scores):
        # find the k-th best score in O(n), then stable-sort only what's at or above it -- same result as
        # a full stable argsort, without sorting a ~1M-listing corpus for every query
        kth = -np.partition(-scores, top_k - 1)[top_k - 1]
        candidates = np.flatnonzero(scores >= kth) if np.isfinite(kth) else np.flatnonzero(np.isfinite(scores))
        order = candidates[np.argsort(-scores[candidates], kind="stable")][:top_k]
    else:
        order = np.argsort(-scores, kind="stable")
    return [ids[i] for i in order if np.isfinite(scores[i])]
