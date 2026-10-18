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
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [ids[i] for i in order if np.isfinite(scores[i])]
