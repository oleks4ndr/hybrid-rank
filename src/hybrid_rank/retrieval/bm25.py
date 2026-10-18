# bm25.py

import re
from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from hybrid_rank.retrieval.base import top_k_ids
from hybrid_rank.schema import Listing


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, listings: List[Listing]):
        self.ids = [l.id for l in listings]
        corpus = [_tokenize(l.text) for l in listings]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query_text: str, top_k: int = 50, mask: Optional[np.ndarray] = None) -> List[str]:
        scores = self.bm25.get_scores(_tokenize(query_text))
        # a listing sharing no term with the query scores 0 -- it isn't a lexical match, so don't return it
        matched = scores > 0
        return top_k_ids(self.ids, scores, top_k, matched if mask is None else matched & mask)
