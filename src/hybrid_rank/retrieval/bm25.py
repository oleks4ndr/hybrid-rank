# bm25.py

import re
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from hybrid_rank.schema import Listing


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, listings: List[Listing]):
        self.ids = [l.id for l in listings]
        corpus = [_tokenize(l.text) for l in listings]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query_text: str, top_k: int = 50) -> List[str]:
        scores = self.bm25.get_scores(_tokenize(query_text))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.ids[i] for i in top_idx]