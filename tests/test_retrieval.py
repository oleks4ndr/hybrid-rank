# test_retrieval.py
# shared top-k selection and the mask contract every retriever follows

import numpy as np

from hybrid_rank.retrieval.base import top_k_ids
from hybrid_rank.retrieval.bm25 import BM25Index
from hybrid_rank.schema import Listing

IDS = ["a", "b", "c", "d"]
SCORES = np.array([0.1, 0.9, 0.5, 0.3])


def test_top_k_is_best_first_and_truncated():
    assert top_k_ids(IDS, SCORES, top_k=2) == ["b", "c"]


def test_top_k_mask_excludes_and_can_return_fewer_than_k():
    mask = np.array([True, False, False, True])
    assert top_k_ids(IDS, SCORES, top_k=3, mask=mask) == ["d", "a"]


def test_top_k_ties_follow_corpus_order():
    assert top_k_ids(IDS, np.array([0.5, 0.5, 0.9, 0.5]), top_k=4) == ["c", "a", "b", "d"]


def test_bm25_returns_only_lexical_matches_and_respects_mask():
    texts = ["walnut coffee table", "velvet sofa", "oak dining table", "floor lamp", "wool rug"]
    index = BM25Index([Listing(id=str(i), text=t) for i, t in enumerate(texts)])
    assert set(index.search("table", top_k=10)) == {"0", "2"}
    mask = np.array([False, True, True, True, True])
    assert index.search("table", top_k=10, mask=mask) == ["2"]
