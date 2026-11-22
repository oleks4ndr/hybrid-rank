# test_retrieval.py
# shared top-k selection and the mask contract every retriever follows

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from hybrid_rank.retrieval.base import top_k_ids
from hybrid_rank.retrieval.bm25 import BM25Index, _tokenize
from hybrid_rank.schema import Listing

# "table" is in 4 of 6 listings, so its raw idf is negative and gets rank_bm25's epsilon floor
CORPUS = [
    "walnut coffee table", "velvet sofa", "oak dining table table", "floor lamp",
    "glass side table", "folding table and chairs with a very long description of the set",
]

IDS = ["a", "b", "c", "d"]
SCORES = np.array([0.1, 0.9, 0.5, 0.3])


def test_top_k_is_best_first_and_truncated():
    assert top_k_ids(IDS, SCORES, top_k=2) == ["b", "c"]


def test_top_k_mask_excludes_and_can_return_fewer_than_k():
    mask = np.array([True, False, False, True])
    assert top_k_ids(IDS, SCORES, top_k=3, mask=mask) == ["d", "a"]


def test_top_k_ties_follow_corpus_order():
    assert top_k_ids(IDS, np.array([0.5, 0.5, 0.9, 0.5]), top_k=4) == ["c", "a", "b", "d"]


def test_top_k_partial_selection_matches_full_stable_sort():
    rng = np.random.default_rng(0)
    ids = [str(i) for i in range(500)]
    for _ in range(50):
        scores = rng.integers(0, 6, size=500).astype(float)  # heavy ties, including at the k-th place
        mask = rng.random(500) > 0.3
        for top_k in (1, 7, 100, 499, 500, 900):
            masked = np.where(mask, scores, -np.inf)
            full = [ids[i] for i in np.argsort(-masked, kind="stable")[:top_k] if np.isfinite(masked[i])]
            assert top_k_ids(ids, scores, top_k, mask) == full


def test_top_k_zero_and_all_masked():
    assert top_k_ids(IDS, SCORES, top_k=0) == []
    assert top_k_ids(IDS, SCORES, top_k=2, mask=np.zeros(4, dtype=bool)) == []


@pytest.mark.parametrize("query", ["table", "oak table", "table table lamp", "chairs set", "unknownword"])
def test_okapi_scores_match_rank_bm25_exactly(query):
    index = BM25Index([Listing(id=str(i), text=t) for i, t in enumerate(CORPUS)], idf="okapi")
    reference = BM25Okapi([_tokenize(t) for t in CORPUS]).get_scores(_tokenize(query))
    assert np.array_equal(index.get_scores(query), reference)


def test_lucene_idf_hand_computed_and_never_floored():
    index = BM25Index([Listing(id=str(i), text=t) for i, t in enumerate(CORPUS)], idf="lucene")
    docs = [_tokenize(t) for t in CORPUS]
    n, avgdl = len(docs), sum(map(len, docs)) / len(docs)

    def expected(term, doc):
        df = sum(term in d for d in docs)
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        tf, dl = doc.count(term), len(doc)
        return idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))

    # "table" is in 4 of 6 listings: its okapi idf is negative and gets floored; lucene's formula alone
    # keeps it positive (how high okapi's floor sits depends on the vocabulary's mean idf)
    assert np.allclose(index.get_scores("table"), [expected("table", d) for d in docs])
    assert index.get_scores("table")[0] > 0


def test_bm25_cache_round_trip(tmp_path):
    listings = [Listing(id=str(i), text=t) for i, t in enumerate(CORPUS)]
    built = BM25Index(listings, cache_dir=tmp_path)
    loaded = BM25Index(listings, cache_dir=tmp_path)
    assert not built.loaded_from_cache and loaded.loaded_from_cache
    assert np.array_equal(built.get_scores("oak table lamp"), loaded.get_scores("oak table lamp"))
    # different text -> different cache entry, never a stale hit
    changed = BM25Index([Listing(id="0", text="something else")], cache_dir=tmp_path)
    assert not changed.loaded_from_cache


def test_bm25_returns_only_lexical_matches_and_respects_mask():
    texts = ["walnut coffee table", "velvet sofa", "oak dining table", "floor lamp", "wool rug"]
    index = BM25Index([Listing(id=str(i), text=t) for i, t in enumerate(texts)])
    assert set(index.search("table", top_k=10)) == {"0", "2"}
    mask = np.array([False, True, True, True, True])
    assert index.search("table", top_k=10, mask=mask) == ["2"]
