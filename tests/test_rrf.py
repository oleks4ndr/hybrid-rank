# test_rrf.py
# hand-computed fusion cases and the retriever's depth / mask contract

import numpy as np

from hybrid_rank.retrieval.rrf import RRFRetriever, reciprocal_rank_fusion


class StubRetriever:
    def __init__(self, ranking):
        self.ranking = ranking
        self.calls = []

    def search(self, query_text, top_k=50, mask=None):
        self.calls.append((query_text, top_k, mask))
        return self.ranking[:top_k]


def test_fusion_hand_computed():
    # k=60: b = 1/62 + 1/61, c = 1/63 + 1/62, a = 1/61, d = 1/63
    assert reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60) == ["b", "c", "a", "d"]


def test_k_controls_top_rank_vs_consensus():
    rankings = [["x", "a", "b", "m"], ["y", "c", "d", "m"]]
    # m = 2/(k+4) vs x = 1/(k+1): a single first place wins at small k, agreement wins at large k
    assert reciprocal_rank_fusion(rankings, k=1)[0] == "x"
    assert reciprocal_rank_fusion(rankings, k=60)[0] == "m"


def test_ties_go_to_the_earlier_ranking():
    assert reciprocal_rank_fusion([["a"], ["b"]]) == ["a", "b"]
    assert reciprocal_rank_fusion([["b"], ["a"]]) == ["b", "a"]


def test_single_ranking_keeps_its_order():
    assert reciprocal_rank_fusion([["c", "a", "b"]]) == ["c", "a", "b"]


def test_empty_rankings():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_retriever_truncates_to_top_k():
    rrf = RRFRetriever([StubRetriever(["a", "b", "c"]), StubRetriever(["b", "c", "d"])], k=60, depth=3)
    assert rrf.search("q", top_k=2) == ["b", "c"]


def test_depth_limits_candidates_from_each_retriever():
    rrf = RRFRetriever([StubRetriever(["a", "b", "c"]), StubRetriever(["d", "e", "f"])], depth=1)
    assert rrf.search("q", top_k=10) == ["a", "d"]


def test_mask_and_depth_reach_every_retriever():
    stubs = [StubRetriever(["a"]), StubRetriever(["b"])]
    mask = np.array([True, False])
    RRFRetriever(stubs, depth=7).search("q", top_k=2, mask=mask)
    for stub in stubs:
        query_text, top_k, passed_mask = stub.calls[0]
        assert (query_text, top_k) == ("q", 7)
        assert passed_mask is mask
