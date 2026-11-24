# test_rerank.py
# reranking stage (head/tail, blend), cross-encoder score cache, LLM ranking parsing and cache -- fakes only

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_rank.rerank.base import RerankRetriever, SavedRun
from hybrid_rank.rerank.cross_encoder import CrossEncoderReranker, RerankerSpec
from hybrid_rank.rerank.llm import LLMReranker, apply_ranking

RUN = SavedRun({"q": ["a", "b", "c", "d"], "one": ["a"], "none": []})
TEXTS = {"a": "aaa", "b": "b", "c": "cc", "d": "dddd"}


class Reverse:
    def rerank(self, query_text, listing_ids):
        return listing_ids[::-1]


class Identity:
    def rerank(self, query_text, listing_ids):
        return list(listing_ids)


def test_only_the_head_is_reordered_and_top_k_respected():
    r = RerankRetriever(RUN, Reverse(), depth=3)
    assert r.search("q", 10) == ["c", "b", "a", "d"]
    assert r.search("q", 2) == ["c", "b"]  # candidates come from the full depth even when top_k is smaller
    assert RerankRetriever(RUN, Reverse(), depth=50).search("q", 10) == ["d", "c", "b", "a"]
    assert r.search("one", 10) == ["a"] and r.search("none", 10) == []


def test_identity_is_a_no_op_with_and_without_blend():
    for blend in (False, True):
        assert RerankRetriever(RUN, Identity(), depth=3, blend=blend).search("q", 10) == ["a", "b", "c", "d"]


def test_blend_fuses_both_orders():
    class LastFirst:
        def rerank(self, query_text, listing_ids):
            return [listing_ids[-1], *listing_ids[:-1]]

    # ranks a (1,2) b (2,3) c (3,1): a 1/61+1/62 > c 1/63+1/61 > b 1/62+1/63
    assert RerankRetriever(RUN, LastFirst(), depth=3, blend=True).search("q", 10) == ["a", "c", "b", "d"]


def test_mask_reaches_the_retriever_and_saved_runs_refuse_one():
    class Recorder:
        def search(self, query_text, top_k=50, mask=None):
            self.mask = mask
            return ["a", "b"]

    inner, mask = Recorder(), np.array([True, False])
    RerankRetriever(inner, Identity(), depth=2).search("q", 2, mask)
    assert inner.mask is mask
    with pytest.raises(ValueError):
        RUN.search("q", 2, mask)


class LengthModel:
    """Scores a pair by its listing's text length."""

    def __init__(self):
        self.n_scored = 0

    def predict(self, pairs, **kwargs):
        self.n_scored += len(pairs)
        return np.array([len(text) for _, text in pairs], dtype=np.float32)


def test_cross_encoder_sorts_by_score_and_caches_by_content(tmp_path):
    model = LengthModel()
    ce = CrossEncoderReranker(RerankerSpec("toy/model"), TEXTS, tmp_path, model)
    assert ce.rerank("q", ["b", "a", "c"]) == ["a", "c", "b"]
    assert ce.rerank("q", ["b", "a", "c"]) == ["a", "c", "b"] and model.n_scored == 3
    again = CrossEncoderReranker(RerankerSpec("toy/model"), TEXTS, tmp_path, LengthModel())
    assert again.score([("q", "a"), ("q", "d")]) == 1  # a was on disk, only d is new
    changed = CrossEncoderReranker(RerankerSpec("toy/model"), {**TEXTS, "a": "new text"}, tmp_path, LengthModel())
    assert len(changed.missing([("q", "a"), ("other query", "b"), ("q", "b")])) == 2


def test_cross_encoder_ties_keep_the_retrieval_order(tmp_path):
    ce = CrossEncoderReranker(RerankerSpec("toy/model"), {"x": "same", "y": "same", "z": "same"}, tmp_path, LengthModel())
    assert ce.rerank("q", ["y", "z", "x"]) == ["y", "z", "x"]


def test_apply_ranking_survives_malformed_answers():
    ids = ["a", "b", "c"]
    assert apply_ranking([3, 1, 2], ids) == ["c", "a", "b"]
    assert apply_ranking([2, 2, 9, 0, -1], ids) == ["b", "a", "c"]  # repeats and unknown numbers ignored
    assert apply_ranking([], ids) == ids and apply_ranking(None, ids) == ids


class FakeClient:
    def __init__(self, reply=None, stop_reason="end_turn"):
        self.reply, self.stop_reason, self.calls = reply, stop_reason, []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="text", text=json.dumps(self.reply))],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )


def test_llm_reranker_caches_and_never_calls_from_rerank(tmp_path):
    client = FakeClient({"ranking": [2, 1, 3]})
    llm = LLMReranker(TEXTS, tmp_path, client=client)
    with pytest.raises(KeyError, match="--call-llm"):
        llm.rerank("q", ["a", "b", "c"])
    assert len(llm.rank_many([("q", ["a", "b", "c"])])) == 1
    assert llm.rerank("q", ["a", "b", "c"]) == ["b", "a", "c"]
    again = LLMReranker(TEXTS, tmp_path, client=FakeClient())
    assert again.rank_many([("q", ["a", "b", "c"])]) == [] and again.rerank("q", ["a", "b", "c"]) == ["b", "a", "c"]
    assert again.missing([("q", ["b", "a", "c"])])  # other candidates or another order is another request
    assert "[1] aaa" in client.calls[0]["messages"][0]["content"]


def test_llm_refusal_keeps_the_original_order(tmp_path):
    llm = LLMReranker(TEXTS, tmp_path, client=FakeClient({}, stop_reason="refusal"))
    new = llm.rank_many([("q", ["a", "b"])])
    assert new[0].error == "refusal" and llm.rerank("q", ["a", "b"]) == ["a", "b"]
