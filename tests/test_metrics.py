# test_metrics.py
# hand-computed cases -- the paper's numbers depend on these

import math

import pytest

from hybrid_rank.eval.config import EVAL_CONFIGS
from hybrid_rank.eval.metrics import (
    condensed_ndcg_at_k,
    hit_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    unlabelled_at_k,
)

IDCG_1_HALF = 1.0 + 0.5 / math.log2(3)  # ideal DCG for gains {1.0, 0.5}


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a": 1.0, "b": 0.5, "c": 0.0}, k=3) == pytest.approx(1.0)


def test_ndcg_hand_computed():
    # b at rank 1 (0.5/log2 2), unjudged x at rank 2, a at rank 3 (1.0/log2 4)
    dcg = 0.5 + 1.0 / 2
    assert ndcg_at_k(["b", "x", "a"], {"a": 1.0, "b": 0.5}, k=3) == pytest.approx(dcg / IDCG_1_HALF)


def test_ndcg_truncates_at_k():
    # a at rank 3 falls outside k=2
    assert ndcg_at_k(["b", "x", "a"], {"a": 1.0, "b": 0.5}, k=2) == pytest.approx(0.5 / IDCG_1_HALF)


def test_ndcg_ideal_uses_all_judged_items_even_if_not_retrieved():
    # only a is retrieved; the ideal still counts b
    assert ndcg_at_k(["a"], {"a": 1.0, "b": 0.5}, k=10) == pytest.approx(1.0 / IDCG_1_HALF)


def test_ndcg_no_relevant_items_is_zero():
    assert ndcg_at_k(["a", "b"], {"a": 0.0}, k=10) == 0.0


def test_precision_counts_short_lists_as_misses():
    assert precision_at_k(["a", "x", "b", "y", "z"], {"a", "b"}, k=5) == pytest.approx(0.4)
    assert precision_at_k(["a"], {"a"}, k=5) == pytest.approx(0.2)


def test_mrr():
    assert mrr(["x", "y", "a", "b"], {"a", "b"}) == pytest.approx(1 / 3)
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_hit_at_k():
    assert hit_at_k(["x", "a"], {"a"}, k=1) == 0.0
    assert hit_at_k(["x", "a"], {"a"}, k=2) == 1.0
    assert hit_at_k([], {"a"}, k=5) == 0.0


def test_eval_configs_keep_wands_metrics_and_match_stark_paper():
    wands = EVAL_CONFIGS["wands"]
    assert wands.metrics == ["nDCG@10", "nDCG@50", "P@5", "MRR", "R@50", "R@100", "cnDCG@10", "unlabelled@10"]
    stark = EVAL_CONFIGS["stark-amazon"]
    assert {"Hit@1", "Hit@5", "R@20", "MRR"} <= set(stark.metrics)
    assert not any(m.startswith(("cnDCG", "unlabelled")) for m in stark.metrics)
    assert stark.depth == 100


def test_recall():
    relevant = {"a", "b", "c", "d"}
    assert recall_at_k(["a", "x", "b"], relevant, k=2) == pytest.approx(0.25)
    assert recall_at_k(["a", "x", "b"], relevant, k=3) == pytest.approx(0.5)


def test_recall_undefined_without_relevant_items():
    assert math.isnan(recall_at_k(["a"], set(), k=10))


def test_condensed_ndcg_skips_unlabelled_but_keeps_labelled_irrelevant():
    relevance = {"a": 1.0, "b": 0.5, "z": 0.0}
    ranked = ["x", "z", "y", "b", "a"]
    # x, y unlabelled -> dropped; z labelled irrelevant -> kept, so the list scored is [z, b, a]
    dcg = 0.0 + 0.5 / math.log2(3) + 1.0 / 2
    assert condensed_ndcg_at_k(ranked, relevance, k=3) == pytest.approx(dcg / IDCG_1_HALF)
    # plain nDCG only sees [x, z, y] in its top 3, all worth 0
    assert ndcg_at_k(ranked, relevance, k=3) == 0.0


def test_unlabelled_share():
    relevance = {"a": 1.0, "z": 0.0}
    assert unlabelled_at_k(["a", "x", "z", "y"], relevance, k=4) == pytest.approx(0.5)
    assert unlabelled_at_k(["x"], relevance, k=10) == pytest.approx(1.0)  # share of what was returned
    assert math.isnan(unlabelled_at_k([], relevance, k=10))
