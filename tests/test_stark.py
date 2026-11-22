# test_stark.py
# STaRK-Amazon adapter: document text must match stark_qa's get_doc_info, and the loader must stay safe

import io
import os
import pickle

import pandas as pd
import pytest

from hybrid_rank.eval.splits import load_split, save_split_if_missing
from hybrid_rank.ingest.stark import (
    StarkAmazonAdapter,
    _DataOnlyUnpickler,
    doc_text,
    is_product,
    node_filters,
    parse_price,
    parse_qa,
)

NAN = float("nan")


def product(**overrides):
    node = {
        "title": "Chess Strategy Guide",
        "brand": "The House of Staunton",
        "details": {"product_dimensions": "8 x 5 x 1 inches ; 12 ounces"},
        "description": ["A guide ", "to openings."],
        "feature": ["Covers the Old Indian", "", "ASIN: B000123"],
        "review": [
            {"summary": "ok", "reviewText": "fine", "vote": NAN},
            {"summary": "great", "reviewText": "best book", "vote": "1,204"},
            {"summary": "good", "reviewText": "helpful", "vote": "3"},
        ],
        "qa": [{"question": "Hardcover?", "answer": "Yes"}],
        "category": ["chess", "board games"],
        "color": ["black"],
        "price": "$24.99",
    }
    node.update(overrides)
    return node


def test_doc_text_matches_stark_format():
    # hand-derived from stark_qa AmazonSKB.get_doc_info: reviews by votes (1,204 > 3 > none) but
    # numbered by original position; empty and ASIN features dropped yet keep their numbering
    assert doc_text(product()) == (
        "- product: Chess Strategy Guide\n"
        "- brand: The House of Staunton\n"
        "- dimensions: 8 x 5 x 1 inches\n- weight: 12 ounces\n"
        "- description: A guide  to openings.\n"
        "- features: \n#1: Covers the Old Indian\n"
        '- reviews: \n#2:\nsummary: great\ntext: "best book"\n'
        '#3:\nsummary: good\ntext: "helpful"\n'
        '#1:\nsummary: ok\ntext: "fine"\n'
        '- Q&A: \n#1:\nquestion: "Hardcover?"\nanswer: "Yes"\n'
    )


def test_doc_text_skips_missing_sections():
    node = {"title": "Tent", "details": NAN, "description": [], "feature": [], "review": [], "qa": []}
    assert doc_text(node) == "- product: Tent\n"


def test_doc_text_caps_reviews_like_stark():
    # the original breaks once its loop index passes max_entries, so it keeps max_entries + 2
    reviews = [{"summary": str(i), "reviewText": "x", "vote": NAN} for i in range(10)]
    text = doc_text(product(review=reviews, qa=[]), max_entries=3)
    assert text.count("summary:") == 5


def test_filters_and_product_detection():
    assert node_filters(product()) == {
        "brand": "The House of Staunton", "category": ["chess", "board games"], "color": ["black"], "price": 24.99,
    }
    assert node_filters({"title": "x"}) == {}
    assert is_product(product())
    assert not is_product({"brand_name": "Coleman"})


def test_parse_price():
    assert parse_price("$1,299.00") == 1299.0
    assert parse_price("$10.99 - $15.99") == 10.99
    assert parse_price("") is None
    assert parse_price(NAN) is None


def test_parse_qa_prefixes_ids_and_reads_answer_lists_safely():
    df = pd.DataFrame({"id": [0, 1], "query": ["a", "b"], "answer_ids": ["[5, 7]", "[9]"]})
    queries, judgments = parse_qa(df, "h", "human")
    assert [q.id for q in queries] == ["h0", "h1"]
    assert {(j.query_id, j.listing_id, j.label) for j in judgments} == {
        ("h0", "5", "relevant"), ("h0", "7", "relevant"), ("h1", "9", "relevant"),
    }
    with pytest.raises(ValueError):
        parse_qa(pd.DataFrame({"id": [0], "query": ["a"], "answer_ids": ["__import__('os')"]}), "", "x")


class _Exploit:
    def __reduce__(self):
        return (os.getcwd, ())


def test_unpickler_loads_data_but_refuses_callables():
    data = {0: {"title": "x", "review": [{"vote": NAN}]}}
    assert _DataOnlyUnpickler(io.BytesIO(pickle.dumps(data))).load()[0]["title"] == "x"
    with pytest.raises(pickle.UnpicklingError, match="posix.getcwd|os.getcwd"):
        _DataOnlyUnpickler(io.BytesIO(pickle.dumps(_Exploit()))).load()


def test_official_split_maps_val_to_dev(tmp_path, monkeypatch):
    adapter = StarkAmazonAdapter(data_dir=tmp_path)
    split_dir = adapter.raw_dir / "qa/amazon/split"
    split_dir.mkdir(parents=True)
    for name, ids in {"train": "3\n1", "val": "4", "test": "10\n2", "test-0.1": "2"}.items():
        (split_dir / f"{name}.index").write_text(ids + "\n")
    human_csv = adapter.raw_dir / "qa/amazon/stark_qa/stark_qa_human_generated_eval.csv"
    human_csv.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 0], "query": ["a", "b"], "answer_ids": ["[1]", "[2]"]}).to_csv(human_csv, index=False)

    split = adapter.official_split()
    assert split["train"] == ["1", "3"] and split["dev"] == ["4"] and split["test"] == ["2", "10"]
    assert split["test-0.1"] == ["2"] and split["human"] == ["h0", "h1"]

    # saved verbatim, and ids the queries don't contain are rejected
    monkeypatch.setattr("hybrid_rank.eval.splits.SPLITS_DIR", tmp_path / "splits")
    all_ids = ["1", "2", "3", "4", "10", "h0", "h1"]
    assert save_split_if_missing("stark-amazon", all_ids, split)
    assert load_split("stark-amazon") == split
    with pytest.raises(ValueError, match="unknown query ids"):
        save_split_if_missing("other", ["1"], split)
