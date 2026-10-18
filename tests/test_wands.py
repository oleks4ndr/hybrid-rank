# test_wands.py
# regressions for the WANDS adapter's parsing bugs

import pandas as pd

from hybrid_rank.eval.splits import make_split
from hybrid_rank.ingest.wands import WandsAdapter, _dedupe_labels


def test_missing_fields_do_not_become_nan_text():
    row = pd.Series({
        "product_id": 1,
        "product_name": "solid wood platform bed",
        "product_class": float("nan"),
        "category hierarchy": "Furniture / Bedroom Furniture / Beds",
        "product_description": float("nan"),
        "product_features": "color:walnut",
    })
    listing = WandsAdapter._to_listing(row)
    assert listing.text == "solid wood platform bed"
    assert listing.filters["category_hierarchy"] == "Furniture / Bedroom Furniture / Beds"
    assert "product_class" not in listing.filters


def test_dedupe_labels_keeps_highest():
    df = pd.DataFrame({
        "query_id": [0, 0, 0, 1],
        "product_id": [10, 10, 11, 10],
        "label": ["Exact", "Partial", "Irrelevant", "Partial"],
    })
    deduped = _dedupe_labels(df).set_index(["query_id", "product_id"])["label"]
    assert len(deduped) == 3
    assert deduped[(0, 10)] == "Exact"


def test_split_is_deterministic_and_disjoint():
    ids = [str(i) for i in range(480)]
    split = make_split(ids, dev_frac=0.2, seed=0)
    assert split == make_split(list(reversed(ids)), dev_frac=0.2, seed=0)
    assert len(split["dev"]) == 96 and len(split["test"]) == 384
    assert set(split["dev"]).isdisjoint(split["test"])
