# test_hard_filter.py
# value matching, the missing-value policy, alternatives vs AND, number bounds, unresolved values, relaxation

import pytest

from hybrid_rank.filtering.hard_filter import FilterIndex, normalize
from hybrid_rank.schema import FilterField

FIELDS = (
    FilterField("brand", "text", "maker", match="prefix"),
    FilterField("color", "text", "color"),
    FilterField("price", "number", "usd"),
)
LISTINGS = [
    {"brand": "Nike", "color": ["Black", "Navy Blue"], "price": 30.0},    # 0
    {"brand": "NIKE Golf", "color": ["Red"], "price": 80.0},               # 1
    {"brand": "TrainingMask", "price": 50.0},                               # 2
    {"brand": "Columbia Sportswear", "color": ["Blue/White"]},              # 3
    {"color": ["Black"]},                                                   # 4: no brand, no price
    {"brand": "The North Face", "color": ["Black Jackets"], "price": 120.0},  # 5
]


@pytest.fixture
def index():
    return FilterIndex(LISTINGS, FIELDS)


def rows(mask):
    return [i for i, keep in enumerate(mask) if keep]


def test_normalize_ignores_case_accents_punctuation_plural_and_leading_the():
    assert normalize("The North-Face Jackets") == ("north", "face", "jacket")
    assert normalize("Café") == ("cafe",)
    assert normalize("Glass") == ("glass",)  # -ss is not a plural
    assert normalize(" / ") == ()


def test_prefix_matches_names_that_start_with_the_value(index):
    assert sorted(index.resolve("brand", "nike")) == [("nike",), ("nike", "golf")]
    assert index.resolve("brand", "North Face") == [("north", "face")]
    assert index.resolve("brand", "Golf") == []  # a prefix, not a substring
    assert index.resolve("brand", "Training Mask") == [("trainingmask",)]  # spacing variants


def test_contains_matches_values_with_all_the_words(index):
    assert sorted(index.resolve("color", "blue")) == [("blue", "white"), ("navy", "blue")]
    assert index.resolve("color", "navy blue") == [("navy", "blue")]
    assert index.resolve("color", "purple") == []


def test_missing_values_are_kept_by_default_and_dropped_when_strict(index):
    keep = index.apply({"brand": ["Nike"]}, min_candidates=1)
    assert rows(keep.mask) == [0, 1, 4]  # 4 has no brand
    strict = index.apply({"brand": ["Nike"]}, missing="drop", min_candidates=1)
    assert rows(strict.mask) == [0, 1]
    assert strict.applied == {"brand": ["Nike"]} and strict.n_candidates == 2


def test_values_in_a_field_are_alternatives_and_fields_combine_with_and(index):
    either = index.apply({"color": ["red", "navy"], "brand": ["nike"]}, missing="drop", min_candidates=1)
    assert rows(either.mask) == [0, 1]
    both = index.apply({"color": ["red"], "brand": ["nike"]}, missing="drop", min_candidates=1)
    assert rows(both.mask) == [1]


def test_price_bounds_keep_unpriced_listings_unless_strict(index):
    assert rows(index.apply({"price": {"max": 60}}, min_candidates=1).mask) == [0, 2, 3, 4]  # 3, 4 unpriced
    bounded = index.apply({"price": {"min": 40, "max": 100}}, missing="drop", min_candidates=1)
    assert rows(bounded.mask) == [1, 2]
    assert index.apply({"price": {}}, min_candidates=1).mask is None


def test_unresolved_values_are_skipped_not_applied(index):
    partly = index.apply({"brand": ["Adidas"], "color": ["red"]}, missing="drop", min_candidates=1)
    assert partly.unresolved == {"brand": ["Adidas"]}
    assert partly.applied == {"color": ["red"]} and rows(partly.mask) == [1]
    # applying it anyway would leave only the listings without a brand
    nothing = index.apply({"brand": ["Adidas"]}, min_candidates=1)
    assert nothing.mask is None and nothing.n_candidates == len(LISTINGS)


def test_too_few_candidates_relaxes_the_lowest_priority_field_first(index):
    # Nike {0, 1} & red {1} & price <= 50 {0, 2} is empty: price goes first, leaving {1}
    r = index.apply({"price": {"max": 50}, "color": ["red"], "brand": ["nike"]}, missing="drop", min_candidates=1)
    assert r.relaxed == ["price"] and rows(r.mask) == [1] and r.applied == {"brand": ["nike"], "color": ["red"]}
    gone = index.apply({"brand": ["nike"], "color": ["red"]}, missing="drop", min_candidates=3)
    assert gone.relaxed == ["color", "brand"] and gone.mask is None


def test_catalog_notes_and_coverage(index):
    notes = index.catalog_notes()
    assert notes["color"].startswith("Most common values: Black,")
    assert "median" in notes["price"]
    assert index.coverage("brand") == pytest.approx(5 / 6)
    assert index.coverage("price") == pytest.approx(4 / 6)


def test_rejects_unknown_policy(index):
    with pytest.raises(ValueError):
        index.apply({"brand": ["nike"]}, missing="sometimes")
