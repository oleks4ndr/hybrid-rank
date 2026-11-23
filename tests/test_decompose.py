# test_decompose.py
# the decomposition prompt/schema, parsing and caching (against a fake client: no API calls), and the pipeline

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_rank.filtering.hard_filter import FilterIndex
from hybrid_rank.pipeline import Pipeline
from hybrid_rank.query.decompose import Decomposer, Decomposition, clean_filters
from hybrid_rank.query.prompts import output_schema, system_prompt
from hybrid_rank.schema import FilterField

FIELDS = (FilterField("brand", "text", "The maker.", match="prefix"), FilterField("price", "number", "In USD."))
NOTES = {"brand": "Most common values: Nike, Adidas."}


class FakeClient:
    """Stands in for anthropic.Anthropic(): replies[query] = (stop_reason, JSON payload)."""

    def __init__(self, replies):
        self.replies, self.calls = replies, []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        query = kwargs["messages"][0]["content"].removeprefix("Search request: ")
        stop_reason, payload = self.replies[query]
        return SimpleNamespace(
            stop_reason=stop_reason,
            content=[SimpleNamespace(type="text", text=json.dumps(payload))],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


def decomposer(tmp_path, replies=None, notes=NOTES):
    return Decomposer("toy", "A toy sports shop.", FIELDS, notes, cache_dir=tmp_path, client=FakeClient(replies or {}))


def test_prompt_is_built_from_the_declared_fields():
    prompt = system_prompt("A toy sports shop.", FIELDS, NOTES)
    assert "A toy sports shop." in prompt
    assert "- brand (text, matches catalog values that start with it): The maker. Most common values: Nike" in prompt
    assert "- price (number: min and/or max): In USD." in prompt
    schema = output_schema(FIELDS)
    filters = schema["properties"]["filters"]
    assert filters["required"] == ["brand", "price"] and filters["additionalProperties"] is False
    assert {"type": "null"} in filters["properties"]["price"]["properties"]["max"]["anyOf"]


def test_clean_filters_keeps_only_real_constraints_on_declared_fields():
    raw = {"brand": ["Nike", " ", "Nike", 3], "price": {"min": None, "max": 50, "x": 1}, "color": ["red"]}
    assert clean_filters(raw, FIELDS) == {"brand": ["Nike"], "price": {"max": 50.0}}
    assert clean_filters({"brand": [], "price": {"min": None, "max": True}}, FIELDS) == {}


def test_decompose_parses_and_falls_back_on_refusal(tmp_path):
    replies = {
        "nike running shoes under $50": ("end_turn", {"semantic_query": "nike running shoes",
                                                      "filters": {"brand": ["Nike"], "price": {"min": None, "max": 50}}}),
        "something odd": ("refusal", {}),
    }
    d = decomposer(tmp_path, replies)
    out = d.decompose_many(list(replies), workers=2)
    nike = out["nike running shoes under $50"]
    assert nike.semantic_query == "nike running shoes" and nike.filters == {"brand": ["Nike"], "price": {"max": 50.0}}
    odd = out["something odd"]
    assert odd.error == "refusal" and odd.semantic_query == "something odd" and odd.filters == {}
    call = d.client.calls[0]
    assert call["output_config"]["format"]["schema"] == output_schema(FIELDS) and call["system"] == d.system


def test_results_are_cached_and_never_requested_twice(tmp_path):
    replies = {"red ball": ("end_turn", {"semantic_query": "red ball", "filters": {"brand": [], "price": {"min": None, "max": None}}})}
    first = decomposer(tmp_path, replies)
    first.decompose_many(["red ball"])
    with open(first.cache_path, "a") as f:
        f.write('{"query": "cut off mid-wri')  # a run killed while writing
    again = decomposer(tmp_path)  # no replies: any API call would raise KeyError
    assert again.cache_path == first.cache_path
    assert again.decompose_many(["red ball"])["red ball"].filters == {}
    assert again.client.calls == [] and again.missing(["red ball", "blue ball"]) == ["blue ball"]
    # a different prompt is a different cache file, so old answers never mix with new ones
    assert decomposer(tmp_path, notes={}).cache_path != first.cache_path


class Recorder:
    def __init__(self):
        self.calls = []

    def search(self, query_text, top_k=50, mask=None):
        self.calls.append((query_text, None if mask is None else np.flatnonzero(mask).tolist()))
        return []


def test_pipeline_switches():
    index = FilterIndex([{"brand": "Nike", "price": 30.0}, {"brand": "Adidas", "price": 80.0}, {}], FIELDS)
    query = "nike shoes under $50"
    decs = {query: Decomposition(query, "nike shoes", {"brand": ["Nike"], "price": {"max": 50.0}})}
    r = Recorder()

    Pipeline(r, decs).search(query, 10)  # nothing switched on: the plain retriever
    Pipeline(r, decs, index, ["brand"], min_candidates=1).search(query, 10)
    Pipeline(r, decs, index, ["brand", "price"], rewrite=True, missing="drop", min_candidates=1).search(query, 10)
    Pipeline(r, decs, index, ["brand"], min_candidates=1).search(query, 10, mask=np.array([False, True, True]))
    assert r.calls == [
        (query, None),
        (query, [0, 2]),         # listing 2 has no brand: kept
        ("nike shoes", [0]),
        (query, [2]),            # a caller's mask is ANDed in
    ]
    with pytest.raises(KeyError, match="decompose.py"):
        Pipeline(r, decs).search("never decomposed", 10)
