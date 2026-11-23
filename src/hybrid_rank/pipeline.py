# pipeline.py
# wires it all: decompose -> hard filter -> retrieve (BM25, dense or RRF over both).
# Each stage is a switch, so every ablation is this one class with different settings:
#   rewrite        search with the LLM's semantic_query instead of the original request
#   filter_fields  which decomposed constraints become hard filters (none = no filtering)
#   missing        "keep" listings that lack a filtered field (default) or "drop" them (strict)
# Decompositions come precomputed (scripts/decompose.py), so a search never calls the LLM.

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import numpy as np

from hybrid_rank.filtering.hard_filter import MIN_CANDIDATES, FilterIndex, FilterResult
from hybrid_rank.query.decompose import Decomposition
from hybrid_rank.retrieval.base import Retriever


@dataclass
class Plan:
    text: str                        # what the retriever searches with
    filters: Optional[FilterResult]  # None when filtering is off


class Pipeline:
    def __init__(
        self,
        retriever: Retriever,
        decompositions: Mapping[str, Decomposition],
        filter_index: Optional[FilterIndex] = None,
        filter_fields: Sequence[str] = (),
        rewrite: bool = False,
        missing: str = "keep",
        min_candidates: int = MIN_CANDIDATES,
    ):
        if filter_fields and filter_index is None:
            raise ValueError("filter_fields needs a filter_index")
        self.retriever = retriever
        self.decompositions = decompositions
        self.filter_index = filter_index
        self.filter_fields = set(filter_fields)
        self.rewrite = rewrite
        self.missing = missing
        self.min_candidates = min_candidates

    def plan(self, query_text: str) -> Plan:
        d = self.decompositions.get(query_text)
        if d is None:
            raise KeyError(f"no decomposition for {query_text!r} -- run scripts/decompose.py for this split first")
        text = d.semantic_query if self.rewrite else query_text
        if not self.filter_fields:
            return Plan(text, None)
        assert self.filter_index is not None
        constraints = {name: c for name, c in d.filters.items() if name in self.filter_fields}
        return Plan(text, self.filter_index.apply(constraints, self.missing, self.min_candidates))

    def search(self, query_text: str, top_k: int = 50, mask: Optional[np.ndarray] = None) -> List[str]:
        plan = self.plan(query_text)
        filter_mask = plan.filters.mask if plan.filters is not None else None
        if filter_mask is not None:
            mask = filter_mask if mask is None else filter_mask & mask
        return self.retriever.search(plan.text, top_k, mask)
