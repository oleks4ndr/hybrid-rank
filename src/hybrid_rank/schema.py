# schema.py
# Common schema a dataset adapter normalizes to.

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

FIELD_KINDS = ("text", "number")
MATCH_MODES = ("contains", "prefix")


@dataclass(frozen=True)
class FilterField:
    """An attribute in Listing.filters that a query can constrain. Adapters declare these; the decomposition
    prompt and the hard filters are built from them, so nothing downstream is dataset-specific."""

    name: str                 # key in Listing.filters; its values are a string, a list of strings, or a number
    kind: str                 # "text" or "number"
    description: str          # what the field means in this catalog -- the LLM reads this
    match: str = "contains"   # text only: "contains" = every word of the query value appears in the catalog
                              # value; "prefix" = the catalog value starts with it (names: brands, makers)

    def __post_init__(self):
        if self.kind not in FIELD_KINDS:
            raise ValueError(f"kind must be one of {FIELD_KINDS}, got {self.kind!r}")
        if self.match not in MATCH_MODES:
            raise ValueError(f"match must be one of {MATCH_MODES}, got {self.match!r}")


@dataclass
class Listing:
    id: str
    text: str                                              # concatenated text for BM25 / embeddings
    filters: Dict[str, Any] = field(default_factory=dict)  # structured, filterable attributes
    source: Dict[str, Any] = field(default_factory=dict)   # original raw row

    def to_record(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "filters": json.dumps(self.filters, default=str),
            "source": json.dumps(self.source, default=str),
        }

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "Listing":
        return cls(
            id=record["id"],
            text=record["text"],
            filters=json.loads(record["filters"]) if record.get("filters") else {},
            source=json.loads(record["source"]) if record.get("source") else {},
        )


@dataclass
class Query:
    id: str
    text: str
    category: Optional[str] = None

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "Query":
        return cls(**record)


@dataclass
class Judgment:
    query_id: str
    listing_id: str
    label: str

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "Judgment":
        return cls(**record)