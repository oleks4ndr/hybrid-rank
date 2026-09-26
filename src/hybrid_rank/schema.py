# schema.py
# Common schema a dataset adapter normalizes to.

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


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