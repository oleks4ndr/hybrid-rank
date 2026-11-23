# base.py
# Adapter interface for ingesting dataset

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hybrid_rank.schema import FilterField, Judgment, Listing, Query


class DatasetAdapter(ABC):
    name: str
    catalog_description: str = ""               # one line telling the LLM what the catalog sells
    filter_fields: Tuple[FilterField, ...] = ()  # attributes a query can hard-filter on, highest priority first

    def __init__(self, data_dir: Path = Path("data")):
        self.raw_dir = data_dir / "raw" / self.name
        self.processed_dir = data_dir / "processed" / self.name

    @abstractmethod
    def fetch_raw(self) -> None:
        """Download source files into self.raw_dir (skip if cached)"""

    @abstractmethod
    def to_common_schema(self) -> Tuple[List[Listing], List[Query], List[Judgment]]:
        """Parse self.raw_dir's files into the common Listing/Query/Judgment schema"""

    def official_split(self) -> Optional[Dict[str, Any]]:
        """The benchmark's own query split ({name: [query ids]}), or None to draw a random dev/test split"""
        return None
