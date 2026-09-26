# base.py
# Adapter interface for ingesting dataset

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

from hybrid_rank.schema import Judgment, Listing, Query


class DatasetAdapter(ABC):
    name: str

    def __init__(self, data_dir: Path = Path("data")):
        self.raw_dir = data_dir / "raw" / self.name
        self.processed_dir = data_dir / "processed" / self.name

    @abstractmethod
    def fetch_raw(self) -> None:
        """Download source files into self.raw_dir (skip if cached)"""

    @abstractmethod
    def to_common_schema(self) -> Tuple[List[Listing], List[Query], List[Judgment]]:
        """Parse self.raw_dir's files into the common Listing/Query/Judgment schema"""