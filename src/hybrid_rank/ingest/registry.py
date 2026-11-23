# registry.py
# dataset name -> adapter class, for the scripts that need a dataset's adapter (ingest) or its declared
# catalog description and filter fields (query decomposition, hard filters)

from typing import Dict, Type

from hybrid_rank.ingest.base import DatasetAdapter
from hybrid_rank.ingest.stark import StarkAmazonAdapter
from hybrid_rank.ingest.wands import WandsAdapter

ADAPTERS: Dict[str, Type[DatasetAdapter]] = {a.name: a for a in (WandsAdapter, StarkAmazonAdapter)}
