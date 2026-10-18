# splits.py
# fixed, committed dev/test split of query ids -- tune on dev, report on test

import json
import random
from pathlib import Path
from typing import Dict, List

SPLITS_DIR = Path("data/splits")


def split_path(dataset: str) -> Path:
    return SPLITS_DIR / f"{dataset}.json"


def make_split(query_ids: List[str], dev_frac: float = 0.2, seed: int = 0) -> Dict:
    shuffled = sorted(query_ids)
    random.Random(seed).shuffle(shuffled)
    n_dev = round(len(shuffled) * dev_frac)
    return {"seed": seed, "dev": sorted(shuffled[:n_dev]), "test": sorted(shuffled[n_dev:])}


def save_split_if_missing(dataset: str, query_ids: List[str]) -> bool:
    """Never overwrites: once committed, the split stays fixed for the whole project."""
    path = split_path(dataset)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_split(query_ids), indent=2))
    return True


def load_split(dataset: str) -> Dict[str, List[str]]:
    return json.loads(split_path(dataset).read_text())
