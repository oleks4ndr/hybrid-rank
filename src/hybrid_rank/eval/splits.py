# splits.py
# fixed, committed query split -- tune on dev, report on test. A benchmark's own split is used as-is
# when it has one (STaRK); otherwise a seeded random dev/test split is drawn once (WANDS).

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

SPLITS_DIR = Path("data/splits")
RESERVED_KEYS = ("seed", "source")  # metadata, not splits


def split_path(dataset: str) -> Path:
    return SPLITS_DIR / f"{dataset}.json"


def make_split(query_ids: List[str], dev_frac: float = 0.2, seed: int = 0) -> Dict:
    shuffled = sorted(query_ids)
    random.Random(seed).shuffle(shuffled)
    n_dev = round(len(shuffled) * dev_frac)
    return {"seed": seed, "dev": sorted(shuffled[:n_dev]), "test": sorted(shuffled[n_dev:])}


def save_split_if_missing(dataset: str, query_ids: List[str], official: Optional[Dict[str, Any]] = None) -> bool:
    """Never overwrites: once committed, the split stays fixed for the whole project."""
    path = split_path(dataset)
    if path.exists():
        return False
    split = official if official is not None else make_split(query_ids)
    known = set(query_ids)
    for name in split_names(split):
        unknown = [qid for qid in split[name] if qid not in known]
        if unknown:
            raise ValueError(f"split {name!r} names {len(unknown)} unknown query ids, e.g. {unknown[:3]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2))
    return True


def split_names(split: Dict[str, Any]) -> List[str]:
    return [k for k in split if k not in RESERVED_KEYS]


def load_split(dataset: str) -> Dict[str, List[str]]:
    return json.loads(split_path(dataset).read_text())
