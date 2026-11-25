# runs.py
# saved eval runs: per-query results and rankings under results/runs/, one summary line per run in
# results/summary.jsonl. Shared by scripts/eval.py and scripts/rerank.py.

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from hybrid_rank.eval.run_eval import EvalResult

RESULTS_DIR = Path("results")
SUMMARY_PATH = RESULTS_DIR / "summary.jsonl"


def runs_dir(dataset: str) -> Path:
    return RESULTS_DIR / "runs" / dataset


def git_commit() -> str:
    """Short sha, marked -dirty if tracked files had uncommitted changes when the numbers were produced.
    results/ doesn't count: the summary log and LLM caches there are tracked outputs, so otherwise every run
    would dirty the next one."""
    def git(*cmd: str) -> str:
        return subprocess.run(["git", *cmd], capture_output=True, text=True).stdout.strip()
    sha = git("rev-parse", "--short", "HEAD") or "unknown"
    changes = git("status", "--porcelain", "--untracked-files=no", "--", ".", f":!{RESULTS_DIR}")
    return sha + ("-dirty" if changes else "")


def save_run(result: EvalResult, dataset: str, system: str, split: str) -> Path:
    run_dir = runs_dir(dataset)
    run_dir.mkdir(parents=True, exist_ok=True)
    rankings = pd.DataFrame(
        [(qid, rank, lid) for qid, ranked in result.rankings.items() for rank, lid in enumerate(ranked, start=1)],
        columns=["query_id", "rank", "listing_id"],
    )
    rankings.to_parquet(run_dir / f"{system}_{split}.rankings.parquet", index=False)
    result.per_query.to_parquet(run_dir / f"{system}_{split}.per_query.parquet", index=False)
    return run_dir


def load_rankings(dataset: str, system: str, split: str) -> Dict[str, List[str]]:
    """query_id -> best-first listing ids of a saved run."""
    path = runs_dir(dataset) / f"{system}_{split}.rankings.parquet"
    if not path.exists():
        raise SystemExit(f"no saved run {path} -- run scripts/eval.py first")
    df = pd.read_parquet(path).sort_values(["query_id", "rank"])
    return df.groupby("query_id")["listing_id"].apply(list).to_dict()


def log_summary(result: EvalResult, dataset: str, system: str, split: str, **provenance: str) -> None:
    record = {
        "dataset": dataset,
        "system": system,
        "split": split,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(),
        **provenance,
        **{k: round(v, 4) for k, v in result.summary.items()},
    }
    with open(SUMMARY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
