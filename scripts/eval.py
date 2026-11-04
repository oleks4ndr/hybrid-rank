# eval.py
# CLI: evaluate one retriever on a dataset split, save its run, append the summary to results/summary.jsonl
#
# usage: python scripts/eval.py wands --retriever bm25 --split test
#        python scripts/eval.py wands --retriever dense --model minilm --split dev
#        python scripts/eval.py wands --retriever rrf --model minilm --split dev

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hybrid_rank.eval.gain_schemes import GAIN_SCHEMES
from hybrid_rank.eval.run_eval import EvalResult, evaluate, load_processed
from hybrid_rank.retrieval.bm25 import BM25Index
from hybrid_rank.retrieval.embed import ENCODERS, DenseIndex, embeddings_dir
from hybrid_rank.retrieval.rrf import DEFAULT_DEPTH, DEFAULT_K, RRFRetriever

RESULTS_DIR = Path("results")
SUMMARY_PATH = RESULTS_DIR / "summary.jsonl"


def build_retriever(args, listings):
    if args.retriever == "bm25":
        return "bm25", BM25Index(listings)
    index = DenseIndex(listings, ENCODERS[args.model], embeddings_dir(args.dataset))
    source = "loaded from" if index.loaded_from_cache else "encoded and saved to"
    print(f"  {index.spec.name} on {index.model.device}, embeddings {source} {index.cache_path}")
    if args.retriever == "dense":
        return f"dense-{args.model}", index
    retriever = RRFRetriever([BM25Index(listings), index], k=args.rrf_k, depth=args.depth)
    return f"rrf-{args.model}-k{args.rrf_k}-d{args.depth}", retriever


def git_commit() -> str:
    """Short sha, marked -dirty if tracked files had uncommitted changes when the numbers were produced.
    The summary log doesn't count: it's tracked, so otherwise every run would dirty the next one."""
    def git(*cmd: str) -> str:
        return subprocess.run(["git", *cmd], capture_output=True, text=True).stdout.strip()
    sha = git("rev-parse", "--short", "HEAD") or "unknown"
    changes = git("status", "--porcelain", "--untracked-files=no", "--", ".", f":!{SUMMARY_PATH}")
    return sha + ("-dirty" if changes else "")


def save_run(result: EvalResult, dataset: str, system: str, split: str) -> Path:
    run_dir = RESULTS_DIR / "runs" / dataset
    run_dir.mkdir(parents=True, exist_ok=True)
    rankings = pd.DataFrame(
        [(qid, rank, lid) for qid, ranked in result.rankings.items() for rank, lid in enumerate(ranked, start=1)],
        columns=["query_id", "rank", "listing_id"],
    )
    rankings.to_parquet(run_dir / f"{system}_{split}.rankings.parquet", index=False)
    result.per_query.to_parquet(run_dir / f"{system}_{split}.per_query.parquet", index=False)
    return run_dir


def log_summary(result: EvalResult, dataset: str, system: str, split: str) -> None:
    record = {
        "dataset": dataset,
        "system": system,
        "split": split,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(),
        **{k: round(v, 4) for k, v in result.summary.items()},
    }
    with open(SUMMARY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(GAIN_SCHEMES))
    parser.add_argument("--retriever", choices=["bm25", "dense", "rrf"], required=True)
    parser.add_argument("--model", choices=list(ENCODERS), default="minilm", help="dense and rrf only")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_K, help="rrf only")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="rrf only: candidates from each retriever")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    args = parser.parse_args()

    listings, queries, df_judgments = load_processed(args.dataset, split=args.split)
    print(f"Building {args.retriever} index over {len(listings):,} listings ...")
    system, retriever = build_retriever(args, listings)

    result = evaluate(queries, df_judgments, search_fn=retriever.search, gain=GAIN_SCHEMES[args.dataset])
    print(f"\n{system} on {args.dataset}/{args.split}:")
    for k, v in result.summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    run_dir = save_run(result, args.dataset, system, args.split)
    log_summary(result, args.dataset, system, args.split)
    print(f"\nRun saved to {run_dir}/, summary appended to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
