# rerank.py
# CLI: rerank a run saved by eval.py and evaluate the result as a new run. Never re-retrieves, so it needs
# only the candidates' texts, and a depth sweep costs nothing once the pairs are scored.
#
# usage: python scripts/rerank.py stark-amazon --run rrf-bge-small-k60-d500+f.brand --reranker minilm --limit 50
#        python scripts/rerank.py stark-amazon --run rrf-bge-small-k60-d500+f.brand --reranker minilm --depth 10 20 50 100 --blend both
#        python scripts/rerank.py wands --run rrf-bge-small-k60-d500 --reranker bge-base --depth 20
#        python scripts/rerank.py stark-amazon --run rrf-bge-small-k60-d500+f.brand --reranker llm --call-llm --limit 5
#
# --limit N times the first N queries (and, for llm with --call-llm, pays for them) without logging a run.
# The llm reranker reads results/reranks/ only, unless --call-llm is given (paid; ANTHROPIC_API_KEY in .env).

import argparse
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from hybrid_rank.eval.config import EVAL_CONFIGS
from hybrid_rank.eval.run_eval import evaluate, load_queries, processed_dir
from hybrid_rank.eval.runs import SUMMARY_PATH, load_rankings, log_summary, save_run
from hybrid_rank.llm import DEFAULT_MODEL, estimate_cost, usage_line
from hybrid_rank.rerank.base import RerankRetriever, SavedRun
from hybrid_rank.rerank.cross_encoder import RERANKERS, CrossEncoderReranker, rerank_cache_dir
from hybrid_rank.rerank.llm import LLMReranker, reranks_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(EVAL_CONFIGS))
    parser.add_argument("--run", required=True, help="a saved run's name, without the split")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--reranker", choices=[*RERANKERS, "llm"], required=True)
    parser.add_argument("--depth", type=int, nargs="+", default=[20], help="how many results to rerank")
    parser.add_argument("--blend", choices=["no", "yes", "both"], default="no",
                        help="fuse the reranked order with the original order (RRF)")
    parser.add_argument("--limit", type=int, help="time the first N queries only; nothing is logged")
    parser.add_argument("--llm-model", default=DEFAULT_MODEL)
    parser.add_argument("--call-llm", action="store_true", help="allow paid API calls for uncached queries")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    config = EVAL_CONFIGS[args.dataset]
    all_queries = load_queries(args.dataset, args.split)
    rankings = load_rankings(args.dataset, args.run, args.split)
    all_queries = [q for q in all_queries if q.id in rankings]
    queries = all_queries[: args.limit]
    base = SavedRun({q.text: rankings[q.id] for q in queries})
    deepest = max(args.depth)

    df = pd.read_parquet(processed_dir(args.dataset) / "listings.parquet", columns=["id", "text"],
                         filters=[("id", "in", {lid for q in queries for lid in rankings[q.id][:deepest]})])
    texts = dict(zip(df["id"], df["text"]))
    del df
    print(f"{args.run} on {args.dataset}/{args.split}: {len(queries)} queries, {len(texts):,} candidate listings")

    if args.reranker == "llm":
        reranker = LLMReranker(texts, reranks_dir(args.dataset), model=args.llm_model)
        items = [(q.text, rankings[q.id][:d]) for d in args.depth for q in queries]
        todo = [t for d in args.depth for t in reranker.missing([(q.text, rankings[q.id][:d]) for q in queries])]
        if todo and not args.call_llm:
            raise SystemExit(f"{len(todo)} of {len(items)} rankings aren't in {reranker.cache_path}\n"
                             f"  add --call-llm to pay for them (try --limit 5 first)")
        for d in args.depth:
            new = reranker.rank_many([(q.text, rankings[q.id][:d]) for q in queries])
            if new:
                print(f"depth {d}: {usage_line(args.llm_model, new)}; errors {sum(bool(r.error) for r in new)}")
                cost = estimate_cost(args.llm_model, new)
                if args.limit and cost is not None:
                    print(f"  all {len(all_queries)} queries: ~${cost / len(new) * len(all_queries):.2f}")
        label = f"llm.{args.llm_model}"
    else:
        reranker = CrossEncoderReranker(RERANKERS[args.reranker], texts, rerank_cache_dir(args.dataset))
        pairs = [(q.text, lid) for q in queries for lid in rankings[q.id][:deepest]]
        n_todo = len(reranker.missing(pairs))
        print(f"{reranker.spec.name}: {n_todo:,} of {len(pairs):,} pairs to score")
        if n_todo:
            reranker.model  # load before timing
            start = time.perf_counter()
            scored = reranker.score(pairs, progress=not args.limit)
            seconds = time.perf_counter() - start
            rate = scored / seconds
            print(f"  {scored:,} pairs in {seconds:.0f}s on {reranker.model.device} = {rate:.0f} pairs/s")
            if args.limit:
                print(f"  all {len(all_queries)} queries at depth {deepest}: ~{len(all_queries) * deepest / rate / 60:.0f} min")
        label = f"ce.{args.reranker}"
    if args.limit:
        return

    df_judgments = pd.read_parquet(processed_dir(args.dataset) / "judgments.parquet")
    for depth in args.depth:
        for blend in {"no": [False], "yes": [True], "both": [False, True]}[args.blend]:
            system = f"{args.run}+{label}-d{depth}" + (".blend" if blend else "")
            result = evaluate(queries, df_judgments, RerankRetriever(base, reranker, depth, blend).search, config,
                              show_progress=False)
            save_run(result, args.dataset, system, args.split)
            log_summary(result, args.dataset, system, args.split, base_run=args.run)
            print(f"{system}: " + "  ".join(f"{m} {result.summary[m]:.4f}" for m in config.metrics))
    print(f"\nSummaries appended to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
