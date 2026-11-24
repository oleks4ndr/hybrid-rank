# eval.py
# CLI: evaluate one retriever on a dataset split, save its run, append the summary to results/summary.jsonl
#
# usage: python scripts/eval.py wands --retriever bm25 --split test
#        python scripts/eval.py wands --retriever dense --model minilm --split dev
#        python scripts/eval.py wands --retriever rrf --model minilm --split dev
#        python scripts/eval.py stark-amazon --retriever bm25 --split dev   # splits: dev, test, test-0.1, human
#
# with LLM query decomposition (run scripts/decompose.py for the split first):
#        python scripts/eval.py stark-amazon --retriever rrf --model bge-small --filters brand
#        python scripts/eval.py stark-amazon --retriever rrf --model bge-small --filters all --rewrite
#        python scripts/eval.py wands --retriever rrf --model bge-small --filters product_class --strict

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from hybrid_rank.eval.config import EVAL_CONFIGS, EvalConfig
from hybrid_rank.eval.run_eval import evaluate, load_processed
from hybrid_rank.eval.runs import SUMMARY_PATH, log_summary, save_run
from hybrid_rank.filtering.hard_filter import FilterIndex
from hybrid_rank.ingest.registry import ADAPTERS
from hybrid_rank.pipeline import Pipeline
from hybrid_rank.query.decompose import DEFAULT_EFFORT, DEFAULT_MODEL, Decomposer
from hybrid_rank.query.prompts import DEFAULT_PROMPT, GROUNDED, PROMPTS
from hybrid_rank.retrieval.bm25 import IDF_VARIANTS, BM25Index, bm25_dir
from hybrid_rank.retrieval.embed import ENCODERS, DenseIndex, embeddings_dir
from hybrid_rank.retrieval.rrf import DEFAULT_DEPTH, DEFAULT_K, RRFRetriever
from hybrid_rank.schema import Listing, Query


def build_bm25(args, listings):
    index = BM25Index(listings, cache_dir=bm25_dir(args.dataset), idf=args.bm25_idf)
    print(f"  BM25 index {'loaded from' if index.loaded_from_cache else 'built and saved to'} {index.cache_path}")
    return index


def build_retriever(args, listings):
    # "bm25" means the default (lucene idf); other variants are named in the run, so logs never mix them
    variant = "" if args.bm25_idf == "lucene" else f"-{args.bm25_idf}"
    if args.retriever == "bm25":
        return f"bm25{variant}", build_bm25(args, listings)
    index = DenseIndex(listings, ENCODERS[args.model], embeddings_dir(args.dataset))
    source = "loaded from" if index.loaded_from_cache else "encoded and saved to"
    print(f"  {index.spec.name} on {index.model.device}, embeddings {source} {index.cache_path}")
    if args.retriever == "dense":
        return f"dense-{args.model}", index
    retriever = RRFRetriever([build_bm25(args, listings), index], k=args.rrf_k, depth=args.depth)
    return f"rrf-{args.model}-k{args.rrf_k}-d{args.depth}{variant}", retriever


def build_pipeline(args, listings: List[Listing], queries: List[Query], system: str,
                   retriever) -> Tuple[str, Pipeline, str]:
    """Wraps the retriever in decompose -> hard filter. The run name records every switch:
    +rw = rewritten query, +f.<fields> = hard filters (.strict = listings missing the field dropped),
    +p.<version> = a non-default decomposition prompt. Also returns the decomposition file, for the log."""
    fields = ADAPTERS[args.dataset].filter_fields
    declared = [f.name for f in fields]
    wanted = declared if args.filters == ["all"] else (args.filters or [])
    unknown = [name for name in wanted if name not in declared]
    if unknown:
        raise SystemExit(f"{args.dataset} has no filter field {', '.join(unknown)}; it has: {', '.join(declared)}, all")
    names = [name for name in declared if name in wanted]  # adapter order, so names are canonical

    print("Indexing filter fields ...")
    index = FilterIndex([l.filters for l in listings], fields)
    decomposer = Decomposer.for_dataset(args.dataset, index, model=args.llm_model, effort=args.llm_effort,
                                        prompt=args.prompt)
    missing = decomposer.missing(q.text for q in queries)
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(queries)} queries have no decomposition in {decomposer.cache_path}\n"
            f"  run: python scripts/decompose.py {args.dataset} --split {args.split} --model {args.llm_model}"
            + (f" --effort {args.llm_effort}" if args.llm_effort != DEFAULT_EFFORT else "")
            + (f" --prompt {args.prompt}" if args.prompt != DEFAULT_PROMPT else "")
        )

    suffix = "+rw" if args.rewrite else ""
    if names:
        suffix += "+f." + ".".join(names) + (".strict" if args.strict else "")
    if args.llm_model != DEFAULT_MODEL or args.llm_effort != DEFAULT_EFFORT:
        suffix += f"+{args.llm_model}" + (f"-{args.llm_effort}" if args.llm_effort != DEFAULT_EFFORT else "")
    if args.prompt != DEFAULT_PROMPT:
        suffix += f"+p.{args.prompt}"
    pipeline = Pipeline(retriever, decomposer.cache, index, names, rewrite=args.rewrite,
                        missing="drop" if args.strict else "keep")
    return system + suffix, pipeline, decomposer.cache_path.name


def filter_trace(pipeline: Pipeline, queries: List[Query], df_judgments: pd.DataFrame, ids: List[str],
                 config: EvalConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Per query: what was searched, which filters applied, and the share of its top-grade answers the
    filter kept (retention) -- a filter that drops an answer caps every metric for that query."""
    top = max(config.gain.values())
    top_labels = [label for label, g in config.gain.items() if g == top]
    answers = df_judgments[df_judgments["label"].isin(top_labels)].groupby("query_id")["listing_id"].apply(list)
    pos = {lid: i for i, lid in enumerate(ids)}
    rows = []
    for q in queries:
        plan = pipeline.plan(q.text)
        f = plan.filters
        mask = f.mask if f is not None else None
        rows_of_answers = [pos[lid] for lid in answers.get(q.id, []) if lid in pos]
        retention = (float(mask[rows_of_answers].mean()) if mask is not None else 1.0) if rows_of_answers else np.nan
        rows.append({
            "query_id": q.id,
            "search_text": plan.text,
            "filtered": mask is not None,
            "applied": json.dumps(f.applied if f else {}),
            "unresolved": json.dumps(f.unresolved if f else {}),
            "relaxed": ",".join(f.relaxed) if f else "",
            "n_candidates": f.n_candidates if f else len(ids),
            "retention": retention,
        })
    trace = pd.DataFrame(rows)
    stats = {"filtered": float(trace["filtered"].mean())}
    on_filtered = trace.loc[trace["filtered"], "retention"].dropna()
    if len(on_filtered):
        stats["retention"] = float(on_filtered.mean())       # over filtered queries with an answer
        stats["lost_all"] = float((on_filtered == 0).mean())  # filtered queries left with no answer at all
    return trace, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(EVAL_CONFIGS))
    parser.add_argument("--retriever", choices=["bm25", "dense", "rrf"], required=True)
    parser.add_argument("--model", choices=list(ENCODERS), default="minilm", help="dense and rrf only")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_K, help="rrf only")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="rrf only: candidates from each retriever")
    parser.add_argument("--bm25-idf", choices=IDF_VARIANTS, default="lucene", help="bm25 and rrf only")
    parser.add_argument("--split", default="dev", help="a split in data/splits/<dataset>.json, or 'all'")
    llm = parser.add_argument_group("query decomposition (reads the cache scripts/decompose.py fills)")
    llm.add_argument("--filters", nargs="+", metavar="FIELD", help="hard-filter on these decomposed fields, or 'all'")
    llm.add_argument("--rewrite", action="store_true", help="search with the LLM-rewritten query")
    llm.add_argument("--strict", action="store_true", help="drop listings missing a filtered field (default: keep)")
    llm.add_argument("--llm-model", default=DEFAULT_MODEL, help="whose decompositions to use")
    llm.add_argument("--llm-effort", default=DEFAULT_EFFORT, help="the effort they were made with")
    llm.add_argument("--prompt", choices=list(PROMPTS), default=DEFAULT_PROMPT, help="decomposition prompt version")
    args = parser.parse_args()
    if args.strict and not args.filters:
        parser.error("--strict needs --filters")
    if args.rewrite and args.prompt in GROUNDED:
        parser.error(f"prompt {args.prompt} doesn't rewrite queries")

    config = EVAL_CONFIGS[args.dataset]
    listings, queries, df_judgments = load_processed(args.dataset, split=args.split)
    print(f"Building {args.retriever} index over {len(listings):,} listings ...")
    system, retriever = build_retriever(args, listings)
    pipeline, provenance = None, {}
    if args.filters or args.rewrite:
        system, pipeline, decompositions = build_pipeline(args, listings, queries, system, retriever)
        provenance["decompositions"] = decompositions
        retriever = pipeline

    result = evaluate(queries, df_judgments, search_fn=retriever.search, config=config)
    trace = None
    if pipeline is not None and pipeline.filter_fields:
        trace, stats = filter_trace(pipeline, queries, df_judgments, [l.id for l in listings], config)
        result.summary.update(stats)
    print(f"\n{system} on {args.dataset}/{args.split}:")
    for k, v in result.summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    run_dir = save_run(result, args.dataset, system, args.split)
    if trace is not None:
        trace.to_parquet(run_dir / f"{system}_{args.split}.filters.parquet", index=False)
    log_summary(result, args.dataset, system, args.split, **provenance)
    print(f"\nRun saved to {run_dir}/, summary appended to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
