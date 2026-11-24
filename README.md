# Hybrid Rank - Hybrid Retrieval Pipeline

### Oleksandr

Domain-agnostic retrieval pipeline for marketplace / product search: LLM query decomposition and hard
filters, then BM25 + dense retrieval fused with reciprocal rank fusion, and reranking. Every
component is evaluated through one harness on two benchmarks.

## Benchmarks

| Dataset | Listings | Queries | Labels | Split |
|---|---|---|---|---|
| `wands` (Wayfair, Chen et al. 2022) | 42,994 | 480 short keyword queries | Exact / Partial / Irrelevant | random, seed 0: dev 96 / test 384 |
| `stark-amazon` (STaRK, Wu et al. 2024) | 957,192 | 9,100 synthesized + 81 human natural-language queries | answers only (binary) | official: train / dev (= val) / test / test-0.1, plus human |

Metrics follow each benchmark's paper. WANDS uses nDCG@10/50, P@5, MRR and R@50/100 over Exact, plus
cnDCG@10 and unlabelled@10. STaRK uses Hit@1, Hit@5, R@20 and MRR, plus nDCG@10, P@5 and R@100
(see `src/hybrid_rank/eval/config.py`).

## Usage

```bash
make venv && source .venv/bin/activate

python scripts/download_data.py wands            # fetch + normalize, write split
python scripts/download_data.py stark-amazon     # 3.5 GB download; ~2 min and ~11 GB RAM to convert
python scripts/verify_data.py <dataset>          # data checks

python scripts/eval.py <dataset> --retriever bm25 --split dev
python scripts/eval.py <dataset> --retriever dense --model bge-small --split dev
python scripts/eval.py <dataset> --retriever rrf --model bge-small --split dev
python scripts/compare_runs.py <dataset> --split dev --baseline bm25 --systems "dense-*" "rrf-*-k60-d500"
python scripts/tune_rrf.py <dataset> --model bge-small     # k x depth grid, dev only
make compare-models DATASET=<dataset>                       # all encoders alone and fused

python scripts/query.py wands "outdoor storage" --model bge-small -k 5   # inspect rankings with labels
python -m pytest -q
```

Each eval run saves per-query results to `results/runs/` and appends a summary line (with the git commit)
to `results/summary.jsonl`. Tune on dev, report on test once.

## Query decomposition and hard filters

An LLM turns each request into a cleaned search query plus constraints on the fields the dataset's adapter
declares (`filter_fields`: STaRK brand / category / color / price, WANDS product_class / color / style).
The constraints become a mask every retriever honours. Listings missing a field are kept, values match
loosely (case, plural, spacing, `prefix` for names), unmatched values are skipped, and constraints are relaxed
lowest-priority first if fewer than 10 listings survive (`src/hybrid_rank/filtering/hard_filter.py`).

```bash
# the only paid step: put ANTHROPIC_API_KEY=... in .env (git-ignored). Cached per query in
# results/decompositions/, so re-runs resume
python scripts/decompose.py stark-amazon --print-prompt              # see the prompt, no API calls
python scripts/decompose.py stark-amazon --split dev --limit 5       # smoke test
python scripts/decompose.py stark-amazon --split dev                 # default claude-haiku-5-5; --model to change

# eval reads the cache; the run name records the switches (+rw, +f.<fields>, .strict)
python scripts/eval.py stark-amazon --retriever rrf --model bge-small --filters brand
python scripts/eval.py stark-amazon --retriever rrf --model bge-small --filters all --rewrite
python scripts/eval.py stark-amazon --retriever rrf --model bge-small --filters brand --strict
```

Filtered runs also log `filtered` (share of queries with a filter applied), `retention` (share of their
top-grade answers the filter kept) and `lost_all` (filtered queries left with no answer), and save a
per-query trace to `results/runs/<dataset>/<run>_<split>.filters.parquet`. Commit
`results/decompositions/` so every number can be reproduced without an API key.

Prompt versions: `--prompt v1` (default) or `v2`, on both `decompose.py` and `eval.py`; runs with v2 get `+p.v2`.
v1 is frozen because its text is part of the cache key. Compare prompts without API calls:

```bash
python scripts/decompose.py stark-amazon --split train --sample 300 --prompt v2     # paid; a fixed random sample
python scripts/check_filters.py stark-amazon --split train --sample 300 --prompts v1 v2
```

## Reranking

`scripts/rerank.py` reorders the top results of a saved run and evaluates that as a new run, so it needs no
indexes in memory. `--blend` fuses the reranked and original orders with RRF.

```bash
python scripts/rerank.py stark-amazon --run rrf-bge-small-k60-d500+f.brand --reranker minilm --limit 50   # timing only
python scripts/rerank.py stark-amazon --run rrf-bge-small-k60-d500+f.brand --reranker minilm --depth 10 20 50 100 --blend both
python scripts/rerank.py stark-amazon --run rrf-bge-small-k60-d500+f.brand --reranker llm --call-llm --limit 5   # paid
```

Cross-encoders (`minilm`, `bge-base`) score pairs of at most 512 tokens, cached in `data/processed/<dataset>/rerank/`.
The `llm` reranker ranks a list in one call per query; without `--call-llm` it only reads `results/reranks/`.

## Caches

BM25 indexes (`data/processed/<dataset>/bm25/`) and embeddings (`data/processed/<dataset>/embeddings/`)
are keyed by a hash of the listing text and settings, so any change rebuilds instead of loading stale data.
On STaRK, encoding ~1M documents takes hours; it saves progress in chunks and resumes if interrupted.
