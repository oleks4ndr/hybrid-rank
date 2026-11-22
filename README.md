# Hybrid Rank - Hybrid Retrieval Pipeline

### Oleksandr

Domain-agnostic retrieval pipeline for marketplace / product search: BM25 + dense retrieval fused with
reciprocal rank fusion, with LLM query decomposition, hard filters and reranking to come. Every component
is evaluated through one harness on two benchmarks.

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

## Caches

BM25 indexes (`data/processed/<dataset>/bm25/`) and embeddings (`data/processed/<dataset>/embeddings/`)
are keyed by a hash of the listing text and settings, so any change rebuilds instead of loading stale data.
On STaRK, encoding ~1M documents takes hours; it saves progress in chunks and resumes if interrupted.
