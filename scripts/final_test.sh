#!/bin/bash
# Final test runs, once, with the configurations frozen on dev. Every step is skipped if its result already
# exists, so an interrupted run resumes by running the same command again, and nothing is paid for or scored twice.
#
# usage (macOS Terminal, venv active, from the repo root, code committed):
#   caffeinate -i bash scripts/final_test.sh            # everything
#   bash scripts/final_test.sh --dry-run                # list the steps, run nothing
set -euo pipefail
cd "$(dirname "$0")/.."
DRY=${1:-}
PY=${PY:-.venv/bin/python}
RRF=rrf-bge-small-k60-d500
BRAND=$RRF+f.brand
CE=ce.bge-base-d100.blend
HAIKU=llm.claude-haiku-5-5-d20

if [ "$DRY" != "--dry-run" ] && [ -n "$(git status --porcelain --untracked-files=no -- . ':!results')" ]; then
  echo "Commit your code first, so the results log a clean commit."; exit 1
fi

# step <dataset> <split> <run name that proves it is done> <command ...>
step() {
  local done_file="results/runs/$1/$3_$2.per_query.parquet" label="$1/$2  $3"; shift 3
  if [ -f "$done_file" ] && [ "$FORCE" = no ]; then echo "skip  $label"; return; fi
  echo; echo "=== $label"
  if [ "$DRY" = "--dry-run" ]; then echo "  $*"; else "$@"; fi
}
FORCE=no
always() { echo; echo "=== $*"; [ "$DRY" = "--dry-run" ] || "$@"; }   # resumable on its own (cached)

# ---------- WANDS test (the 18 Oct bm25 test run used the old idf: the Okapi row below replaces it by name)
w() { step wands test "$1" $PY scripts/eval.py wands --split test "${@:2}"; }
always $PY scripts/decompose.py wands --split test --show 0
if [ ! -f results/runs/wands/bm25-okapi_test.per_query.parquet ]; then FORCE=yes; fi
w bm25 --retriever bm25
FORCE=no
w bm25-okapi --retriever bm25 --bm25-idf okapi
w dense-bge-small --retriever dense --model bge-small
w $RRF --retriever rrf --model bge-small
w $RRF+f.product_class.color.style --retriever rrf --model bge-small --filters all
w $RRF+rw --retriever rrf --model bge-small --rewrite
step wands test $RRF+ce.minilm-d20.blend $PY scripts/rerank.py wands --split test --run $RRF --reranker minilm --depth 20 --blend yes
step wands test $RRF+$HAIKU $PY scripts/rerank.py wands --split test --run $RRF --reranker llm --call-llm --depth 20

# ---------- STaRK test (test-0.1 is part of test) and the human-written queries
for split in test human; do
  s() { step stark-amazon $split "$1" $PY scripts/eval.py stark-amazon --split $split "${@:2}"; }
  r() { step stark-amazon $split "$1" $PY scripts/rerank.py stark-amazon --split $split "${@:2}"; }
  always $PY scripts/decompose.py stark-amazon --split $split --show 0
  s bm25 --retriever bm25
  s bm25-okapi --retriever bm25 --bm25-idf okapi
  s dense-bge-small --retriever dense --model bge-small
  s $RRF --retriever rrf --model bge-small
  s $BRAND --retriever rrf --model bge-small --filters brand
  s $BRAND.strict --retriever rrf --model bge-small --filters brand --strict
  s $RRF+f.brand.category.color.price --retriever rrf --model bge-small --filters all
  s $RRF+rw --retriever rrf --model bge-small --rewrite
  r $BRAND+ce.minilm-d100.blend --run $BRAND --reranker minilm --depth 100 --blend yes
  r $RRF+$CE --run $RRF --reranker bge-base --depth 100 --blend yes
  r $BRAND+$CE --run $BRAND --reranker bge-base --depth 100 --blend yes
  r $BRAND+$CE+$HAIKU --run $BRAND+$CE --reranker llm --call-llm --depth 20
done

if [ "$DRY" != "--dry-run" ]; then
  $PY scripts/final_tables.py | tee results/final_tables.txt
  echo; echo "ALL DONE. Tables are in results/final_tables.txt"
fi
