PYTHON  ?= python3
VENV    := .venv
BIN     := $(VENV)/bin
MODELS  ?= minilm bge-small bge-base e5-base
DATASET ?= wands

.PHONY: venv compare-models

venv:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e ".[dev]"

# every embedding model alone and fused with BM25 (default k / depth), on dev only
# e.g. make compare-models DATASET=stark-amazon MODELS=bge-small
compare-models:
	$(BIN)/python scripts/eval.py $(DATASET) --retriever bm25 --split dev
	for m in $(MODELS); do \
		$(BIN)/python scripts/eval.py $(DATASET) --retriever dense --model $$m --split dev && \
		$(BIN)/python scripts/eval.py $(DATASET) --retriever rrf --model $$m --split dev || exit 1; \
	done
	$(BIN)/python scripts/compare_runs.py $(DATASET) --split dev --baseline bm25 --systems "dense-*" "rrf-*-k60-d500"
