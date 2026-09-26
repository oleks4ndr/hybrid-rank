PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin

.PHONY: venv

venv:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e .
