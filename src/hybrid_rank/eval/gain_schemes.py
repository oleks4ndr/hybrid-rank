# gain_schemes.py
# per-benchmark relevance gains (WANDS's 1/0.5/0, ESCI's 1/0.1/0.01/0, custom)

WANDS_GAIN = {"Exact": 1.0, "Partial": 0.5, "Irrelevant": 0.0}
STARK_GAIN = {"relevant": 1.0}  # binary: only a query's answers are labelled

GAIN_SCHEMES = {"wands": WANDS_GAIN, "stark-amazon": STARK_GAIN}
