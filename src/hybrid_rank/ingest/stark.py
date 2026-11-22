# stark.py
# STaRK-Amazon (Wu et al., NeurIPS 2024): Amazon Sports & Outdoors products, plus natural-language queries
# that mix textual needs with brand / color / category constraints. Labels are binary: a query's answers.
# Source: https://huggingface.co/datasets/snap-stanford/stark
#
# Reads the benchmark's own files directly instead of depending on the stark-qa package, which pulls in
# colbert-ai, gritlm, llm2vec, torch_geometric, ... Document text replicates stark_qa's get_doc_info, so our
# BM25 / dense runs see the same documents as the paper's baselines.

import ast
import pickle
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from hybrid_rank.ingest.base import DatasetAdapter
from hybrid_rank.schema import Judgment, Listing, Query

REPO = "snap-stanford/stark"
QA_DIR = "qa/amazon"
SKB_ZIP = "skb/amazon/processed.zip"
NODE_INFO = "processed/cache/brand-category-color/node_info.pkl"  # products, then brand/category/color nodes
QA_CSV = f"{QA_DIR}/stark_qa/stark_qa.csv"
HUMAN_CSV = f"{QA_DIR}/stark_qa/stark_qa_human_generated_eval.csv"
# STaRK split -> ours. Tune on val, report on test. test-0.1 is the 10% subset its LLM rerankers were scored on.
SPLITS = {"train": "train", "val": "dev", "test": "test", "test-0.1": "test-0.1"}
FILES = (QA_CSV, HUMAN_CSV, *(f"{QA_DIR}/split/{s}.index" for s in SPLITS), SKB_ZIP)

MAX_ENTRIES = 25  # stark_qa's default cap on reviews / Q&A per product (its loop keeps up to 27)
META_NODE_KEYS = {"brand_name", "category_name", "color_name"}
HUMAN_PREFIX = "h"  # human-generated query ids restart at 0, so they get a prefix


class _DataOnlyUnpickler(pickle.Unpickler):
    """node_info.pkl is third-party, and pickle.load runs whatever callables a pickle names.
    Allow only what plain data and numpy scalars need; anything else fails loudly instead of running."""

    ALLOWED = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("collections", "OrderedDict"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} -- not on the allow-list")


def doc_text(node: Dict[str, Any], max_entries: int = MAX_ENTRIES) -> str:
    """A product's text exactly as stark_qa's AmazonSKB.get_doc_info(idx) builds it (add_rel=False,
    compact=False), quirks included: reviews are ordered by helpful votes and numbered by original
    position, and missing review fields print as 'nan'."""
    doc = f"- product: {node['title']}\n"
    if "brand" in node:
        doc += f"- brand: {node['brand']}\n"
    try:
        dimensions, weight = node["details"]["product_dimensions"].split(" ; ")
        doc += f"- dimensions: {dimensions}\n- weight: {weight}\n"
    except Exception:  # no details, no dimensions, or not 'dims ; weight' -- the original skips all of these
        pass
    if node.get("description"):
        description = " ".join(node["description"]).strip(" ")
        if description:
            doc += f"- description: {description}\n"

    feature_text = ""
    if node.get("feature"):
        feature_text = "- features: \n"
        for i, feature in enumerate(node["feature"]):
            if feature and "asin" not in feature.lower():
                feature_text += f"#{i + 1}: {feature}\n"

    review_text = ""
    if node.get("review"):
        review_text = "- reviews: \n"
        votes = [0 if pd.isnull(r["vote"]) else int(r["vote"].replace(",", "")) for r in node["review"]]
        for i, review_idx in enumerate(np.argsort(-np.array(votes))):  # same (unstable) sort as the original
            review = node["review"][review_idx]
            review_text += f'#{review_idx + 1}:\nsummary: {review["summary"]}\ntext: "{review["reviewText"]}"\n'
            if i > max_entries:
                break

    qa_text = ""
    if node.get("qa"):
        qa_text = "- Q&A: \n"
        for i, qa in enumerate(node["qa"]):
            qa_text += f'#{i + 1}:\nquestion: "{qa["question"]}"\nanswer: "{qa["answer"]}"\n'
            if i > max_entries:
                break

    return doc + feature_text + review_text + qa_text


def parse_price(raw: Any) -> Optional[float]:
    # '$12.99' -> 12.99; a range '$10.99 - $15.99' -> its low end; anything else -> None
    if not isinstance(raw, str):
        return None
    prices = [float(p.replace(",", "")) for p in re.findall(r"\$\s*([\d,]+(?:\.\d+)?)", raw)]
    return min(prices) if prices else None


def node_filters(node: Dict[str, Any]) -> Dict[str, Any]:
    # brand / category / color are the fields STaRK turns into has_brand / has_category / has_color edges
    filters: Dict[str, Any] = {}
    if node.get("brand"):
        filters["brand"] = node["brand"]
    for key in ("category", "color"):
        if node.get(key):
            filters[key] = list(node[key])
    price = parse_price(node.get("price"))
    if price is not None:
        filters["price"] = price
    return filters


def is_product(node: Dict[str, Any]) -> bool:
    return not META_NODE_KEYS & node.keys()


def parse_qa(df: pd.DataFrame, id_prefix: str, category: str) -> Tuple[List[Query], List[Judgment]]:
    queries, judgments = [], []
    for row in df.itertuples(index=False):
        qid = f"{id_prefix}{row.id}"
        queries.append(Query(id=qid, text=str(row.query), category=category))
        # answer_ids is a stringified list; literal_eval, not eval -- it's downloaded data
        judgments += [Judgment(qid, str(aid), "relevant") for aid in ast.literal_eval(row.answer_ids)]
    return queries, judgments


class StarkAmazonAdapter(DatasetAdapter):
    name = "stark-amazon"

    def fetch_raw(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        for filename in FILES:
            print(f"  {filename} ...")
            hf_hub_download(REPO, filename, repo_type="dataset", local_dir=self.raw_dir)

    def load_node_info(self) -> Dict[int, Dict[str, Any]]:
        # streamed out of the zip: the 5.4 GB pickle never has to be extracted to disk
        with zipfile.ZipFile(self.raw_dir / SKB_ZIP) as z, z.open(NODE_INFO) as f:
            return _DataOnlyUnpickler(f).load()

    def to_common_schema(self) -> Tuple[List[Listing], List[Query], List[Judgment]]:
        print("  loading node_info.pkl (~2 min, ~11 GB of RAM) ...")
        node_info = self.load_node_info()
        listings = [
            Listing(
                id=str(node_id),
                text=doc_text(node),
                filters=node_filters(node),
                source={"asin": node.get("asin"), "title": node.get("title")},
            )
            for node_id, node in node_info.items()
            if is_product(node)
        ]
        del node_info

        queries, judgments = parse_qa(pd.read_csv(self.raw_dir / QA_CSV), "", "synthesized")
        human_q, human_j = parse_qa(pd.read_csv(self.raw_dir / HUMAN_CSV), HUMAN_PREFIX, "human")
        return listings, queries + human_q, judgments + human_j

    def official_split(self) -> Dict[str, Any]:
        split: Dict[str, Any] = {"source": "STaRK official split (val -> dev); human = human-generated eval set"}
        for stark_name, ours in SPLITS.items():
            lines = (self.raw_dir / QA_DIR / "split" / f"{stark_name}.index").read_text().split()
            split[ours] = sorted(lines, key=int)
        human_ids = pd.read_csv(self.raw_dir / HUMAN_CSV)["id"]
        split["human"] = [f"{HUMAN_PREFIX}{i}" for i in sorted(human_ids)]
        return split
