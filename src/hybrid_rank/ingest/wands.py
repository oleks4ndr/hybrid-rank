# wands.py
# common schema for WANDS dataset of products and queries. 
# Source: https://github.com/wayfair/WANDS (MIT)

import urllib.request
from typing import Any, Dict, List, Tuple

import pandas as pd

from hybrid_rank.ingest.base import DatasetAdapter
from hybrid_rank.schema import Judgment, Listing, Query

BASE_URL = "https://github.com/wayfair/WANDS/raw/main/dataset/"
FILES = ("product.csv", "query.csv", "label.csv")
LABEL_RANK = {"Irrelevant": 0, "Partial": 1, "Exact": 2}


def _parse_features(raw: Any) -> Dict[str, str]:
    # product_features is a '|'-delimited string of 'attribute:value' pairs
    if not isinstance(raw, str) or not raw.strip():
        return {}
    features = {}
    for pair in raw.split("|"):
        if ":" not in pair:
            continue
        key, _, value = pair.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key and value:
            features[key] = value
    return features


def _dedupe_labels(df_labels: pd.DataFrame) -> pd.DataFrame:
    # label.csv repeats some (query, product) pairs, a few with conflicting labels -- keep the highest
    ranked = df_labels.assign(_rank=df_labels["label"].map(LABEL_RANK))
    deduped = ranked.sort_values("_rank").drop_duplicates(["query_id", "product_id"], keep="last")
    return deduped.drop(columns="_rank")


class WandsAdapter(DatasetAdapter):
    name = "wands"

    def fetch_raw(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        for filename in FILES:
            dest = self.raw_dir / filename
            if dest.exists():
                print(f"  {filename} cached, skipping.")
                continue
            print(f"  downloading {filename} ...")
            urllib.request.urlretrieve(f"{BASE_URL}{filename}", dest)

    def to_common_schema(self) -> Tuple[List[Listing], List[Query], List[Judgment]]:
        df_products = pd.read_csv(self.raw_dir / "product.csv", sep="\t")
        df_queries = pd.read_csv(self.raw_dir / "query.csv", sep="\t")
        df_labels = _dedupe_labels(pd.read_csv(self.raw_dir / "label.csv", sep="\t"))

        listings = [self._to_listing(row) for _, row in df_products.iterrows()]
        queries = [
            Query(id=str(r["query_id"]), text=str(r["query"]), category=r.get("query_class"))
            for _, r in df_queries.iterrows()
        ]
        judgments = [
            Judgment(query_id=str(r["query_id"]), listing_id=str(r["product_id"]), label=str(r["label"]))
            for _, r in df_labels.iterrows()
        ]
        return listings, queries, judgments

    @staticmethod
    def _to_listing(row: pd.Series) -> Listing:
        filters = {
            **_parse_features(row.get("product_features")),
            "product_class": row.get("product_class"),
            "category_hierarchy": row.get("category hierarchy"),  # raw column name has a space
            "average_rating": row.get("average_rating"),
            "rating_count": row.get("rating_count"),
        }
        filters = {k: v for k, v in filters.items() if pd.notna(v)}

        # pd.notna, not `or ""`: NaN is truthy and would otherwise become the literal token "nan"
        text = " ".join(
            str(row[f]) for f in ("product_name", "product_class", "product_description") if pd.notna(row.get(f))
        ).strip()

        source = {str(key): value for key, value in row.to_dict().items()}
        return Listing(id=str(row["product_id"]), text=text, filters=filters, source=source)