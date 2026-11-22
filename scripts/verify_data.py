# verify_data.py
# script to verify that data was properly retrieved
#
# usage: python scripts/verify_data.py wands
#        python scripts/verify_data.py stark-amazon

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pandas as pd

from hybrid_rank.eval.splits import load_split, split_names, split_path
from hybrid_rank.schema import Listing, Query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()

    processed_dir = Path("data/processed") / args.dataset
    df_listings = pd.read_parquet(processed_dir / "listings.parquet")
    df_queries = pd.read_parquet(processed_dir / "queries.parquet")
    df_judgments = pd.read_parquet(processed_dir / "judgments.parquet")

    print(f"=== {args.dataset} ===")
    print(f"Listings:  {len(df_listings):,}")
    print(f"Queries:   {len(df_queries):,}")
    print(f"Judgments: {len(df_judgments):,}\n")

    print("Label distribution:")
    print(df_judgments["label"].value_counts(), "\n")

    listing_record = cast(dict[str, Any], df_listings.iloc[0].to_dict())
    sample = Listing.from_record(listing_record)
    print("Sample listing:")
    print(f"  id: {sample.id}")
    print(f"  text: {' '.join(sample.text.split())[:200]}...")
    print(f"  filters: {sample.filters}\n")

    query_record = cast(dict[str, Any], df_queries.iloc[0].to_dict())
    sample_q = Query.from_record(query_record)
    print("Sample query:")
    print(f"  id: {sample_q.id}  text: {sample_q.text}  category: {sample_q.category}\n")

    # The checks that actually matter -- catch silent parsing failures early
    text_len = df_listings["text"].str.len()
    empty_text = (df_listings["text"].str.strip() == "").sum()
    empty_filters = (df_listings["filters"] == "{}").sum()
    orphans = (~df_judgments["listing_id"].isin(df_listings["id"])).sum()
    nan_text = df_listings["text"].str.contains(r"\bnan\b").sum()
    dup_judgments = df_judgments.duplicated(["query_id", "listing_id"]).sum()
    dup_listing_ids = df_listings["id"].duplicated().sum()
    per_query = df_judgments.groupby("query_id").size()
    unjudged = (~df_queries["id"].isin(per_query.index)).sum()

    print(f"Listing text length (chars): median {text_len.median():,.0f}, mean {text_len.mean():,.0f}, max {text_len.max():,}")
    print(f"Listings with empty text: {empty_text}")
    print(f"Listings with zero parsed filters: {empty_filters} ({empty_filters/len(df_listings):.1%})")
    print(f"Listings with literal 'nan' in text: {nan_text} ({nan_text/len(df_listings):.1%})"
          + ("  <- STaRK's own document format prints missing review fields as 'nan'" if args.dataset.startswith("stark") else ""))
    print(f"Duplicate listing ids: {dup_listing_ids}")
    print(f"Judgments pointing at missing listings: {orphans}")
    print(f"Duplicate (query, listing) judgments: {dup_judgments}")
    print(f"Queries without judgments: {unjudged}")
    print(f"Judgments per query: min {per_query.min()}, median {per_query.median():.0f}, max {per_query.max()}")

    coverage = Counter(k for f in df_listings["filters"] for k in json.loads(f))
    print("Filter coverage (share of listings with the key):")
    for key, n in coverage.most_common(8):
        print(f"  {key}: {n/len(df_listings):.1%}")

    if split_path(args.dataset).exists():
        split = load_split(args.dataset)
        print("Splits: " + ", ".join(f"{name} {len(split[name])}" for name in split_names(split)))


if __name__ == "__main__":
    main()
