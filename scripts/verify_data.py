# verify_data.py
# script to verify that data was properly retrieved
# 
# usage: python scripts/verify_data.py wands

import argparse
from pathlib import Path
from typing import Any, cast

import pandas as pd

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
    print(f"  text: {sample.text[:200]}...")
    print(f"  filters: {sample.filters}\n")

    query_record = cast(dict[str, Any], df_queries.iloc[0].to_dict())
    sample_q = Query.from_record(query_record)
    print("Sample query:")
    print(f"  id: {sample_q.id}  text: {sample_q.text}  category: {sample_q.category}\n")

    # The checks that actually matter -- catch silent parsing failures early
    empty_text = (df_listings["text"].str.strip() == "").sum()
    empty_filters = (df_listings["filters"] == "{}").sum()
    orphans = (~df_judgments["listing_id"].isin(df_listings["id"])).sum()

    print(f"Listings with empty text: {empty_text}")
    print(f"Listings with zero parsed filters: {empty_filters} ({empty_filters/len(df_listings):.1%})")
    print(f"Judgments pointing at missing listings: {orphans}")


if __name__ == "__main__":
    main()