# download_data.py
# CLI: pick a domain, run its ingest adapter
# 
# usage: python scripts/download_data.py wands

import argparse

import pandas as pd

from hybrid_rank.ingest.wands import WandsAdapter

ADAPTERS = {"wands": WandsAdapter}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=ADAPTERS.keys())
    args = parser.parse_args()

    adapter = ADAPTERS[args.dataset]()
    print(f"Fetching raw {args.dataset} data ...")
    adapter.fetch_raw()

    print("Normalizing to common schema ...")
    listings, queries, judgments = adapter.to_common_schema()

    adapter.processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([l.to_record() for l in listings]).to_parquet(adapter.processed_dir / "listings.parquet", index=False)
    pd.DataFrame([q.to_record() for q in queries]).to_parquet(adapter.processed_dir / "queries.parquet", index=False)
    pd.DataFrame([j.to_record() for j in judgments]).to_parquet(adapter.processed_dir / "judgments.parquet", index=False)

    print(f"Listings:  {len(listings):,}")
    print(f"Queries:   {len(queries):,}")
    print(f"Judgments: {len(judgments):,}")
    print(f"Saved to {adapter.processed_dir}/")


if __name__ == "__main__":
    main()