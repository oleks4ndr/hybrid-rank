# download_data.py
# CLI: pick a domain, run its ingest adapter
#
# usage: python scripts/download_data.py wands
#        python scripts/download_data.py stark-amazon   # 3.5 GB download; ~2 min and ~11 GB of RAM to convert

import argparse

import pandas as pd

from hybrid_rank.eval.splits import save_split_if_missing, split_path
from hybrid_rank.ingest.registry import ADAPTERS


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

    if save_split_if_missing(args.dataset, [q.id for q in queries], adapter.official_split()):
        print(f"Wrote split to {split_path(args.dataset)}")
    else:
        print(f"Keeping existing split {split_path(args.dataset)}")


if __name__ == "__main__":
    main()
