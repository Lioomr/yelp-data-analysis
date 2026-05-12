"""
Load cleaned business.json + review_small.json into Cassandra for EDA queries.

Prerequisites:
  python prepare_data.py   # produces cleaned JSONL
  python scripts/setup_cassandra.py

Docker (from spark service):
  docker compose exec spark python /app/scripts/ingest_cassandra.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cassandra.cluster import Cluster
from cassandra.concurrent import execute_concurrent_with_args

from pipeline_config import (
    BUSINESS_JSON,
    REVIEW_SMALL_JSON,
    CASSANDRA_HOSTS,
    CASSANDRA_PORT,
    CASSANDRA_KEYSPACE,
)


def _read_jsonl(path: str, limit: int | None):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truncate", action="store_true", help="TRUNCATE tables before load")
    parser.add_argument("--review-limit", type=int, default=None, help="Max reviews to load (debug)")
    args = parser.parse_args()

    hosts = CASSANDRA_HOSTS or ["localhost"]
    if not os.path.isfile(BUSINESS_JSON):
        print(f"Missing {BUSINESS_JSON}. Run prepare_data.py first.")
        return 1
    if not os.path.isfile(REVIEW_SMALL_JSON):
        print(f"Missing {REVIEW_SMALL_JSON}. Run prepare_data.py first.")
        return 1

    cluster = Cluster(hosts, port=CASSANDRA_PORT)
    session = cluster.connect(CASSANDRA_KEYSPACE)

    if args.truncate:
        session.execute("TRUNCATE businesses")
        session.execute("TRUNCATE reviews")

    prep_b = session.prepare(
        """
        INSERT INTO businesses (business_id, city, categories, stars, review_count)
        VALUES (?, ?, ?, ?, ?)
        """
    )
    prep_r = session.prepare(
        """
        INSERT INTO reviews (review_id, business_id, user_id, stars, text)
        VALUES (?, ?, ?, ?, ?)
        """
    )

    biz_rows = []
    for r in _read_jsonl(BUSINESS_JSON, None):
        bid = r.get("business_id")
        if not bid:
            continue
        biz_rows.append(
            (
                bid,
                r.get("city") or "",
                r.get("categories") or "",
                float(r.get("stars") or 0.0),
                int(r.get("review_count") or 0),
            )
        )

    print(f"Loading {len(biz_rows)} businesses...")
    execute_concurrent_with_args(session, prep_b, biz_rows, concurrency=64)

    rev_rows = []
    for r in _read_jsonl(REVIEW_SMALL_JSON, args.review_limit):
        rid = r.get("review_id")
        if not rid:
            continue
        rev_rows.append(
            (
                rid,
                r.get("business_id") or "",
                r.get("user_id") or "",
                float(r.get("stars") or 0.0),
                r.get("text") or "",
            )
        )

    print(f"Loading {len(rev_rows)} reviews...")
    execute_concurrent_with_args(session, prep_r, rev_rows, concurrency=32)

    print("Ingest complete.")
    cluster.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
