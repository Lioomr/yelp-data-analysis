"""
Create Yelp keyspace and tables for optional Cassandra EDA (data_visualization.py).

Run inside Docker after Cassandra is healthy:
  docker compose exec spark python /app/scripts/setup_cassandra.py

Hosts: CASSANDRA_HOSTS (comma-separated), default localhost.
In compose, set: environment CASSANDRA_HOSTS=cassandra
"""

from __future__ import annotations

import os
import sys
import time

# Allow running as script from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cassandra.cluster import Cluster

from pipeline_config import CASSANDRA_HOSTS, CASSANDRA_PORT, CASSANDRA_KEYSPACE


def wait_for_cassandra(hosts: list[str], port: int, attempts: int = 30, delay: float = 2.0):
    last_err = None
    for i in range(attempts):
        try:
            c = Cluster(hosts, port=port, connect_timeout=5)
            c.connect()
            c.shutdown()
            print(f"Cassandra reachable at {hosts}:{port}")
            return
        except Exception as e:
            last_err = e
            print(f"[{i+1}/{attempts}] Waiting for Cassandra... ({e})")
            time.sleep(delay)
    raise RuntimeError(f"Cassandra not available after {attempts} attempts: {last_err}")


def main() -> int:
    hosts = CASSANDRA_HOSTS
    if not hosts:
        hosts = ["localhost"]

    wait_for_cassandra(hosts, CASSANDRA_PORT)

    cluster = Cluster(hosts, port=CASSANDRA_PORT)
    session = cluster.connect()

    session.execute(
        f"""
        CREATE KEYSPACE IF NOT EXISTS {CASSANDRA_KEYSPACE}
        WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}};
        """
    )
    session.set_keyspace(CASSANDRA_KEYSPACE)

    session.execute(
        """
        CREATE TABLE IF NOT EXISTS businesses (
            business_id text PRIMARY KEY,
            city text,
            categories text,
            stars float,
            review_count int
        );
        """
    )

    session.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            review_id text PRIMARY KEY,
            business_id text,
            user_id text,
            stars float,
            text text
        );
        """
    )

    print(f"OK: keyspace '{CASSANDRA_KEYSPACE}', tables businesses, reviews")
    cluster.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
