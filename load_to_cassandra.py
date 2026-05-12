"""
Load cleaned business.json and review_small.json into Cassandra.
Handles large review text by using smaller batches and individual inserts.
"""

import json
import os
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement, SimpleStatement, ConsistencyLevel

from pipeline_config import (
    BUSINESS_JSON,
    REVIEW_SMALL_JSON,
    CASSANDRA_HOSTS,
    CASSANDRA_PORT,
    CASSANDRA_KEYSPACE,
)


def create_tables(session):
    """Create Cassandra tables if they don't exist."""
    
    # Drop and recreate for clean state
    session.execute("DROP TABLE IF EXISTS businesses")
    session.execute("DROP TABLE IF EXISTS reviews")
    
    # Businesses table
    session.execute("""
        CREATE TABLE IF NOT EXISTS businesses (
            business_id TEXT PRIMARY KEY,
            name TEXT,
            address TEXT,
            city TEXT,
            state TEXT,
            postal_code TEXT,
            latitude DOUBLE,
            longitude DOUBLE,
            stars DOUBLE,
            review_count INT,
            is_open INT,
            categories TEXT
        )
    """)
    
    # Reviews table
    session.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id TEXT PRIMARY KEY,
            user_id TEXT,
            business_id TEXT,
            stars DOUBLE,
            useful INT,
            funny INT,
            cool INT,
            text TEXT,
            date TEXT
        )
    """)
    
    print("✅ Tables created")


def load_businesses(session):
    """Load business.json into Cassandra."""
    
    if not os.path.exists(BUSINESS_JSON):
        print(f"⚠️  {BUSINESS_JSON} not found, skipping businesses")
        return
    
    print(f"\n⏳ Loading businesses from {BUSINESS_JSON}...")
    
    insert_stmt = session.prepare("""
        INSERT INTO businesses (
            business_id, name, address, city, state, postal_code,
            latitude, longitude, stars, review_count, is_open, categories
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)
    insert_stmt.consistency_level = ConsistencyLevel.ONE
    
    count = 0
    batch = BatchStatement()
    batch_size = 0
    
    with open(BUSINESS_JSON, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                biz = json.loads(line.strip())
                
                # Small businesses can be batched
                batch.add(insert_stmt, (
                    biz.get('business_id', ''),
                    biz.get('name', ''),
                    biz.get('address', ''),
                    biz.get('city', ''),
                    biz.get('state', ''),
                    biz.get('postal_code', ''),
                    float(biz.get('latitude', 0.0)),
                    float(biz.get('longitude', 0.0)),
                    float(biz.get('stars', 0.0)),
                    int(biz.get('review_count', 0)),
                    int(biz.get('is_open', 0)),
                    biz.get('categories', '')
                ))
                
                batch_size += 1
                count += 1
                
                # Execute batch every 50 rows (reduced from 100)
                if batch_size >= 50:
                    session.execute(batch)
                    batch = BatchStatement()
                    batch_size = 0
                    print(f"   Loaded {count:,} businesses...", end='\r')
                    
            except Exception as e:
                print(f"\n⚠️  Error loading business: {e}")
                continue
    
    # Execute remaining batch
    if batch_size > 0:
        session.execute(batch)
    
    print(f"\n✅ Loaded {count:,} businesses")


def load_reviews(session):
    """Load review_small.json into Cassandra with individual inserts for large reviews."""
    
    if not os.path.exists(REVIEW_SMALL_JSON):
        print(f"⚠️  {REVIEW_SMALL_JSON} not found, skipping reviews")
        return
    
    print(f"\n⏳ Loading reviews from {REVIEW_SMALL_JSON}...")
    
    insert_stmt = session.prepare("""
        INSERT INTO reviews (
            review_id, user_id, business_id, stars, useful, funny, cool, text, date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)
    insert_stmt.consistency_level = ConsistencyLevel.ONE
    
    count = 0
    skipped = 0
    
    with open(REVIEW_SMALL_JSON, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                review = json.loads(line.strip())
                
                # Truncate very long reviews (> 5000 chars)
                text = review.get('text', '')
                if len(text) > 5000:
                    text = text[:5000] + "... [truncated]"
                
                # Insert individually (no batching for reviews due to variable text size)
                session.execute(insert_stmt, (
                    review.get('review_id', ''),
                    review.get('user_id', ''),
                    review.get('business_id', ''),
                    float(review.get('stars', 0.0)),
                    int(review.get('useful', 0)),
                    int(review.get('funny', 0)),
                    int(review.get('cool', 0)),
                    text,
                    review.get('date', '')
                ))
                
                count += 1
                
                if count % 1000 == 0:
                    print(f"   Loaded {count:,} reviews...", end='\r')
                    
            except Exception as e:
                skipped += 1
                if skipped <= 5:  # Only print first 5 errors
                    print(f"\n⚠️  Error loading review {count}: {e}")
                continue
    
    print(f"\n✅ Loaded {count:,} reviews ({skipped} skipped)")


def main():
    print("=" * 70)
    print("LOADING DATA INTO CASSANDRA (FIXED FOR LARGE TEXT)")
    print("=" * 70)
    
    hosts = CASSANDRA_HOSTS if CASSANDRA_HOSTS else ["localhost"]
    cluster = Cluster(hosts, port=CASSANDRA_PORT)
    session = cluster.connect()
    
    # Create keyspace if not exists
    session.execute(f"""
        CREATE KEYSPACE IF NOT EXISTS {CASSANDRA_KEYSPACE}
        WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}
    """)
    
    session.set_keyspace(CASSANDRA_KEYSPACE)
    
    # Create tables
    create_tables(session)
    
    # Load data
    load_businesses(session)
    load_reviews(session)
    
    # Verify
    try:
        biz_count = session.execute("SELECT COUNT(*) FROM businesses").one()[0]
    except:
        biz_count = 0
        
    try:
        rev_count = session.execute("SELECT COUNT(*) FROM reviews").one()[0]
    except:
        rev_count = 0
    
    print("\n" + "=" * 70)
    print("LOADING COMPLETE")
    print("=" * 70)
    print(f"📊 Businesses: {biz_count:,}")
    print(f"📊 Reviews: {rev_count:,}")
    
    cluster.shutdown()


if __name__ == "__main__":
    main()