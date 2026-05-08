import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc, when
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

def main():
    print("🔹 Initializing Spark Session with Cassandra Connector...")
    spark = SparkSession.builder \
        .appName("Yelp Data Processing") \
        .config("spark.cassandra.connection.host", "cassandra") \
        .config("spark.cassandra.connection.port", "9042") \
        .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0") \
        .config("spark.driver.memory", "24g") \
        .config("spark.executor.memory", "24g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    # Decrease log verbosity
    spark.sparkContext.setLogLevel("WARN")

    business_file = "/app/business.json"
    review_file = "/app/review.json"

    print("\n🔹 Step 2 – Load Data using Spark")
    # Load business
    df_business = spark.read.json(business_file)
    print("Schema of Business DataFrame:")
    df_business.printSchema()
    print("First 5 rows of Business DataFrame:")
    df_business.show(5, truncate=False)

    # Load review
    df_review = spark.read.json(review_file)
    print("Schema of Review DataFrame:")
    df_review.printSchema()
    print("First 5 rows of Review DataFrame:")
    df_review.show(5, truncate=False)

    print("\n🔹 Step 4 – Data Cleaning")
    # Count before
    bus_count_before = df_business.count()
    rev_count_before = df_review.count()
    print(f"Businesses before cleaning: {bus_count_before}")
    print(f"Reviews before cleaning: {rev_count_before}")

    # Drop nulls and duplicates (specifically on ID columns if we want, or any row)
    df_business_clean = df_business.dropna().dropDuplicates(['business_id'])
    df_review_clean = df_review.dropna().dropDuplicates(['review_id'])

    # Count after
    bus_count_after = df_business_clean.count()
    rev_count_after = df_review_clean.count()
    print(f"Businesses after cleaning: {bus_count_after}")
    print(f"Reviews after cleaning: {rev_count_after}")

    print("\n🔹 Step 5 – Basic Analysis using Spark")
    print(f"Total number of businesses: {bus_count_after}")
    print(f"Total number of reviews: {rev_count_after}")

    # Count reviews per business
    print("Top 5 Businesses by Review Count (calculated from reviews):")
    df_review_per_bus = df_review_clean.groupBy("business_id").count().orderBy(desc("count"))
    df_review_per_bus.show(5)

    # Filter businesses: stars > 4 and review_count > 50
    print("Businesses where stars > 4 and review_count > 50:")
    df_filtered_bus = df_business_clean.filter((col("stars") > 4) & (col("review_count") > 50))
    # We will show a subset of columns
    if "name" in df_filtered_bus.columns:
        df_filtered_bus.select("business_id", "name", "stars", "review_count", "city").show(10, truncate=False)
    else:
        df_filtered_bus.show(10)

    print("\nStep 6 - MLlib Machine Learning: Predict Business Difficulty")
    # Difficulty label:
    # 0 = low difficulty: stars >= 4.0
    # 1 = medium difficulty: 3.0 <= stars < 4.0
    # 2 = high difficulty: stars < 3.0
    df_ml = df_business_clean.select(
        "business_id", "stars", "review_count", "is_open", "latitude", "longitude"
    ).dropna()

    df_ml = df_ml.withColumn(
        "difficulty",
        when(col("stars") >= 4.0, 0.0)
        .when(col("stars") >= 3.0, 1.0)
        .otherwise(2.0)
    )

    print("Difficulty label distribution:")
    df_ml.groupBy("difficulty").count().orderBy("difficulty").show()

    feature_columns = ["review_count", "is_open", "latitude", "longitude"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df_ml_ready = assembler.transform(df_ml).select("features", "difficulty")

    train_data, test_data = df_ml_ready.randomSplit([0.8, 0.2], seed=42)

    if train_data.count() > 0 and test_data.count() > 0:
        classifier = DecisionTreeClassifier(
            featuresCol="features",
            labelCol="difficulty",
            predictionCol="prediction",
            maxDepth=5,
            seed=42
        )
        model = classifier.fit(train_data)
        predictions = model.transform(test_data)

        evaluator = MulticlassClassificationEvaluator(
            labelCol="difficulty",
            predictionCol="prediction",
            metricName="accuracy"
        )
        accuracy = evaluator.evaluate(predictions)
        print(f"MLlib Decision Tree accuracy: {accuracy:.2%}")
        print("Sample ML predictions:")
        predictions.select("difficulty", "prediction", "features").show(10, truncate=False)
    else:
        print("Not enough data to train and test the MLlib model.")

    print("\nStep 7 - Store Data using Cassandra")
    # Creating tables in Cassandra using cassandra-driver from Python
    from cassandra.cluster import Cluster
    print("Connecting to Cassandra to create keyspace and tables...")
    cluster = Cluster(['cassandra'], port=9042)
    session = cluster.connect()
    
    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS yelp 
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}
    """)
    session.set_keyspace('yelp')

    # Create Business table 
    # From business.json: business_id, name, address, city, state, postal_code, latitude, longitude, stars, review_count, is_open, attributes, categories, hours
    session.execute("""
        CREATE TABLE IF NOT EXISTS businesses (
            business_id text PRIMARY KEY,
            name text,
            address text,
            city text,
            state text,
            postal_code text,
            stars float,
            review_count int,
            is_open int,
            categories text
        )
    """)

    # Create Review table
    # review_id, user_id, business_id, stars, useful, funny, cool, text, date
    session.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id text PRIMARY KEY,
            user_id text,
            business_id text,
            stars float,
            useful int,
            funny int,
            cool int,
            text text,
            date text
        )
    """)
    
    # Need to match PySpark DF columns to Cassandra table columns exactly or select proper ones
    df_business_insert = df_business_clean.select(
        "business_id", "name", "address", "city", "state", "postal_code",
        "stars", "review_count", "is_open", "categories"
    )
    
    df_review_insert = df_review_clean.select(
        "review_id", "user_id", "business_id", "stars", "useful", "funny", "cool", "text", "date"
    )

    print("Writing businesses to Cassandra (this might take a few minutes)...")
    df_business_insert.write.format("org.apache.spark.sql.cassandra") \
        .mode('append') \
        .options(table="businesses", keyspace="yelp") \
        .save()
    print("✅ Businesses saved to Cassandra")

    print("Writing reviews to Cassandra (this might take longer due to size)...")
    df_review_insert.write.format("org.apache.spark.sql.cassandra") \
        .mode('append') \
        .options(table="reviews", keyspace="yelp") \
        .save()
    print("✅ Reviews saved to Cassandra")

    spark.stop()

if __name__ == "__main__":
    main()
