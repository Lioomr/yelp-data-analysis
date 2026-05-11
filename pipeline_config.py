"""
Central paths and defaults for the Yelp ML pipeline.
All scripts should import from here so data locations stay consistent.
"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Raw / renamed Yelp Open Dataset filenames (same folder as scripts by default)
OLD_BUSINESS_NAME = "yelp_academic_dataset_business.json"
OLD_REVIEW_NAME = "yelp_academic_dataset_review.json"
BUSINESS_JSON = os.path.join(ROOT_DIR, "business.json")
REVIEW_JSON = os.path.join(ROOT_DIR, "review.json")
REVIEW_SMALL_JSON = os.path.join(ROOT_DIR, "review_small.json")

# ── Outputs (see FileManager in utils.py)
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
DATA_QUALITY_JSON = os.path.join(OUTPUTS_DIR, "data_quality.json")
EDA_SUMMARY_JSON = os.path.join(OUTPUTS_DIR, "eda_summary.json")
MODEL_METRICS_JSON = os.path.join(OUTPUTS_DIR, "model_metrics.json")
VISUALIZATIONS_DIR = os.path.join(OUTPUTS_DIR, "visualizations")
BEST_MODEL_DIR = os.path.join(ROOT_DIR, "models", "yelp_best_model")

# ── Cleaning / sampling
CLEANING_VERSION = "2.1"
REVIEW_SAMPLE_MAX_LINES = 100_000

# Fast prepare (PREPARE_FAST_SAMPLE=1): max cleaned rows -> review_small.json; max JSONL lines to scan
FAST_SAMPLE_MAX_LINES = int(os.environ.get("FAST_SAMPLE_MAX_LINES", str(REVIEW_SAMPLE_MAX_LINES)))
_FAST_CAP_ENV = os.environ.get("FAST_SAMPLE_RAW_CAP")
if _FAST_CAP_ENV:
    FAST_SAMPLE_RAW_CAP = int(_FAST_CAP_ENV)
else:
    FAST_SAMPLE_RAW_CAP = max(REVIEW_SAMPLE_MAX_LINES * 5, 600_000)

MIN_REVIEW_CHARS = 15
MAX_REVIEW_CHARS = 8000
STAR_MIN = 1.0
STAR_MAX = 5.0
STAR_HIGH_LABEL = 4.0  # label 1 = stars >= this value

# ── Training defaults (also tuned in advanced_yelp_ml.py)
TRAIN_TEST_SEED = 42
TRAIN_FRACTION = 0.8

# ── Cassandra (Docker: set CASSANDRA_HOSTS=cassandra on the Spark app container)
CASSANDRA_HOSTS = [h.strip() for h in os.environ.get("CASSANDRA_HOSTS", "localhost").split(",") if h.strip()]
CASSANDRA_PORT = int(os.environ.get("CASSANDRA_PORT", "9042"))
CASSANDRA_KEYSPACE = os.environ.get("CASSANDRA_KEYSPACE", "yelp")

# Static plots from Cassandra EDA (data_visualization.py)
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
