"""
PySpark ML pipeline: review text + categories -> binary star label.
TRIES sentence-transformers, automatically falls back to Word2Vec if it fails.

Outputs: outputs/model_metrics.json, outputs/metrics.txt, models/yelp_best_model/

─── Integration with the 4-file pipeline ─────────────────────────────────────
  prepare_data.py      → cleans business.json + review_small.json (run first)
  data_visualization.py→ Cassandra EDA plots (run independently after ingestion)
  advanced_yelp_ml.py  → THIS FILE — Spark training + evaluation
  ml_visualizations.py → MLVisualizer called HERE at the end automatically
──────────────────────────────────────────────────────────────────────────────

Execution order:
  1.  python prepare_data.py          (or PREPARE_FAST_SAMPLE=1 python prepare_data.py)
  2.  python data_visualization.py    (optional EDA — needs Cassandra running)
  3.  python advanced_yelp_ml.py      (this file — trains + saves + visualises)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, when, split, length, trim, udf
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    NGram,
    HashingTF,
    IDF,
    CountVectorizer,
    VectorAssembler,
    Word2Vec,
)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, VectorUDT

# ── Shared pipeline config (used by all 4 files) ──────────────────────────────
from pipeline_config import (
    BUSINESS_JSON,
    REVIEW_SMALL_JSON,
    BEST_MODEL_DIR,
    STAR_HIGH_LABEL,
    TRAIN_TEST_SEED,
    TRAIN_FRACTION,
    OUTPUTS_DIR,           # needed to resolve outputs/ consistently
)

# ── Optional shared utilities (utils.py) ─────────────────────────────────────
try:
    from utils import FileManager, MetricsManager, ReportGenerator
except ImportError:
    MetricsManager = None
    ReportGenerator = None

    class FileManager:
        @staticmethod
        def ensure_dirs():
            os.makedirs("outputs", exist_ok=True)
            os.makedirs("models", exist_ok=True)

# ── ml_visualizations.py integration ─────────────────────────────────────────
# Import MLVisualizer so we can auto-generate all HTML/PNG charts at the end.
# If the file is not on sys.path yet, add the script directory first.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from ml_visualizations import MLVisualizer
    _HAS_VISUALIZER = True
except ImportError:
    _HAS_VISUALIZER = False
    print("⚠️  ml_visualizations.py not found — skipping auto-visualisation step.")

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================

def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _require_inputs() -> None:
    """
    Mirrors the check in prepare_data.py: both cleaned files must exist before
    Spark can load them.  Run `python prepare_data.py` if they are missing.
    """
    missing = [p for p in (BUSINESS_JSON, REVIEW_SMALL_JSON) if not os.path.isfile(p)]
    if missing:
        print("Missing input files — run prepare_data.py first:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)


def get_tvs_validation_auc(tvs_model) -> float:
    metrics = getattr(tvs_model, "avgMetrics", None)
    if metrics is None:
        metrics = getattr(tvs_model, "validationMetrics", None)
    if metrics is None:
        raise AttributeError("TrainValidationSplitModel has no avgMetrics or validationMetrics")
    return float(max(metrics))


def add_roc_to_metrics(predictions, metrics: dict) -> None:
    """
    Attach roc_fpr / roc_tpr lists to the metrics dict using a stratified sample.
    ml_visualizations.py reads these keys to draw the empirical ROC curve.
    Falls back gracefully when scikit-learn is not installed.
    """
    try:
        from sklearn.metrics import roc_curve
    except ImportError:
        logger.warning("scikit-learn not installed; skipping ROC curve arrays")
        metrics["roc_note"] = "install scikit-learn for ROC arrays"
        return

    try:
        n = predictions.count()
        sample_n = min(25_000, max(1_000, int(n * 0.15)))
        sampled = (
            predictions.select("label", "probability")
            .orderBy(rand(seed=TRAIN_TEST_SEED))
            .limit(sample_n)
            .toPandas()
        )
        sampled["score"] = sampled["probability"].apply(lambda v: float(v[1]))
        fpr, tpr, _ = roc_curve(sampled["label"], sampled["score"])
        metrics["roc_fpr"] = [float(x) for x in fpr]
        metrics["roc_tpr"] = [float(x) for x in tpr]
    except Exception as e:
        metrics["roc_note"] = f"roc extraction failed: {e}"


def confusion_counts(predictions):
    """Return (TN, FP, FN, TP) counts from a predictions DataFrame."""
    tp = predictions.filter((col("label") == 1.0) & (col("prediction") == 1.0)).count()
    tn = predictions.filter((col("label") == 0.0) & (col("prediction") == 0.0)).count()
    fp = predictions.filter((col("label") == 0.0) & (col("prediction") == 1.0)).count()
    fn = predictions.filter((col("label") == 1.0) & (col("prediction") == 0.0)).count()
    return int(tn), int(fp), int(fn), int(tp)


def _extract_best_params(best_tvs_model, model_name: str) -> dict:
    """
    Extract best hyperparameter values from the winning TrainValidationSplitModel
    so that ml_visualizations.py can display them in the hyperparameter table.
    Keys expected by MLVisualizer.create_hyperparameter_summary():
      num_trees, max_depth, min_instances_per_node, max_iter, reg_param, elastic_net
    """
    params: dict = {}
    best = best_tvs_model.bestModel

    # Walk down Pipeline stages to find the actual classifier stage
    stages = getattr(best, "stages", [best])
    classifier = stages[-1]  # always the last stage in our pipelines

    if model_name == "Random Forest":
        params["num_trees"]              = int(getattr(classifier, "getNumTrees",
                                               lambda: classifier._java_obj.getNumTrees())())
        params["max_depth"]              = int(classifier.getMaxDepth())
        params["min_instances_per_node"] = int(classifier.getMinInstancesPerNode())

    elif model_name == "Gradient Boosted Trees":
        params["max_depth"]  = int(classifier.getMaxDepth())
        params["max_iter"]   = int(classifier.getMaxIter())
        params["step_size"]  = float(classifier.getStepSize())

    elif model_name == "Logistic Regression":
        params["max_iter"]   = int(classifier.getMaxIter())
        params["reg_param"]  = float(classifier.getRegParam())
        params["elastic_net"] = float(classifier.getElasticNetParam())

    return params


# =============================================================================
# EMBEDDING GENERATION WITH AUTOMATIC FALLBACK
# =============================================================================

def generate_embeddings_with_sentence_transformers(
    reviews_list: list[str], batch_size: int = 32
):
    """
    Try sentence-transformers (all-MiniLM-L6-v2).
    Returns (embeddings_array | None, dimension, use_spark_fallback: bool).
    prepare_data.py already cleaned the text, so we pass it directly.
    """
    try:
        print("\n" + "=" * 70)
        print("ATTEMPTING SENTENCE-TRANSFORMERS EMBEDDINGS")
        print("=" * 70)

        from sentence_transformers import SentenceTransformer

        print("✅ sentence-transformers imported successfully")
        print("Loading model: all-MiniLM-L6-v2...")

        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        print(f"Generating embeddings for {len(reviews_list):,} reviews...")
        start_time = time.time()

        embeddings = model.encode(
            reviews_list,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        elapsed = time.time() - start_time
        print(f"✅ Generated embeddings: shape {embeddings.shape}")
        print(f"   Time: {elapsed:.2f}s ({len(reviews_list)/elapsed:.0f} reviews/sec)")

        return embeddings, 384, False   # success — no fallback needed

    except ImportError as e:
        print(f"\n⚠️  sentence-transformers import failed: {e}")
        print("   Will use PySpark Word2Vec instead")
        return None, 0, True

    except Exception as e:
        print(f"\n⚠️  Embedding generation failed: {e}")
        print("   Will use PySpark Word2Vec instead")
        return None, 0, True


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    os.chdir(_script_dir())
    _require_inputs()          # same guard as prepare_data.py uses
    FileManager.ensure_dirs()
    start_time = time.time()

    print("\n" + "=" * 70)
    print("YELP ML PIPELINE — SMART EMBEDDINGS")
    print("=" * 70)
    print("Tries sentence-transformers, falls back to Word2Vec if needed.")
    print("Auto-runs ml_visualizations.py on completion.")
    print("=" * 70)

    # ── Spark session ─────────────────────────────────────────────────────────
    spark = (
        SparkSession.builder.appName("Yelp_Smart_Embedding_Model")
        .master("local[4]")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "80")
        .config("spark.sql.debug.maxToStringFields", "200")
        .config("spark.driver.maxResultSize", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # ── Load data cleaned by prepare_data.py ──────────────────────────────────
    print("\nLoading cleaned data (produced by prepare_data.py)...")
    biz_df = spark.read.json(BUSINESS_JSON).select("business_id", "categories", "stars")
    rev_df = spark.read.json(REVIEW_SMALL_JSON).select("business_id", "text")

    labels = biz_df.select(
        "business_id",
        when(col("stars") >= float(STAR_HIGH_LABEL), 1.0).otherwise(0.0).alias("label"),
        "categories",
    )

    df = (
        rev_df.join(labels, on="business_id", how="inner")
        .withColumn("text", trim(col("text")))
        .filter(length(col("text")) >= 10)
        .filter(col("categories").isNotNull())
        .filter(length(trim(col("categories"))) > 0)
    )

    # categories cleaned to ", "-separated by prepare_data.format_categories_csv()
    df = df.withColumn("cat_array", split(col("categories"), ", "))
    df = df.dropDuplicates(["business_id", "text"])

    # ── Balance classes ───────────────────────────────────────────────────────
    major_df = df.filter(col("label") == 1.0)
    minor_df = df.filter(col("label") == 0.0)
    positive_count  = major_df.count()
    negative_count  = minor_df.count()
    raw_dataset_size = df.count()
    baseline_accuracy = float(max(positive_count, negative_count)) / max(raw_dataset_size, 1)

    if positive_count == 0 or negative_count == 0:
        print("Error: need both classes in join result. Check business/review overlap.")
        spark.stop()
        return 1

    ratio = min(negative_count / positive_count, 1.0)
    df_balanced = major_df.sample(False, ratio, seed=TRAIN_TEST_SEED).union(minor_df).cache()
    bal_count = df_balanced.count()

    print(f"\nDataset Statistics:")
    print(f"  Raw dataset size:                       {raw_dataset_size:,}")
    print(f"  Positive (label=1, stars>={STAR_HIGH_LABEL}):        {positive_count:,}")
    print(f"  Negative (label=0):                     {negative_count:,}")
    print(f"  Baseline majority-class accuracy:       {baseline_accuracy:.4f}")
    print(f"  Balanced dataset size:                  {bal_count:,}")

    # ── Embedding strategy ────────────────────────────────────────────────────
    use_spark_fallback = False
    embedding_type = "unknown"

    df_pandas = df_balanced.select("business_id", "text", "label", "cat_array").toPandas()
    reviews_text = df_pandas["text"].tolist()

    embeddings_array, embedding_dim, use_spark_fallback = (
        generate_embeddings_with_sentence_transformers(reviews_text)
    )

    if not use_spark_fallback:
        # ── Path A: sentence-transformers ─────────────────────────────────────
        print("\n✅ Using Sentence-Transformers embeddings (384-dim)")
        embedding_type = "sentence_transformers"

        df_pandas["text_embedding"] = embeddings_array.tolist()
        df_with_embeddings = spark.createDataFrame(df_pandas)

        def list_to_vector(lst):
            return Vectors.dense(lst)

        list_to_vector_udf = udf(list_to_vector, VectorUDT())
        df_with_embeddings = df_with_embeddings.withColumn(
            "embedding_features",
            list_to_vector_udf(col("text_embedding")),
        )

        cv_cat = CountVectorizer(inputCol="cat_array", outputCol="cat_features", vocabSize=150)
        assembler = VectorAssembler(
            inputCols=["embedding_features", "cat_features"],
            outputCol="features",
            handleInvalid="skip",
        )
        base_pipeline = Pipeline(stages=[cv_cat, assembler])

    else:
        # ── Path B: PySpark Word2Vec + TF-IDF (fallback) ──────────────────────
        print("\n" + "=" * 70)
        print("USING PYSPARK WORD2VEC FALLBACK")
        print("=" * 70)
        embedding_type = "word2vec_tfidf"
        embedding_dim  = 256

        df_with_embeddings = df_balanced

        tokenizer = Tokenizer(inputCol="text", outputCol="words_raw")
        remover   = StopWordsRemover(inputCol="words_raw", outputCol="words_clean")
        ngram     = NGram(n=2, inputCol="words_clean", outputCol="bigrams")

        word2vec = Word2Vec(
            vectorSize=256,
            minCount=2,
            maxIter=15,
            seed=TRAIN_TEST_SEED,
            inputCol="words_clean",
            outputCol="word2vec_features",
        )

        hashingTF = HashingTF(inputCol="bigrams", outputCol="raw_tf", numFeatures=5_000)
        idf       = IDF(inputCol="raw_tf", outputCol="tfidf_features")
        cv_cat    = CountVectorizer(inputCol="cat_array", outputCol="cat_features", vocabSize=200)

        assembler = VectorAssembler(
            inputCols=["word2vec_features", "tfidf_features", "cat_features"],
            outputCol="features",
            handleInvalid="skip",
        )

        base_pipeline = Pipeline(
            stages=[tokenizer, remover, ngram, word2vec, hashingTF, idf, cv_cat, assembler]
        )

    # ── Train / Test split ────────────────────────────────────────────────────
    train_df, test_df = df_with_embeddings.randomSplit(
        [TRAIN_FRACTION, 1.0 - TRAIN_FRACTION], seed=TRAIN_TEST_SEED
    )
    train_size = train_df.count()
    test_size  = test_df.count()

    print(f"\nTrain/Test Split:")
    print(f"  Train: {train_size:,}  |  Test: {test_size:,}")

    # ── Classifiers ───────────────────────────────────────────────────────────
    lr = LogisticRegression(
        labelCol="label",
        featuresCol="features",
        maxIter=100,
        elasticNetParam=0.1,
        standardization=True,
    )
    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=60,
        maxDepth=10,
        minInstancesPerNode=3,
        seed=TRAIN_TEST_SEED,
        subsamplingRate=0.8,
    )
    gbt = GBTClassifier(
        labelCol="label",
        featuresCol="features",
        maxIter=50,
        maxDepth=7,
        stepSize=0.1,
        seed=TRAIN_TEST_SEED,
    )

    binary_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

    lr_pipeline  = Pipeline(stages=base_pipeline.getStages() + [lr])
    rf_pipeline  = Pipeline(stages=base_pipeline.getStages() + [rf])
    gbt_pipeline = Pipeline(stages=base_pipeline.getStages() + [gbt])

    # ── Hyperparameter grids ──────────────────────────────────────────────────
    lr_param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam,        [0.001, 0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0,   0.1,  0.3])
        .build()
    )
    rf_param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [40, 60, 80])
        .addGrid(rf.maxDepth, [8, 10, 12])
        .build()
    )
    gbt_param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxIter, [30, 50, 70])
        .addGrid(gbt.maxDepth, [5,  7,  9])
        .build()
    )

    def _make_tvs(pipeline, grid):
        return TrainValidationSplit(
            estimator=pipeline,
            estimatorParamMaps=grid,
            evaluator=binary_evaluator,
            trainRatio=0.8,
            parallelism=2,
            seed=TRAIN_TEST_SEED,
        )

    lr_tvs  = _make_tvs(lr_pipeline,  lr_param_grid)
    rf_tvs  = _make_tvs(rf_pipeline,  rf_param_grid)
    gbt_tvs = _make_tvs(gbt_pipeline, gbt_param_grid)

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING MODELS (LR → RF → GBT)")
    print("=" * 70)

    print("  [1/3] Fitting Logistic Regression...")
    t0 = time.time()
    lr_model = lr_tvs.fit(train_df)
    lr_val_auc = get_tvs_validation_auc(lr_model)
    print(f"        LR  validation AUC: {lr_val_auc:.4f}  ({time.time()-t0:.1f}s)")

    print("  [2/3] Fitting Random Forest...")
    t0 = time.time()
    rf_model = rf_tvs.fit(train_df)
    rf_val_auc = get_tvs_validation_auc(rf_model)
    print(f"        RF  validation AUC: {rf_val_auc:.4f}  ({time.time()-t0:.1f}s)")

    print("  [3/3] Fitting Gradient Boosted Trees...")
    t0 = time.time()
    gbt_model = gbt_tvs.fit(train_df)
    gbt_val_auc = get_tvs_validation_auc(gbt_model)
    print(f"        GBT validation AUC: {gbt_val_auc:.4f}  ({time.time()-t0:.1f}s)")

    # ── Select best model ─────────────────────────────────────────────────────
    models = [
        (lr_model,  "Logistic Regression",      lr_val_auc),
        (rf_model,  "Random Forest",             rf_val_auc),
        (gbt_model, "Gradient Boosted Trees",    gbt_val_auc),
    ]
    best_tvs, best_model_name, best_val_auc = max(models, key=lambda x: x[2])
    best_model = best_tvs.bestModel

    print(f"\n✅ Best model: {best_model_name}  (validation AUC {best_val_auc:.4f})")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    predictions = best_model.transform(test_df)

    multi_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction"
    )

    auc       = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
    accuracy  = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
    f1        = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
    precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
    recall    = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})

    train_predictions = best_model.transform(train_df)
    train_auc      = binary_evaluator.evaluate(train_predictions, {binary_evaluator.metricName: "areaUnderROC"})
    train_accuracy = multi_evaluator.evaluate(train_predictions, {multi_evaluator.metricName: "accuracy"})

    auc_diff = train_auc - auc
    acc_diff = train_accuracy - accuracy

    print(f"\n{'Metric':<22} {'Test':>8}  {'Train':>8}  {'Gap':>8}")
    print("-" * 52)
    print(f"{'AUC-ROC':<22} {auc:>8.4f}  {train_auc:>8.4f}  {auc_diff:>+8.4f}")
    print(f"{'Accuracy':<22} {accuracy:>8.4f}  {train_accuracy:>8.4f}  {acc_diff:>+8.4f}")
    print(f"{'F1 Score':<22} {f1:>8.4f}")
    print(f"{'Precision':<22} {precision:>8.4f}")
    print(f"{'Recall':<22} {recall:>8.4f}")

    tn, fp, fn, tp = confusion_counts(predictions)
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn:,}  FP={fp:,}")
    print(f"  FN={fn:,}  TP={tp:,}")

    # ── Build metrics dict ────────────────────────────────────────────────────
    # Include all keys expected by ml_visualizations.MLVisualizer so every
    # chart (gauges, hyperparameter table, ROC, radar, donut, overfitting) works.
    best_params = _extract_best_params(best_tvs, best_model_name)

    metrics = {
        # ── Core performance (read by create_metrics_gauge_charts, breakdown, radar, donut)
        "auc":       float(auc),
        "accuracy":  float(accuracy),
        "f1":        float(f1),
        "precision": float(precision),
        "recall":    float(recall),

        # ── Train metrics (read by create_train_test_comparison, overfitting)
        "train_auc":      float(train_auc),
        "train_accuracy": float(train_accuracy),
        "auc_diff":       float(auc_diff),
        "acc_diff":       float(acc_diff),

        # ── Dataset stats (read by create_hyperparameter_summary)
        "raw_dataset_size": int(raw_dataset_size),
        "positive_count":   int(positive_count),
        "negative_count":   int(negative_count),
        "baseline_accuracy": float(baseline_accuracy),
        "dataset_size":     int(bal_count),
        "train_size":       int(train_size),
        "test_size":        int(test_size),

        # ── Model identity (read by create_hyperparameter_summary)
        "model_type":       best_model_name,
        "validation_auc":   float(best_val_auc),
        "embedding_type":   embedding_type,
        "embedding_dimension": embedding_dim,

        # ── Best hyperparameters (read by create_hyperparameter_summary)
        # All keys default to None so the visualiser handles them gracefully.
        "num_trees":              best_params.get("num_trees"),
        "max_depth":              best_params.get("max_depth"),
        "min_instances_per_node": best_params.get("min_instances_per_node"),
        "max_iter":               best_params.get("max_iter"),
        "reg_param":              best_params.get("reg_param"),
        "elastic_net":            best_params.get("elastic_net"),
        "step_size":              best_params.get("step_size"),

        # ── Confusion matrix (available for custom downstream use)
        "confusion_tn": tn,
        "confusion_fp": fp,
        "confusion_fn": fn,
        "confusion_tp": tp,

        # ── Thresholds / seeds (reproducibility)
        "star_label_threshold": float(STAR_HIGH_LABEL),
        "train_test_seed":      int(TRAIN_TEST_SEED),
        "train_fraction":       float(TRAIN_FRACTION),

        # ── All three models' validation AUCs (informational)
        "all_validation_aucs": {
            "Logistic Regression":   float(lr_val_auc),
            "Random Forest":         float(rf_val_auc),
            "Gradient Boosted Trees": float(gbt_val_auc),
        },

        "training_config": {
            "models_tested": ["LogisticRegression", "RandomForest", "GradientBoostedTrees"],
        },
    }

    # Attach ROC arrays for create_roc_simulation() empirical branch
    add_roc_to_metrics(predictions, metrics)

    # ── Persist metrics ───────────────────────────────────────────────────────
    if MetricsManager is not None:
        MetricsManager.save_metrics(metrics)
        if ReportGenerator is not None:
            ReportGenerator.save_report(metrics)
    else:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        metrics_path = os.path.join(OUTPUTS_DIR, "model_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Metrics saved → {metrics_path}")

        # Plain-text report
        report_lines = [
            "=" * 60,
            "YELP ML PIPELINE — FINAL REPORT",
            "=" * 60,
            f"Best Model:        {best_model_name}",
            f"Embedding:         {embedding_type} ({embedding_dim}-dim)",
            "",
            "TEST METRICS",
            f"  AUC-ROC:   {auc:.4f}",
            f"  Accuracy:  {accuracy:.4f}",
            f"  F1 Score:  {f1:.4f}",
            f"  Precision: {precision:.4f}",
            f"  Recall:    {recall:.4f}",
            "",
            "TRAIN METRICS",
            f"  Train AUC:      {train_auc:.4f}",
            f"  Train Accuracy: {train_accuracy:.4f}",
            f"  AUC Gap:        {auc_diff:+.4f}",
            f"  Acc Gap:        {acc_diff:+.4f}",
            "",
            "DATASET",
            f"  Raw size:    {raw_dataset_size:,}",
            f"  Balanced:    {bal_count:,}",
            f"  Train split: {train_size:,}",
            f"  Test split:  {test_size:,}",
        ]
        report_text = "\n".join(report_lines)
        with open(os.path.join(OUTPUTS_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
            f.write(report_text)
        print(report_text)

    # ── Save best model ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    best_model.write().overwrite().save(BEST_MODEL_DIR)
    print(f"✅ Model saved → {BEST_MODEL_DIR}")

    spark.stop()

    # ── Auto-run ml_visualizations.py ────────────────────────────────────────
    # MLVisualizer reads outputs/model_metrics.json that we just wrote, so this
    # must happen after spark.stop() to free driver memory.
    print("\n" + "=" * 70)
    print("GENERATING VISUALISATIONS  (ml_visualizations.py)")
    print("=" * 70)

    if _HAS_VISUALIZER:
        metrics_file = os.path.join(OUTPUTS_DIR, "model_metrics.json")
        visualizer = MLVisualizer(metrics_file=metrics_file)
        success = visualizer.create_all_visualizations()
        if success:
            print("✅ All charts saved → outputs/visualizations/")
        else:
            print("⚠️  Some visualisations failed — check plotly installation.")
    else:
        print("⚠️  ml_visualizations.py not importable — run it separately:")
        print("     python ml_visualizations.py")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"⏱️  Total time:      {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"📊 Embedding method: {embedding_type}")
    print(f"🏆 Best model:      {best_model_name}  (AUC {auc:.4f})")
    print(f"📁 Outputs:         {OUTPUTS_DIR}/")

    return 0


# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as ex:
        logger.exception(ex)
        raise SystemExit(1) from ex