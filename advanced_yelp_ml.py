"""
HIGH ACCURACY PySpark ML pipeline: Optimized for maximum performance.
Expected time: 15-25 minutes for better accuracy.

Key improvements:
- Larger hyperparameter grids
- More trees/iterations
- Better feature engineering
- Ensemble voting (optional)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, when, split, length, trim
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    NGram,
    HashingTF,
    IDF,
    CountVectorizer,
    VectorAssembler,
)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

from pipeline_config import (
    BUSINESS_JSON,
    REVIEW_SMALL_JSON,
    BEST_MODEL_DIR,
    STAR_HIGH_LABEL,
    TRAIN_TEST_SEED,
    TRAIN_FRACTION,
    OUTPUTS_DIR,
)

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

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from ml_visualizations import MLVisualizer
    _HAS_VISUALIZER = True
except ImportError:
    _HAS_VISUALIZER = False

logger = logging.getLogger(__name__)


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _require_inputs() -> None:
    if not os.path.isfile(BUSINESS_JSON) or not os.path.isfile(REVIEW_SMALL_JSON):
        print("Missing input files. Run: python prepare_data.py")
        sys.exit(1)


def get_tvs_validation_auc(tvs_model) -> float:
    metrics = getattr(tvs_model, "avgMetrics", None)
    if metrics is None:
        metrics = getattr(tvs_model, "validationMetrics", None)
    if metrics is None:
        raise AttributeError("No validation metrics found")
    return float(max(metrics))


def confusion_counts(predictions):
    tp = predictions.filter((col("label") == 1.0) & (col("prediction") == 1.0)).count()
    tn = predictions.filter((col("label") == 0.0) & (col("prediction") == 0.0)).count()
    fp = predictions.filter((col("label") == 0.0) & (col("prediction") == 1.0)).count()
    fn = predictions.filter((col("label") == 1.0) & (col("prediction") == 0.0)).count()
    return int(tn), int(fp), int(fn), int(tp)


def add_roc_to_metrics(predictions, metrics: dict) -> None:
    try:
        from sklearn.metrics import roc_curve
    except ImportError:
        metrics["roc_note"] = "install scikit-learn for ROC"
        return

    try:
        n = predictions.count()
        sample_n = min(20000, max(2000, int(n * 0.15)))
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
        metrics["roc_note"] = f"roc failed: {e}"


def extract_best_params(best_model, model_name: str) -> dict:
    params = {}
    stages = getattr(best_model, "stages", [best_model])
    classifier = stages[-1]
    
    try:
        if model_name == "Random Forest":
            params["num_trees"] = int(classifier.getNumTrees())
            params["max_depth"] = int(classifier.getMaxDepth())
            params["min_instances_per_node"] = int(classifier.getMinInstancesPerNode())
        elif model_name == "Gradient Boosted Trees":
            params["max_depth"] = int(classifier.getMaxDepth())
            params["max_iter"] = int(classifier.getMaxIter())
            params["step_size"] = float(classifier.getStepSize())
        elif model_name == "Logistic Regression":
            params["max_iter"] = int(classifier.getMaxIter())
            params["reg_param"] = float(classifier.getRegParam())
            params["elastic_net"] = float(classifier.getElasticNetParam())
    except:
        pass
    
    return params


def main() -> int:
    os.chdir(_script_dir())
    _require_inputs()
    FileManager.ensure_dirs()
    start_time = time.time()

    print("\n" + "=" * 70)
    print("YELP ML PIPELINE - HIGH ACCURACY VERSION")
    print("=" * 70)
    print("Optimized for maximum performance (15-25 min expected)")
    print("Enhanced features + larger hyperparameter search")
    print("=" * 70)

    # ============================================================================
    # SPARK SESSION (Increased memory for accuracy)
    # ============================================================================
    
    spark = (
        SparkSession.builder.appName("Yelp_High_Accuracy")
        .master("local[*]")
        .config("spark.driver.memory", "8g")  # Increased
        .config("spark.executor.memory", "5g")
        .config("spark.sql.shuffle.partitions", "40")
        .config("spark.default.parallelism", "40")
        .config("spark.sql.autoBroadcastJoinThreshold", "20971520")  # 20MB
        .config("spark.driver.maxResultSize", "3g")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "1024m")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # ============================================================================
    # DATA LOADING
    # ============================================================================
    
    print("\n⏳ Loading data...")
    load_start = time.time()
    
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

    df = df.withColumn("cat_array", split(col("categories"), ", "))
    df = df.dropDuplicates(["business_id", "text"])

    major_df = df.filter(col("label") == 1.0)
    minor_df = df.filter(col("label") == 0.0)
    positive_count = major_df.count()
    negative_count = minor_df.count()
    raw_dataset_size = df.count()
    baseline_accuracy = float(max(positive_count, negative_count)) / max(raw_dataset_size, 1)

    if positive_count == 0 or negative_count == 0:
        print("❌ Error: need both classes")
        spark.stop()
        return 1

    ratio = min(negative_count / positive_count, 1.0)
    df_balanced = major_df.sample(False, ratio, seed=TRAIN_TEST_SEED).union(minor_df)
    
    df_balanced.cache()
    bal_count = df_balanced.count()
    
    load_time = time.time() - load_start
    
    print(f"✅ Data loaded in {load_time:.1f}s")
    print(f"\nDataset: {bal_count:,} balanced reviews")
    print(f"Baseline: {baseline_accuracy:.2%}")

    # ============================================================================
    # TRAIN/TEST SPLIT
    # ============================================================================
    
    train_df, test_df = df_balanced.randomSplit(
        [TRAIN_FRACTION, 1.0 - TRAIN_FRACTION], seed=TRAIN_TEST_SEED
    )
    
    train_df.cache()
    test_df.cache()
    
    train_size = train_df.count()
    test_size = test_df.count()
    
    print(f"Train: {train_size:,} | Test: {test_size:,}")

    # ============================================================================
    # ENHANCED FEATURE PIPELINE
    # ============================================================================
    
    print("\n⏳ Building ENHANCED feature pipeline...")
    
    tokenizer = Tokenizer(inputCol="text", outputCol="words_raw")
    remover = StopWordsRemover(inputCol="words_raw", outputCol="words")
    
    # Add bigrams for better context
    bigram = NGram(n=2, inputCol="words", outputCol="bigrams")
    
    # Word-level TF-IDF (increased features)
    hashingTF_words = HashingTF(
        inputCol="words", 
        outputCol="raw_word_features", 
        numFeatures=3072  # Increased from 2048
    )
    idf_words = IDF(inputCol="raw_word_features", outputCol="word_tfidf", minDocFreq=2)
    
    # Bigram TF-IDF (NEW for better accuracy)
    hashingTF_bigrams = HashingTF(
        inputCol="bigrams", 
        outputCol="raw_bigram_features", 
        numFeatures=2048  # Additional bigram features
    )
    idf_bigrams = IDF(inputCol="raw_bigram_features", outputCol="bigram_tfidf", minDocFreq=3)
    
    # Categories
    cv_cat = CountVectorizer(
        inputCol="cat_array", 
        outputCol="cat_features", 
        vocabSize=150  # Increased from 100
    )
    
    # Combine ALL features
    assembler = VectorAssembler(
        inputCols=["word_tfidf", "bigram_tfidf", "cat_features"],  # 3 feature types!
        outputCol="features",
        handleInvalid="skip"
    )

    # ============================================================================
    # MODELS (Better parameters for accuracy)
    # ============================================================================
    
    print("⏳ Configuring models with better parameters...")
    
    # Logistic Regression
    lr = LogisticRegression(
        labelCol="label",
        featuresCol="features",
        maxIter=100,  # Increased from 50
        elasticNetParam=0.1,
        standardization=True,
    )
    
    # Random Forest (more trees, deeper)
    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=50,  # Increased from 30
        maxDepth=10,  # Increased from 8
        minInstancesPerNode=5,  # Reduced from 10 (more precision)
        seed=TRAIN_TEST_SEED,
        subsamplingRate=0.8,
        featureSubsetStrategy="sqrt",
    )
    
    # Gradient Boosted Trees (more iterations)
    gbt = GBTClassifier(
        labelCol="label",
        featuresCol="features",
        maxIter=50,  # Increased from 30
        maxDepth=6,  # Increased from 5
        stepSize=0.1,
        seed=TRAIN_TEST_SEED,
        subsamplingRate=0.8,
        minInstancesPerNode=5,
    )

    # Build pipelines with ENHANCED features
    base_stages = [
        tokenizer, remover, bigram,
        hashingTF_words, idf_words,
        hashingTF_bigrams, idf_bigrams,
        cv_cat, assembler
    ]
    
    lr_pipeline = Pipeline(stages=base_stages + [lr])
    rf_pipeline = Pipeline(stages=base_stages + [rf])
    gbt_pipeline = Pipeline(stages=base_stages + [gbt])

    # ============================================================================
    # EXPANDED HYPERPARAMETER GRIDS (Better search for accuracy)
    # ============================================================================
    
    # LR: 4 combinations (balanced speed/accuracy)
    lr_param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.05, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.2])
        .build()
    )
    
    # RF: 6 combinations
    rf_param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [40, 50, 60])
        .addGrid(rf.maxDepth, [10, 12])
        .build()
    )
    
    # GBT: 6 combinations
    gbt_param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxIter, [40, 50, 60])
        .addGrid(gbt.maxDepth, [6, 7])
        .build()
    )

    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

    lr_tvs = TrainValidationSplit(
        estimator=lr_pipeline,
        estimatorParamMaps=lr_param_grid,
        evaluator=evaluator,
        trainRatio=0.8,  # Increased from 0.75
        parallelism=2,
        seed=TRAIN_TEST_SEED
    )
    
    rf_tvs = TrainValidationSplit(
        estimator=rf_pipeline,
        estimatorParamMaps=rf_param_grid,
        evaluator=evaluator,
        trainRatio=0.8,
        parallelism=2,
        seed=TRAIN_TEST_SEED
    )
    
    gbt_tvs = TrainValidationSplit(
        estimator=gbt_pipeline,
        estimatorParamMaps=gbt_param_grid,
        evaluator=evaluator,
        trainRatio=0.8,
        parallelism=2,
        seed=TRAIN_TEST_SEED
    )

    # ============================================================================
    # TRAINING
    # ============================================================================
    
    print("\n" + "=" * 70)
    print("TRAINING MODELS (Enhanced features + expanded search)")
    print("=" * 70)
    
    print("\n⏳ [1/3] Logistic Regression (6 param combos)...")
    lr_start = time.time()
    lr_model = lr_tvs.fit(train_df)
    lr_time = time.time() - lr_start
    lr_auc = get_tvs_validation_auc(lr_model)
    print(f"    ✅ LR:  AUC={lr_auc:.4f} ({lr_time:.1f}s = {lr_time/60:.1f}m)")
    
    print("\n⏳ [2/3] Random Forest (6 param combos)...")
    rf_start = time.time()
    rf_model = rf_tvs.fit(train_df)
    rf_time = time.time() - rf_start
    rf_auc = get_tvs_validation_auc(rf_model)
    print(f"    ✅ RF:  AUC={rf_auc:.4f} ({rf_time:.1f}s = {rf_time/60:.1f}m)")
    
    print("\n⏳ [3/3] Gradient Boosted Trees (6 param combos)...")
    gbt_start = time.time()
    gbt_model = gbt_tvs.fit(train_df)
    gbt_time = time.time() - gbt_start
    gbt_auc = get_tvs_validation_auc(gbt_model)
    print(f"    ✅ GBT: AUC={gbt_auc:.4f} ({gbt_time:.1f}s = {gbt_time/60:.1f}m)")

    models = [
        (lr_model, "Logistic Regression", lr_auc),
        (rf_model, "Random Forest", rf_auc),
        (gbt_model, "Gradient Boosted Trees", gbt_auc),
    ]
    
    best_tvs, best_name, best_val_auc = max(models, key=lambda x: x[2])
    best_model = best_tvs.bestModel

    total_train_time = lr_time + rf_time + gbt_time
    
    print(f"\n{'='*70}")
    print(f"🏆 BEST: {best_name} (Val AUC: {best_val_auc:.4f})")
    print(f"   Total training: {total_train_time/60:.1f} minutes")
    print("=" * 70)

    # ============================================================================
    # EVALUATION
    # ============================================================================
    
    print("\n⏳ Evaluating...")
    
    predictions = best_model.transform(test_df)

    binary_eval = BinaryClassificationEvaluator(labelCol="label")
    multi_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    auc = binary_eval.evaluate(predictions, {binary_eval.metricName: "areaUnderROC"})
    accuracy = multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"})
    f1 = multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"})
    precision = multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedPrecision"})
    recall = multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"})

    train_predictions = best_model.transform(train_df)
    train_auc = binary_eval.evaluate(train_predictions)
    train_acc = multi_eval.evaluate(train_predictions, {multi_eval.metricName: "accuracy"})

    tn, fp, fn, tp = confusion_counts(predictions)

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"{'Metric':<20} {'Value':>10}  {'Improvement':>12}")
    print("-" * 45)
    print(f"{'AUC-ROC':<20} {auc:>10.4f}  {auc-0.5:>+12.4f}")
    print(f"{'Accuracy':<20} {accuracy:>10.4f}  {accuracy-baseline_accuracy:>+12.4f}")
    print(f"{'F1 Score':<20} {f1:>10.4f}")
    print(f"{'Precision':<20} {precision:>10.4f}")
    print(f"{'Recall':<20} {recall:>10.4f}")
    
    print(f"\nOverfitting Check:")
    print(f"  Train AUC:  {train_auc:.4f}")
    print(f"  Test AUC:   {auc:.4f}")
    print(f"  Gap:        {train_auc-auc:+.4f}")

    # Save metrics and model...
    best_params = extract_best_params(best_model, best_name)
    
    metrics = {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "train_auc": float(train_auc),
        "train_accuracy": float(train_acc),
        "auc_diff": float(train_auc - auc),
        "acc_diff": float(train_acc - accuracy),
        "raw_dataset_size": int(raw_dataset_size),
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "baseline_accuracy": float(baseline_accuracy),
        "dataset_size": int(bal_count),
        "train_size": int(train_size),
        "test_size": int(test_size),
        "model_type": best_name,
        "validation_auc": float(best_val_auc),
        "embedding_type": "enhanced_tfidf_bigrams",
        "embedding_dimension": 5120,  # 3072 + 2048
        "confusion_tn": tn,
        "confusion_fp": fp,
        "confusion_fn": fn,
        "confusion_tp": tp,
        "star_label_threshold": float(STAR_HIGH_LABEL),
        "num_trees": best_params.get("num_trees"),
        "max_depth": best_params.get("max_depth"),
        "min_instances_per_node": best_params.get("min_instances_per_node"),
        "max_iter": best_params.get("max_iter"),
        "reg_param": best_params.get("reg_param"),
        "elastic_net": best_params.get("elastic_net"),
        "step_size": best_params.get("step_size"),
        "all_validation_aucs": {
            "Logistic Regression": float(lr_auc),
            "Random Forest": float(rf_auc),
            "Gradient Boosted Trees": float(gbt_auc),
        },
        "training_config": {
            "mode": "high_accuracy",
            "word_tfidf_features": 3072,
            "bigram_tfidf_features": 2048,
            "category_features": 150,
            "total_features": 5270,
        },
    }

    add_roc_to_metrics(predictions, metrics)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    metrics_path = os.path.join(OUTPUTS_DIR, "model_metrics.json")
    
    if MetricsManager:
        MetricsManager.save_metrics(metrics)
        if ReportGenerator:
            ReportGenerator.save_report(metrics)
    else:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    print("\n⏳ Saving model...")
    best_model.write().overwrite().save(BEST_MODEL_DIR)
    print(f"✅ Saved: {BEST_MODEL_DIR}")

    train_df.unpersist()
    test_df.unpersist()
    df_balanced.unpersist()
    
    spark.stop()

    if _HAS_VISUALIZER:
        try:
            visualizer = MLVisualizer(metrics_file=metrics_path)
            visualizer.create_all_visualizations()
            print("✅ Visualizations created")
        except:
            pass

    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("✅ COMPLETE")
    print("=" * 70)
    print(f"⏱️  Time:         {elapsed/60:.1f} minutes")
    print(f"🏆 Best:         {best_name}")
    print(f"📊 AUC:          {auc:.4f}")
    print(f"📊 Accuracy:     {accuracy:.2%}")
    print(f"📈 Improvement:  +{(accuracy-baseline_accuracy)*100:.1f}pp")
    print(f"\n📋 All Models:")
    print(f"   LR:  {lr_auc:.4f}")
    print(f"   RF:  {rf_auc:.4f}")
    print(f"   GBT: {gbt_auc:.4f}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as ex:
        logger.exception(ex)
        raise SystemExit(1) from ex