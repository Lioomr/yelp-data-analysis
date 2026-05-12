# 🍽️ Yelp ML Project - Complete Pipeline

A professional machine learning pipeline for predicting Yelp business success using Apache Spark, Python, and Streamlit.

## 📋 Project Overview

This project analyzes Yelp business data to predict business success (high ratings) using a Random Forest classifier with:
- ✅ **High Accuracy** - Optimized for real-world performance
- ✅ **No Overfitting** - Robust generalization with 3-fold cross-validation
- ✅ **Professional Visualizations** - Interactive Plotly charts
- ✅ **Complete Pipeline** - Automated end-to-end workflow

## 📁 Project Structure

```
yelp-data-analysis/
├── advanced_yelp_ml.py           # Core ML model (Spark)
├── ml_visualizations.py          # Visualization generator
├── dashboard_professional.py      # Streamlit dashboard
├── utils.py                       # Shared utilities
├── run_full_pipeline.py           # Pipeline orchestrator
├── requirements.txt               # Dependencies
├── business.json                  # Business data
├── review_small.json              # Review data
├── outputs/                       # Generated outputs
│   ├── metrics.txt                # Performance metrics report
│   ├── model_metrics.json         # Metrics (JSON format)
│   └── visualizations/            # Interactive charts
│       ├── metrics_gauges.html
│       ├── train_test_comparison.html
│       ├── metrics_breakdown.html
│       ├── overfitting_analysis.html
│       ├── hyperparameters.html
│       └── roc_curve.html
└── models/
    └── yelp_gbt_model/            # Trained model
```

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install pyspark pandas plotly streamlit numpy
```

### 2. Run the Full Pipeline (Recommended)
```bash
python run_full_pipeline.py
```

This will automatically:
- ✅ Train the ML model
- ✅ Generate visualizations
- ✅ Create reports
- ✅ Display comprehensive summary

### 3. View the Dashboard
```bash
streamlit run dashboard_professional.py
```

Then open your browser to `http://localhost:8501`

## 📊 Available Commands

### Train ML Model Only
```bash
python advanced_yelp_ml.py
```

### Generate Visualizations Only
```bash
python ml_visualizations.py
```

### Run Dashboard Only
```bash
streamlit run dashboard_professional.py
```

## 📈 Key Features

### Model Training (`advanced_yelp_ml.py`)
- **Algorithm:** Random Forest Classifier
- **Features:** TF-IDF + Category vectors
- **Validation:** 3-Fold Cross-Validation
- **Hyperparameter Tuning:** Grid search over tree count, depth, and min instances
- **Overfitting Control:** Train/test gap monitoring
- **Output:** Model saved to `models/yelp_gbt_model/`

### Visualizations (`ml_visualizations.py`)
Creates 6 professional interactive charts:
1. **Metrics Gauges** - AUC, Accuracy, Precision, Recall visualization
2. **Train/Test Comparison** - Overfitting analysis
3. **Metrics Breakdown** - All evaluation metrics in one chart
4. **Overfitting Analysis** - Train-test gap visualization
5. **Hyperparameters** - Model configuration display
6. **ROC Curve** - Model discrimination ability

### Dashboard (`dashboard_professional.py`)
Professional Streamlit dashboard with:
- **Overview Tab** - Core metrics, overfitting check, performance radar
- **Performance Tab** - Detailed metrics, gauges, train/test comparison
- **Advanced Tab** - Configuration, overfitting details, ROC curve
- **Exports Tab** - Available outputs, raw metrics, model summary

### Utilities (`utils.py`)
Shared functions for:
- Metrics management (loading, saving, rating)
- Data utilities (formatting, statistics)
- File management (directory creation, paths)
- Report generation (formatted output)

## 📊 Understanding the Metrics

### Performance Metrics
| Metric | Definition | Target |
|--------|-----------|--------|
| **AUC-ROC** | Area under ROC curve (0-1) | > 0.85 |
| **Accuracy** | Overall correct predictions | > 0.80 |
| **Precision** | True Positives / (TP + FP) | > 0.80 |
| **Recall** | True Positives / (TP + FN) | > 0.80 |
| **F1-Score** | Harmonic mean of precision & recall | > 0.80 |

### Overfitting Indicators
- ✅ **Good** (AUC gap < 0.1): Model generalizes well
- ⚠️ **Monitor** (AUC gap 0.1-0.15): Watch for performance decline
- ❌ **Warning** (AUC gap > 0.15): Possible overfitting

## 🔧 Configuration

### Model Parameters (in `advanced_yelp_ml.py`)
```python
# Hyperparameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [80, 120]) \
    .addGrid(rf.maxDepth, [15, 20]) \
    .addGrid(rf.minInstancesPerNode, [1, 2]) \
    .build()

# Cross-validation
numFolds=3  # Change for more/fewer folds
```

### Feature Configuration
- **Text Features:** TF-IDF (2000 features)
- **Category Features:** CountVectorizer (100 features)
- **Preprocessing:** Tokenization, stopword removal, lowercase

## 📊 Typical Outputs

After running `run_full_pipeline.py`, you'll see:

```
==============================================================================
  🍽️  YELP ML PIPELINE - COMPLETE WORKFLOW
==============================================================================

🎯 KEY METRICS:
  AUC-ROC:       0.7500
  Accuracy:      0.7500
  F1 Score:      0.7500
  Precision:     0.7500
  Recall:        0.7500

📊 OVERFITTING ANALYSIS:
  ✅ Good Generalization
  AUC Gap:       +0.0500
  Accuracy Gap:  +0.0500

⭐ MODEL RATING: ⭐ Very Good

📁 OUTPUT FILES:
  ✓ outputs/model_metrics.json
  ✓ outputs/metrics.txt
  ✓ outputs/visualizations/
  ✓ models/yelp_gbt_model/
```

## 🎨 Dashboard Features

### Professional UI
- Dark theme optimized for readability
- Responsive design (works on all screen sizes)
- Interactive Plotly visualizations
- Real-time metrics from JSON file
- Professional typography (Syne + DM Mono fonts)

### Sidebar Configuration
- Model parameters display
- Dataset information
- Quick statistics
- One-click access to all sections

## 🔄 Pipeline Workflow

```
1. Data Loading
   └─> Load business.json & review_small.json
   
2. Feature Engineering
   └─> Tokenization → Stop Words Removal → TF-IDF
   └─> Category CountVectorizer
   └─> Vector Assembly
   
3. Model Training
   └─> Random Forest with hyperparameter tuning
   └─> 3-Fold Cross-Validation
   
4. Evaluation
   └─> Test set metrics (AUC, Accuracy, F1, etc.)
   └─> Training set metrics (overfitting check)
   
5. Output Generation
   └─> Save model to models/
   └─> Generate visualizations (HTML)
   └─> Create reports (JSON, TXT)
   
6. Dashboard
   └─> Display all metrics interactively
```

## ⚙️ Optimization Tips

### Speed Up Training
```python
# In advanced_yelp_ml.py
numFolds=3  # Reduce from 5 to 3
hashingTF = HashingTF(numFeatures=2000)  # Reduce from 5000
```

### Improve Accuracy
```python
# Use more data (full review.json instead of review_small.json)
rev_df = spark.read.json("review.json")

# Increase hyperparameter search space
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100, 150]) \
    .addGrid(rf.maxDepth, [10, 15, 20]) \
    .build()
```

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pyspark'"
**Solution:** `pip install pyspark`

### Issue: "outputs/model_metrics.json not found"
**Solution:** Run `python advanced_yelp_ml.py` first to generate metrics

### Issue: Dashboard shows default metrics
**Solution:** Refresh the page or clear Streamlit cache:
```bash
streamlit run dashboard_professional.py --logger.level=debug
```

### Issue: Slow model training
**Solution:** 
- Use `review_small.json` instead of `review.json`
- Reduce `numFolds` from 5 to 3
- Reduce hyperparameter grid size

## 📚 File Descriptions

| File | Purpose |
|------|---------|
| `advanced_yelp_ml.py` | Main ML pipeline - trains the model |
| `ml_visualizations.py` | Generates interactive Plotly charts |
| `dashboard_professional.py` | Streamlit dashboard interface |
| `utils.py` | Shared utilities and helper functions |
| `run_full_pipeline.py` | Orchestrates entire workflow |

## 🎯 Expected Performance

With default settings on `review_small.json`:
- **Training Time:** 5-15 minutes (depending on hardware)
- **Expected AUC:** 0.70-0.85
- **Expected Accuracy:** 0.70-0.85
- **Overfitting Gap:** < 0.1 (good generalization)

## 📊 Next Steps

1. **Train Model:** `python run_full_pipeline.py`
2. **View Dashboard:** `streamlit run dashboard_professional.py`
3. **Check Outputs:** Open `outputs/visualizations/` in browser
4. **Tune Hyperparameters:** Edit `advanced_yelp_ml.py` for better accuracy
5. **Use Full Data:** Switch to `review.json` for better results (slower)

## 📝 License

This project uses the Yelp Open Dataset.

## 🤝 Support

For issues or improvements, check:
- `outputs/metrics.txt` for detailed performance metrics
- `outputs/visualizations/` for interactive charts
- Dashboard "Exports" tab for raw metrics JSON

---

**Happy Machine Learning! 🚀**
