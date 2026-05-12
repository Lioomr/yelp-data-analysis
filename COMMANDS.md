# 🍽️ Yelp ML Project - CMD Commands & Setup Guide

## Installation & Setup

### Step 1: Install Python Dependencies
Open CMD and run:
```cmd
pip install pyspark pandas plotly streamlit numpy
```

Or use requirements.txt:
```cmd
pip install -r requirements.txt
```

### Step 2: Navigate to Project Directory
```cmd
cd C:\Users\user\Desktop\yelp-data-analysis
```

## Running the Complete Pipeline

### Option A: Use the Interactive Menu (Easiest)
```cmd
startup.bat
```
Then select option 1 for full pipeline

### Option B: Run Full Pipeline via Command Line
```cmd
python run_full_pipeline.py
```

**What it does:**
- ✅ Trains the ML model using Spark
- ✅ Generates interactive visualizations
- ✅ Creates detailed reports
- ✅ Saves all outputs automatically

**Expected output:**
```
==============================================================================
  🍽️  YELP ML PIPELINE - COMPLETE WORKFLOW
==============================================================================
⏳ Training Gradient Boosted Trees with Cross-Validation...
✅ Training completed
...
🎉 PIPELINE COMPLETED SUCCESSFULLY!
```

## Individual Commands

### 1. Train Model Only
```cmd
python advanced_yelp_ml.py
```

**Output:**
- `outputs/model_metrics.json` - Metrics in JSON format
- `outputs/metrics.txt` - Human-readable report
- `models/yelp_gbt_model/` - Trained model files

**Runtime:** 5-15 minutes

### 2. Generate Visualizations
```cmd
python ml_visualizations.py
```

**Output:**
- `outputs/visualizations/metrics_gauges.html`
- `outputs/visualizations/train_test_comparison.html`
- `outputs/visualizations/metrics_breakdown.html`
- `outputs/visualizations/overfitting_analysis.html`
- `outputs/visualizations/hyperparameters.html`
- `outputs/visualizations/roc_curve.html`

### 3. View Professional Dashboard
```cmd
streamlit run dashboard_professional.py
```

**Access:** http://localhost:8501

**Features:**
- Overview tab with core metrics
- Performance tab with detailed charts
- Advanced tab with ROC curves
- Exports tab with all outputs

Press `Ctrl+C` to stop the server.

## Quick Reference Commands

```cmd
# Full pipeline
python run_full_pipeline.py

# Train model
python advanced_yelp_ml.py

# Generate charts
python ml_visualizations.py

# Launch dashboard
streamlit run dashboard_professional.py

# View metrics
type outputs\metrics.txt

# Check metrics JSON
type outputs\model_metrics.json

# Use interactive menu
startup.bat
```

## File Organization

After running the pipeline, your project will have:

```
outputs/
├── metrics.txt                 # Text report
├── model_metrics.json          # JSON metrics
└── visualizations/
    ├── metrics_gauges.html
    ├── train_test_comparison.html
    ├── metrics_breakdown.html
    ├── overfitting_analysis.html
    ├── hyperparameters.html
    └── roc_curve.html

models/
└── yelp_gbt_model/
    ├── metadata/
    ├── data/
    └── treesMetadata/
```

## Understanding the Output

### Metrics Report (outputs/metrics.txt)
```
==============================================================
YELP ML MODEL - PERFORMANCE METRICS
==============================================================

TEST SET METRICS:
  AUC-ROC:   0.7500
  Accuracy:  0.7500
  F1 Score:  0.7500
  Precision: 0.7500
  Recall:    0.7500

TRAINING SET METRICS:
  AUC-ROC:   0.7850
  Accuracy:  0.7850

OVERFITTING ANALYSIS:
  AUC Difference:      +0.0350
  Accuracy Difference: +0.0350

MODEL CONFIGURATION:
  Number of Trees: 80
  Max Depth:       15
  Min Instances:   1
  Dataset Size:    10,465
```

### Key Metrics Explained
| Metric | What it means | Good range |
|--------|--------------|-----------|
| **AUC-ROC** | Overall discrimination ability | > 0.75 |
| **Accuracy** | Correct predictions overall | > 0.75 |
| **Precision** | Correct positive predictions | > 0.75 |
| **Recall** | Correct positive out of all positives | > 0.75 |
| **F1-Score** | Balance between precision & recall | > 0.75 |

### Overfitting Gap Interpretation
- **< 0.05** - ✅ Excellent generalization
- **0.05-0.10** - ✅ Good generalization
- **0.10-0.15** - ⚠️ Monitor overfitting
- **> 0.15** - ❌ Likely overfitting

## Advanced: Customizing the Model

To improve accuracy, edit `advanced_yelp_ml.py`:

### Use Full Dataset (Better Results, Slower)
```python
# Change from:
rev_df = spark.read.json("review_small.json")

# To:
rev_df = spark.read.json("review.json")
```

### Increase Cross-Validation Folds
```python
# Change from:
numFolds=3

# To:
numFolds=5
```

### Expand Hyperparameter Search
```python
# Change from:
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [80, 120]) \
    .addGrid(rf.maxDepth, [15, 20]) \
    .build()

# To:
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100, 150]) \
    .addGrid(rf.maxDepth, [10, 15, 20]) \
    .build()
```

## Troubleshooting

### "Python is not recognized"
Install Python from python.org and add to PATH

### "ModuleNotFoundError: No module named 'pyspark'"
```cmd
pip install pyspark
```

### "No metrics found"
Run the training first:
```cmd
python advanced_yelp_ml.py
```

### Dashboard doesn't load
```cmd
pip install --upgrade streamlit
streamlit run dashboard_professional.py
```

### Model training is slow
Use `review_small.json` instead of `review.json`
Reduce `numFolds` from 5 to 3

## Complete Workflow Example

```cmd
REM Navigate to project
cd C:\Users\user\Desktop\yelp-data-analysis

REM Install dependencies
pip install -r requirements.txt

REM Train model
python advanced_yelp_ml.py

REM Generate visualizations
python ml_visualizations.py

REM Launch dashboard
streamlit run dashboard_professional.py

REM View metrics in another CMD window
type outputs\metrics.txt
```

## Batch Processing

Create a file called `run_all.bat`:
```batch
@echo off
echo Running full ML pipeline...
python advanced_yelp_ml.py
if errorlevel 1 (
    echo Training failed
    pause
    exit /b 1
)
echo.
echo Generating visualizations...
python ml_visualizations.py
if errorlevel 1 (
    echo Visualization failed
    pause
    exit /b 1
)
echo.
echo All complete! Opening dashboard...
streamlit run dashboard_professional.py
```

Then run:
```cmd
run_all.bat
```

## Performance Tips

### Fast Run (< 10 minutes)
```cmd
REM Use small dataset
python advanced_yelp_ml.py
REM This uses review_small.json by default
```

### Best Accuracy (30-60 minutes)
Edit `advanced_yelp_ml.py`:
- Change to `review.json`
- Set `numFolds=5`
- Expand `paramGrid` with more options

## Next Steps

1. **First time?** Run: `python run_full_pipeline.py`
2. **View results?** Run: `streamlit run dashboard_professional.py`
3. **Better accuracy?** Edit `advanced_yelp_ml.py` and modify hyperparameters
4. **Analyze more?** Check `outputs/visualizations/` folder

## Files Overview

| File | Purpose | Command |
|------|---------|---------|
| `advanced_yelp_ml.py` | ML model training | `python advanced_yelp_ml.py` |
| `ml_visualizations.py` | Chart generation | `python ml_visualizations.py` |
| `dashboard_professional.py` | Web interface | `streamlit run dashboard_professional.py` |
| `run_full_pipeline.py` | Complete workflow | `python run_full_pipeline.py` |
| `utils.py` | Shared functions | (imported by other scripts) |
| `startup.bat` | Interactive menu | `startup.bat` |

## Questions?

- Check `README.md` for project overview
- Check `outputs/metrics.txt` for detailed metrics
- Check dashboard "Exports" tab for raw data
- Check visualizations in `outputs/visualizations/` folder

---

**Start with:** `python run_full_pipeline.py`

Then view: `streamlit run dashboard_professional.py`
