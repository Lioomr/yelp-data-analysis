# 🚀 QUICK START GUIDE - Yelp ML Project

## The Fastest Way to Get Started (3 Steps)

### Step 1️⃣: Install Requirements (2 minutes)
Open CMD in the project folder and run:
```cmd
pip install pyspark pandas plotly streamlit numpy
```

### Step 2️⃣: Train Your Model (10 minutes)
```cmd
python run_full_pipeline.py
```

This automatically:
- ✅ Trains the ML model
- ✅ Creates visualizations
- ✅ Generates reports
- ✅ Saves everything

### Step 3️⃣: View the Dashboard (Immediate)
```cmd
streamlit run dashboard_professional.py
```

Opens in browser: http://localhost:8501

---

## 📊 What You Just Created

After running these 3 commands, you have:

✅ **Trained ML Model** - In `models/yelp_gbt_model/`
✅ **Performance Metrics** - In `outputs/metrics.txt`
✅ **6 Interactive Charts** - In `outputs/visualizations/`
✅ **Professional Dashboard** - Running on localhost:8501

---

## 🎯 Understanding Your Results

### Check Your Metrics
Open this file to see performance:
```
outputs/metrics.txt
```

You'll see something like:
```
AUC-ROC:   0.75
Accuracy:  0.75
F1 Score:  0.75
```

### Check for Overfitting
Look for this in `outputs/metrics.txt`:
```
✅ Good Generalization    (means no overfitting - GOOD!)
AUC Difference: +0.03     (train vs test gap)
```

### View Interactive Charts
Open these in your browser:
- `outputs/visualizations/metrics_gauges.html`
- `outputs/visualizations/train_test_comparison.html`
- `outputs/visualizations/roc_curve.html`

---

## 💡 What Each Score Means

| Score | What It Means | Is It Good? |
|-------|--------------|-----------|
| AUC = 0.75 | Model is 75% better than random | ⭐ Good |
| Accuracy = 75% | 75% of predictions correct | ⭐ Good |
| No Overfitting | Train & test scores are close | ✅ Perfect |

---

## 🎮 Interactive Menu (Even Easier)

Instead of typing commands, use:
```cmd
startup.bat
```

Then just pick from the menu!

---

## 📁 Where Are My Results?

```
📂 outputs/
   ├─ metrics.txt              👈 Read this first!
   ├─ model_metrics.json       (same data, JSON format)
   └─ visualizations/
      ├─ metrics_gauges.html
      ├─ train_test_comparison.html
      ├─ metrics_breakdown.html
      ├─ overfitting_analysis.html
      ├─ hyperparameters.html
      └─ roc_curve.html         👈 Open in browser!

📂 models/
   └─ yelp_gbt_model/          (trained model)
```

---

## ⚡ What Each Command Does

```cmd
REM Run everything automatically
python run_full_pipeline.py

REM Just train the model
python advanced_yelp_ml.py

REM Just create charts
python ml_visualizations.py

REM View dashboard (interactive web interface)
streamlit run dashboard_professional.py

REM View results in CMD
type outputs\metrics.txt
```

---

## ❌ Common Issues & Fixes

### "Python is not found"
- Download from: https://www.python.org/
- Add Python to PATH during installation

### "ModuleNotFoundError"
Run this:
```cmd
pip install pyspark pandas plotly streamlit numpy
```

### "No metrics.txt file"
Run the training first:
```cmd
python advanced_yelp_ml.py
```

### Dashboard won't open
Try this:
```cmd
pip install --upgrade streamlit
streamlit run dashboard_professional.py
```

---

## 🎓 How to Get Better Results

### Method 1: Use More Data (Slower but Better)
Edit `advanced_yelp_ml.py` line ~31:
```python
# Change this:
rev_df = spark.read.json("review_small.json")

# To this:
rev_df = spark.read.json("review.json")
```

Then run: `python advanced_yelp_ml.py`

### Method 2: Train Longer (More Thorough Search)
Edit `advanced_yelp_ml.py` around line ~105:
```python
# Change this:
.addGrid(rf.numTrees, [80, 120]) \

# To this:
.addGrid(rf.numTrees, [50, 100, 150, 200]) \
```

Then run: `python advanced_yelp_ml.py`

---

## 🎯 Success Checklist

After running `python run_full_pipeline.py`:

- [ ] No errors in console
- [ ] `outputs/metrics.txt` exists
- [ ] `outputs/visualizations/` has 6 HTML files
- [ ] `models/yelp_gbt_model/` exists
- [ ] Can open dashboard with `streamlit run dashboard_professional.py`
- [ ] AUC-ROC is > 0.70
- [ ] No overfitting warning

If all checked ✅ - You're done! 🎉

---

## 📖 Learn More

For detailed info, read:
- `README.md` - Full project overview
- `COMMANDS.md` - All available commands
- `outputs/metrics.txt` - Your model's detailed metrics

---

## 🚦 Quick Decision Tree

**Q: I want to start right now!**
A: Run this:
```cmd
python run_full_pipeline.py
```

**Q: I want a fancy web dashboard**
A: Run this:
```cmd
streamlit run dashboard_professional.py
```

**Q: I want better accuracy**
A: Edit line 31 in `advanced_yelp_ml.py` to use full dataset, then train

**Q: I want to see charts**
A: Open any `.html` file in `outputs/visualizations/`

**Q: I want to understand the model**
A: Read `outputs/metrics.txt`

---

## ⏱️ Timing Guide

| Task | Time |
|------|------|
| Install packages | 2 min |
| Train model (small data) | 10 min |
| Generate charts | 1 min |
| View dashboard | Instant |
| **TOTAL** | **~13 min** |

---

## 🎉 You're Ready!

1. Open CMD
2. Navigate to project folder
3. Run: `pip install pyspark pandas plotly streamlit numpy`
4. Run: `python run_full_pipeline.py`
5. Run: `streamlit run dashboard_professional.py`

That's it! Your professional ML dashboard is running! 🚀

---

## 📞 Need Help?

- Check if Python is installed: `python --version`
- Check if pip works: `pip --version`
- Read error messages carefully - they usually tell you what's wrong
- Google the error message + "python"

---

**Happy Machine Learning! 🤖**

*Next time just run: `python run_full_pipeline.py`*
