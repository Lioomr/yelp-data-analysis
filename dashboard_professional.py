"""
Professional Yelp ML Dashboard — Fully Integrated
Combines: dashboard_professional.py + ml_visualizations.py +
          prepare_data.py (data quality) + data_visualization.py (Cassandra EDA) +
          advanced_yelp_ml.py (all metric keys + model comparison)
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    import streamlit.components.v1 as components
except ImportError:
    components = None

try:
    from utils import MetricsManager
    _HAS_UTILS = True
except ImportError:
    _HAS_UTILS = False

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Yelp ML Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0D0F14;
    color: #E8E3D5;
}
.stApp { background-color: #0D0F14; }

[data-testid="stSidebar"] {
    background: #13161E;
    border-right: 1px solid #2A2D38;
}
[data-testid="metric-container"] {
    background: #13161E;
    border: 1px solid #2A2D38;
    border-radius: 12px;
    padding: 16px;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: #F4A261; }
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #F4A261 !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    color: #8A8FA8 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    color: #E8E3D5 !important;
}
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #F4A261;
    margin-bottom: 8px;
    border-left: 3px solid #F4A261;
    padding-left: 10px;
}
.info-box {
    background: #13161E;
    border: 1px solid #2A2D38;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 8px 0;
}
.pipeline-step {
    background: #13161E;
    border: 1px solid #2A2D38;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 6px 0;
    border-left: 3px solid #F4A261;
}
hr { border-color: #2A2D38 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# THEME CONSTANTS
# ─────────────────────────────────────────────
ORANGE = "#F4A261"
TEAL   = "#2EC4B6"
RED    = "#E63946"
GREEN  = "#2A9D8F"
GRID   = dict(gridcolor="#2A2D38", zerolinecolor="#2A2D38")
DARK_LAYOUT = dict(
    paper_bgcolor="#0D0F14",
    plot_bgcolor="#13161E",
    font=dict(family="DM Mono, monospace", color="#E8E3D5", size=12),
)

# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
VIZ_DIR = os.path.join(OUTPUTS_DIR, "visualizations")


@st.cache_data
def load_metrics() -> dict:
    """Load model_metrics.json (written by advanced_yelp_ml.py via MetricsManager)."""
    path = os.path.join(OUTPUTS_DIR, "model_metrics.json")
    if _HAS_UTILS:
        try:
            return MetricsManager.load_metrics() or {}
        except Exception:
            pass
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_data_quality() -> dict:
    """Load data_quality.json (written by prepare_data.py)."""
    if _HAS_UTILS:
        try:
            return MetricsManager.load_data_quality() or {}
        except Exception:
            pass
    path = os.path.join(OUTPUTS_DIR, "data_quality.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_eda_summary() -> dict:
    """Load eda_summary.json (written by prepare_data.write_quality_artifact)."""
    path = os.path.join(OUTPUTS_DIR, "eda_summary.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_metrics_txt() -> str:
    """Load plain-text report (written by advanced_yelp_ml.py)."""
    path = os.path.join(OUTPUTS_DIR, "metrics.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return ""


def list_html_visuals() -> list[str]:
    """Return sorted list of HTML files in outputs/visualizations/ (from ml_visualizations.py)."""
    if os.path.isdir(VIZ_DIR):
        return sorted(f for f in os.listdir(VIZ_DIR) if f.endswith(".html"))
    return []


def embed_html_visual(filename: str, height: int = 520) -> None:
    """Embed a saved HTML visualization from ml_visualizations.py."""
    path = os.path.join(VIZ_DIR, filename)
    if os.path.isfile(path) and components:
        with open(path, "r", encoding="utf-8") as fh:
            components.html(fh.read(), height=height, scrolling=True)
    elif os.path.isfile(path):
        st.info(f"Open locally: `{path}`")
    else:
        st.caption(f"⚠ `{filename}` not found — run `python ml_visualizations.py`.")


def fmt(v, decimals=4):
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v) if v is not None else "N/A"


def get_overfitting_status(m: dict) -> tuple[str, str]:
    auc_gap = abs(m.get("auc_diff", 0))
    acc_gap = abs(m.get("acc_diff", 0))
    if auc_gap < 0.05 and acc_gap < 0.05:
        return "✅ Minimal Overfitting", "green"
    if auc_gap < 0.1 and acc_gap < 0.1:
        return "✅ Good Generalization", "green"
    if auc_gap < 0.15 or acc_gap < 0.15:
        return "⚠️ Moderate Overfitting", "orange"
    return "❌ High Overfitting", "red"


# ─────────────────────────────────────────────
# LOAD ALL DATA
# ─────────────────────────────────────────────
metrics = load_metrics()
data_quality = load_data_quality()
eda_summary = load_eda_summary()
metrics_txt = load_metrics_txt()

_model_title = metrics.get("model_type") or "Logistic Regression / Random Forest"
_star_thr = metrics.get("star_label_threshold", 4.0)
_embed = metrics.get("embedding_type", "Word2Vec")
_embed_dim = metrics.get("embedding_dimension", "—")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">⚙ Model Configuration</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
      <b>Model</b><br/>{_model_title}<br/><br/>
      <b>Embedding</b><br/>{_embed} ({_embed_dim}-dim)<br/><br/>
      <b>Validation AUC</b><br/>{fmt(metrics.get('validation_auc', 0))}<br/><br/>
      <b>Num Trees</b><br/>{metrics.get('num_trees', '— if RF')}<br/><br/>
      <b>Max Depth</b><br/>{metrics.get('max_depth', '— if RF')}<br/><br/>
      <b>Reg / ElasticNet</b><br/>{fmt(metrics.get('reg_param', 'N/A'))} / {fmt(metrics.get('elastic_net', '—'))}<br/><br/>
      <b>Features</b><br/>Word2Vec + TF-IDF + Categories<br/><br/>
      <b>Validation</b><br/>TrainValidationSplit
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">📦 Dataset Info</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
      <b>Source</b><br/>business.json + review_small.json<br/><br/>
      <b>Raw Reviews</b><br/>{fmt(metrics.get('raw_dataset_size', 0))}<br/><br/>
      <b>Balanced Records</b><br/>{fmt(metrics.get('dataset_size', 0))}<br/><br/>
      <b>Train / Test</b><br/>{fmt(metrics.get('train_size', 0))} / {fmt(metrics.get('test_size', 0))}<br/><br/>
      <b>Label Threshold</b><br/>High Rating (≥{_star_thr}) vs Low<br/><br/>
      <b>Balance</b><br/>Undersampled Majority
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">📊 Quick Stats</p>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.metric("AUC", fmt(metrics.get("auc", 0)))
        st.metric("Precision", fmt(metrics.get("precision", 0)))
    with col_r:
        st.metric("Accuracy", fmt(metrics.get("accuracy", 0)))
        st.metric("Recall", fmt(metrics.get("recall", 0)))

    # Pipeline status
    st.markdown('<p class="section-label">🔄 Pipeline Status</p>', unsafe_allow_html=True)
    steps = [
        ("prepare_data.py", bool(data_quality)),
        ("advanced_yelp_ml.py", bool(metrics)),
        ("ml_visualizations.py", bool(list_html_visuals())),
        ("data_visualization.py", os.path.isdir(os.path.join(BASE_DIR, "plots"))),
    ]
    for step, done in steps:
        icon = "✅" if done else "⏳"
        st.markdown(f'<div class="pipeline-step">{icon} <code>{step}</code></div>',
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
if not metrics:
    st.warning(
        "No trained metrics found. Run `prepare_data.py` → `advanced_yelp_ml.py` → "
        "`ml_visualizations.py` to populate this dashboard."
    )

st.markdown(f"""
<h1 style="font-size:2.2rem; margin-bottom:0">
  🍽️ Yelp Business ML
  <span style="color:#F4A261">Analytics Dashboard</span>
</h1>
<p style="color:#8A8FA8; font-size:0.85rem; margin-top:4px">
  {_model_title} · {_embed} embeddings · Spark ML · Train/Validation Split
</p>
""", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🎯 Performance",
    "📈 Advanced",
    "🔍 Data & EDA",
    "🗂️ All Visualizations",
    "💾 Exports",
])

# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-label">Core Metrics</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🎯 Accuracy",  fmt(metrics.get("accuracy", 0)),  help="Overall correct predictions")
    c2.metric("📐 F1-Score",  fmt(metrics.get("f1", 0)),         help="Harmonic mean of precision & recall")
    c3.metric("🎪 Precision", fmt(metrics.get("precision", 0)), help="True positives / predicted positives")
    c4.metric("🔁 Recall",    fmt(metrics.get("recall", 0)),    help="True positives / actual positives")
    c5.metric("📡 AUC-ROC",   fmt(metrics.get("auc", 0)),       help="Area under the ROC curve")

    st.markdown(
        "<div class='info-box' style='margin-top:12px;'>"
        f"<b>Trained Samples</b>&nbsp;&nbsp;{fmt(metrics.get('train_size', 0))} (train)&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"<b>Test Samples</b>&nbsp;&nbsp;{fmt(metrics.get('test_size', 0))} (test)&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"<b>Raw Dataset</b>&nbsp;&nbsp;{fmt(metrics.get('raw_dataset_size', 0))} reviews"
        "</div>", unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown('<p class="section-label">Overfitting Detection</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        train_acc = metrics.get("train_accuracy", 0)
        test_acc  = metrics.get("accuracy", 0)
        fig_gap = go.Figure()
        fig_gap.add_trace(go.Bar(
            x=["Training", "Testing"],
            y=[train_acc, test_acc],
            marker_color=[TEAL, ORANGE],
            text=[fmt(train_acc), fmt(test_acc)],
            textposition="outside",
            textfont=dict(family="Syne", size=13, color="#E8E3D5"),
            width=0.4,
        ))
        fig_gap.update_layout(
            **DARK_LAYOUT,
            title="Train vs Test Accuracy (Overfitting Check)",
            yaxis=dict(range=[0, max(train_acc, test_acc) + 0.12], **GRID),
            xaxis=dict(**GRID),
            showlegend=False,
            height=380,
        )
        st.plotly_chart(fig_gap, use_container_width=True)

    with col_b:
        auc_gap = abs(metrics.get("auc_diff", 0))
        acc_gap_abs = abs(metrics.get("acc_diff", 0))
        status_text, status_color = get_overfitting_status(metrics)
        color_map = {"green": GREEN, "orange": ORANGE, "red": RED}
        border = color_map.get(status_color, "#8A8FA8")
        st.markdown(f"""
        <div class="info-box" style="text-align:center; margin-top:40px; border: 2px solid {border}">
          <div style="font-size:2.5rem">{'✅' if 'Good' in status_text or 'Minimal' in status_text else '⚠️' if 'Moderate' in status_text else '❌'}</div>
          <div style="font-family:Syne,sans-serif; font-size:1.1rem; font-weight:800; margin:8px 0">
            {status_text}
          </div>
          <div style="margin-top:10px; color:#8A8FA8; font-size:0.75rem">
            <b>AUC Gap:</b> {auc_gap:+.4f}<br/>
            <b>Accuracy Gap:</b> {acc_gap_abs:+.4f}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-label">Performance Metrics Breakdown</p>', unsafe_allow_html=True)
    metrics_list = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
        "Score":  [
            metrics.get("accuracy", 0),
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("f1", 0),
            metrics.get("auc", 0),
        ],
    }).sort_values("Score", ascending=True)

    fig_bar = px.bar(
        metrics_list, x="Score", y="Metric", orientation="h",
        color="Score", color_continuous_scale=["#E76F51", ORANGE, GREEN],
        title="All Evaluation Metrics (Test Set)",
    )
    fig_bar.update_layout(**DARK_LAYOUT, showlegend=False, height=380)
    fig_bar.update_traces(text=metrics_list["Score"].round(4), textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Model comparison (all_validation_aucs from advanced_yelp_ml.py)
    all_aucs: dict = metrics.get("all_validation_aucs", {})
    if all_aucs:
        st.markdown("---")
        st.markdown('<p class="section-label">Model Selection — Validation AUC Comparison</p>', unsafe_allow_html=True)
        auc_df = pd.DataFrame(
            {"Model": list(all_aucs.keys()), "Validation AUC": list(all_aucs.values())}
        ).sort_values("Validation AUC", ascending=True)
        fig_models = px.bar(
            auc_df, x="Validation AUC", y="Model", orientation="h",
            color="Validation AUC", color_continuous_scale=["#E76F51", ORANGE, GREEN],
            title="All Models — Validation AUC (TrainValidationSplit)",
            text="Validation AUC",
        )
        fig_models.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_models.update_layout(**DARK_LAYOUT, showlegend=False, height=300,
                                  xaxis=dict(range=[0, 1], **GRID), yaxis=dict(**GRID))
        st.plotly_chart(fig_models, use_container_width=True)
        st.caption(f"🏆 Best model selected: **{_model_title}** (Validation AUC {fmt(metrics.get('validation_auc', 0))})")

# ══════════════════════════════════════════════
# TAB 2 — PERFORMANCE
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-label">Train vs Test Performance</p>', unsafe_allow_html=True)
    comparison_df = pd.DataFrame({
        "Metric":   ["AUC-ROC", "Accuracy"],
        "Training": [metrics.get("train_auc", 0), metrics.get("train_accuracy", 0)],
        "Testing":  [metrics.get("auc", 0),        metrics.get("accuracy", 0)],
    })
    fig_cmp = go.Figure(data=[
        go.Bar(x=comparison_df["Metric"], y=comparison_df["Training"], name="Training", marker_color=TEAL),
        go.Bar(x=comparison_df["Metric"], y=comparison_df["Testing"],  name="Testing",  marker_color=ORANGE),
    ])
    fig_cmp.update_layout(
        **DARK_LAYOUT, title="Performance Comparison (Overfitting Monitor)",
        barmode="group", yaxis=dict(range=[0, 1], **GRID), xaxis=dict(**GRID), height=400,
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-label">Metric Gauges</p>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)

    def _gauge(val, title, bar_color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 1]},
                "bar":  {"color": bar_color},
                "steps": [
                    {"range": [0, 0.5], "color": RED},
                    {"range": [0.5, 0.8], "color": ORANGE},
                    {"range": [0.8, 1],   "color": GREEN},
                ],
                "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.75, "value": 0.75},
            },
            title={"text": title},
        ))
        fig.update_layout(**DARK_LAYOUT, height=300, showlegend=False)
        return fig

    with g1:
        st.plotly_chart(_gauge(metrics.get("auc", 0), "AUC-ROC", ORANGE), use_container_width=True)
    with g2:
        st.plotly_chart(_gauge(metrics.get("precision", 0), "Precision", GREEN), use_container_width=True)
    with g3:
        st.plotly_chart(_gauge(metrics.get("recall", 0), "Recall", TEAL), use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-label">ROC Curve</p>', unsafe_allow_html=True)
    auc_value = metrics.get("auc", 0.75)
    roc_fpr = metrics.get("roc_fpr")
    roc_tpr = metrics.get("roc_tpr")
    fig_roc = go.Figure()
    if (isinstance(roc_fpr, list) and isinstance(roc_tpr, list)
            and len(roc_fpr) == len(roc_tpr) and len(roc_fpr) > 2):
        fig_roc.add_trace(go.Scatter(
            x=roc_fpr, y=roc_tpr, mode="lines",
            name=f"ROC (test sample, AUC={auc_value:.4f})",
            line=dict(color=ORANGE, width=3), fill="tozeroy",
        ))
        roc_title = f"ROC Curve — empirical (AUC = {auc_value:.4f})"
    else:
        fpr = np.linspace(0, 1, 200)
        tpr = np.clip(1 - np.exp(-5 * fpr), 0, 1) * auc_value / 0.618 + (1 - auc_value / 0.618)
        tpr = np.clip(tpr, 0, 1)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"Approximate ROC (AUC={auc_value:.4f})",
            line=dict(color=ORANGE, width=3), fill="tozeroy",
        ))
        roc_title = "ROC — approximate (run with scikit-learn for empirical curve)"
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color="#8A8FA8", width=2, dash="dash"),
    ))
    fig_roc.update_layout(
        **DARK_LAYOUT, title=roc_title,
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=420,
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    if metrics.get("roc_note"):
        st.caption(str(metrics["roc_note"]))

    # Confusion matrix
    if all(k in metrics for k in ("confusion_tp", "confusion_fp", "confusion_fn", "confusion_tn")):
        st.markdown("---")
        st.markdown('<p class="section-label">Confusion Matrix (Test Set)</p>', unsafe_allow_html=True)
        tn = metrics["confusion_tn"]
        fp = metrics["confusion_fp"]
        fn = metrics["confusion_fn"]
        tp = metrics["confusion_tp"]

        z = [[tn, fp], [fn, tp]]
        fig_cm = go.Figure(go.Heatmap(
            z=z,
            x=["Predicted 0 (Low)", "Predicted 1 (High)"],
            y=["Actual 0 (Low)", "Actual 1 (High)"],
            text=[[str(v) for v in row] for row in z],
            texttemplate="%{text}",
            colorscale=[[0, "#13161E"], [1, ORANGE]],
            showscale=False,
        ))
        fig_cm.update_layout(**DARK_LAYOUT, title="Confusion Matrix", height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

        total = tn + fp + fn + tp
        col_cm1, col_cm2, col_cm3, col_cm4 = st.columns(4)
        col_cm1.metric("True Negative",  f"{tn:,}", f"{tn/total:.1%}" if total else "")
        col_cm2.metric("False Positive", f"{fp:,}", f"{fp/total:.1%}" if total else "")
        col_cm3.metric("False Negative", f"{fn:,}", f"{fn/total:.1%}" if total else "")
        col_cm4.metric("True Positive",  f"{tp:,}", f"{tp/total:.1%}" if total else "")

# ══════════════════════════════════════════════
# TAB 3 — ADVANCED
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-label">Model Configuration</p>', unsafe_allow_html=True)
    _nt = metrics.get("num_trees")
    _md = metrics.get("max_depth")
    _mi = metrics.get("min_instances_per_node", metrics.get("min_instances", 1))
    config_df = pd.DataFrame({
        "Parameter": [
            "Algorithm", "Embedding Type", "Embedding Dimension",
            "Number of Trees", "Max Depth", "Min Instances / Node",
            "Max Iter (LR)", "Reg Param", "Elastic Net",
            "Dataset Size (balanced)", "Train / Test Split",
            "Label Threshold", "Random Seed",
        ],
        "Value": [
            str(metrics.get("model_type", "—")),
            str(metrics.get("embedding_type", "Word2Vec")),
            str(metrics.get("embedding_dimension", "—")),
            _nt if _nt is not None else "— (not RF)",
            _md if _md is not None else "— (not RF)",
            _mi,
            metrics.get("max_iter", "— (not LR)"),
            fmt(metrics.get("reg_param"), 4) if metrics.get("reg_param") is not None else "N/A",
            fmt(metrics.get("elastic_net"), 3) if metrics.get("elastic_net") is not None else "— (not LR)",
            fmt(metrics.get("dataset_size", 0)) if metrics.get("dataset_size") else "N/A",
            f"{fmt(metrics.get('train_fraction', 0.8), 0)} / {fmt(1 - metrics.get('train_fraction', 0.8), 0)}",
            f"≥ {metrics.get('star_label_threshold', 4.0)} stars = High",
            str(metrics.get("train_test_seed", "—")),
        ],
    })
    st.dataframe(config_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<p class="section-label">Metrics Radar Chart</p>', unsafe_allow_html=True)
    categories = ["AUC-ROC", "Accuracy", "Precision", "Recall", "F1 Score"]
    values = [
        metrics.get("auc", 0), metrics.get("accuracy", 0),
        metrics.get("precision", 0), metrics.get("recall", 0),
        metrics.get("f1", 0),
    ]
    fig_radar = go.Figure(go.Scatterpolar(
        r=values + values[:1], theta=categories + categories[:1],
        fill="toself", name="Model Metrics",
        line=dict(color=ORANGE, width=3),
    ))
    fig_radar.update_layout(
        **DARK_LAYOUT,
        title="Model Performance Radar",
        polar=dict(
            bgcolor="#0D0F14",
            radialaxis=dict(range=[0, 1], visible=True, tickfont=dict(color="#E8E3D5")),
            angularaxis=dict(tickfont=dict(color="#E8E3D5")),
        ),
        height=500,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-label">Metric Strength Donut</p>', unsafe_allow_html=True)
    fig_donut = go.Figure(go.Pie(
        labels=categories,
        values=values,
        hole=0.55,
        marker=dict(colors=[ORANGE, GREEN, "#E76F51", "#264653", "#8AB17D"]),
        textinfo="label+percent", hoverinfo="label+value",
    ))
    fig_donut.update_layout(**DARK_LAYOUT, title="Metric Strength Distribution", height=450)
    st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-label">Overfitting Analysis Details</p>', unsafe_allow_html=True)
    auc_g = metrics.get("auc_diff", 0)
    acc_g = metrics.get("acc_diff", 0)
    analysis_df = pd.DataFrame({
        "Metric":  ["AUC-ROC", "Accuracy"],
        "Train":   [fmt(metrics.get("train_auc", 0)), fmt(metrics.get("train_accuracy", 0))],
        "Test":    [fmt(metrics.get("auc", 0)),        fmt(metrics.get("accuracy", 0))],
        "Gap":     [f"{auc_g:+.4f}", f"{acc_g:+.4f}"],
        "Status":  [
            "✅ Good" if abs(auc_g) < 0.1 else "⚠️ Monitor",
            "✅ Good" if abs(acc_g) < 0.1 else "⚠️ Monitor",
        ],
    })
    st.dataframe(analysis_df, use_container_width=True, hide_index=True)

    # Positive / negative class breakdown (advanced_yelp_ml.py)
    pos = metrics.get("positive_count")
    neg = metrics.get("negative_count")
    baseline = metrics.get("baseline_accuracy")
    if pos is not None and neg is not None:
        st.markdown("---")
        st.markdown('<p class="section-label">Class Balance (before undersampling)</p>', unsafe_allow_html=True)
        cb1, cb2, cb3 = st.columns(3)
        cb1.metric("Positive (High ★)", fmt(pos))
        cb2.metric("Negative (Low ★)",  fmt(neg))
        cb3.metric("Baseline Accuracy", fmt(baseline) if baseline else "N/A",
                   help="Accuracy of majority-class classifier")

# ══════════════════════════════════════════════
# TAB 4 — DATA & EDA
# ══════════════════════════════════════════════
with tab4:
    # ── prepare_data.py output ──
    st.markdown('<p class="section-label">Data Quality Report (prepare_data.py)</p>', unsafe_allow_html=True)
    if data_quality:
        # High-level summary cards
        biz = data_quality.get("business", {})
        rev_full = data_quality.get("review_full", {})
        rev_small = data_quality.get("review_small", {})

        dq1, dq2, dq3, dq4 = st.columns(4)
        dq1.metric("Businesses Written",    fmt(biz.get("business_records_written", 0)))
        dq2.metric("Biz Dropped (dup)",     fmt(biz.get("business_dropped_duplicate_id", 0)))
        dq3.metric("Reviews Written",       fmt(rev_full.get("review_records_written", rev_full.get("review_records_written_small", 0))))
        dq4.metric("Review Small (sample)", fmt(rev_small.get("review_small_lines", 0)))

        st.markdown("---")
        st.markdown('<p class="section-label">Full Quality JSON</p>', unsafe_allow_html=True)
        st.json(data_quality, expanded=False)
    else:
        st.info("Run `python prepare_data.py` to generate `outputs/data_quality.json`.")

    # ── eda_summary.json ──
    if eda_summary:
        st.markdown("---")
        st.markdown('<p class="section-label">EDA Summary (prepare_data.py)</p>', unsafe_allow_html=True)
        es1, es2, es3 = st.columns(3)
        es1.metric("Businesses", fmt(eda_summary.get("n_businesses", 0)))
        es2.metric("Reviews (small sample)", fmt(eda_summary.get("n_reviews_small", 0)))
        es3.metric("Fast Mode", "✅ Yes" if eda_summary.get("fast_mode") else "❌ No")
        st.caption(f"Cleaning version: `{eda_summary.get('cleaning_version', '—')}`")

    st.markdown("---")
    st.markdown('<p class="section-label">EDA Charts (ml_visualizations.py)</p>', unsafe_allow_html=True)
    for fname, title in [
        ("eda_stars.html",         "Business star rating distribution"),
        ("eda_review_length.html", "Review text length distribution"),
    ]:
        st.subheader(title)
        embed_html_visual(fname, height=480)

    # ── data_visualization.py Cassandra EDA static plots ──
    st.markdown("---")
    st.markdown('<p class="section-label">Cassandra EDA Plots (data_visualization.py)</p>',
                unsafe_allow_html=True)
    cassandra_plots_dir = os.path.join(BASE_DIR, "plots")
    cassandra_plots = {
        "businesses_per_city_bar.png":     "Top 10 Cities by Number of Businesses",
        "businesses_per_category_bar.png": "Top 10 Categories by Number of Businesses",
        "stars_distribution_pie.png":      "Business Star Ratings Distribution",
        "categories_distribution_pie.png": "Top 5 Business Categories Distribution",
        "stars_histogram.png":             "Histogram of Business Star Ratings",
        "review_counts_histogram.png":     "Histogram of Review Counts",
    }
    found_any = False
    for fname, title in cassandra_plots.items():
        path = os.path.join(cassandra_plots_dir, fname)
        if os.path.isfile(path):
            found_any = True
            st.subheader(title)
            st.image(path)

    if not found_any:
        st.info(
            "No Cassandra EDA plots found. Run `python data_visualization.py` "
            "(requires a Cassandra node with the Yelp keyspace loaded). "
            "Expected plot directory: `./plots/`"
        )

    st.markdown("---")
    st.markdown(
        "**Cleaning policy (prepare_data.py):** Unicode NFKC, strip URLs/emails, lowercase, "
        "keep letters/digits/apostrophes, collapse whitespace, dedupe `(business_id, text)`, "
        "min/max review length filter, categories normalized to `', '`-separated lowercase tokens."
    )

# ══════════════════════════════════════════════
# TAB 5 — ALL VISUALIZATIONS (ml_visualizations.py)
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-label">Generated HTML Visualizations (ml_visualizations.py)</p>',
                unsafe_allow_html=True)

    html_files = list_html_visuals()
    VIZ_CATALOG = {
        "metrics_gauges.html":        ("Model Performance Gauges",          560),
        "train_test_comparison.html": ("Train vs Test Comparison",           480),
        "metrics_breakdown.html":     ("Metrics Breakdown (Horizontal Bar)", 420),
        "overfitting_analysis.html":  ("Overfitting Gap Analysis",           420),
        "hyperparameters.html":       ("Hyperparameter Summary Table",       500),
        "roc_curve.html":             ("ROC Curve",                          520),
        "metrics_radar.html":         ("Performance Radar Chart",            570),
        "metric_strength_donut.html": ("Metric Strength Donut",              540),
        "eda_stars.html":             ("EDA — Star Rating Distribution",     480),
        "eda_review_length.html":     ("EDA — Review Length Distribution",   480),
    }

    if not html_files:
        st.info("No HTML visualizations found. Run `python ml_visualizations.py` to generate them.")
    else:
        for fname in html_files:
            title, height = VIZ_CATALOG.get(fname, (fname.replace("_", " ").replace(".html", "").title(), 500))
            st.markdown(f"#### {title}")
            embed_html_visual(fname, height=height)
            st.markdown("---")

# ══════════════════════════════════════════════
# TAB 6 — EXPORTS
# ══════════════════════════════════════════════
with tab6:
    st.markdown('<p class="section-label">📂 Available Outputs</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        visuals = list_html_visuals()
        file_list = "<br/>".join([f"✓ {f}" for f in visuals]) if visuals else "None yet — run ml_visualizations.py"
        st.markdown(f"""
        <div class="info-box">
        <b>📊 Visualization Files</b><br/>{file_list}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        dq_exists  = os.path.exists(os.path.join(OUTPUTS_DIR, "data_quality.json"))
        met_exists = os.path.exists(os.path.join(OUTPUTS_DIR, "model_metrics.json"))
        txt_exists = os.path.exists(os.path.join(OUTPUTS_DIR, "metrics.txt"))
        mdl_exists = os.path.isdir(os.path.join(BASE_DIR, "models", "yelp_best_model"))
        st.markdown(f"""
        <div class="info-box">
        <b>📄 Reports</b><br/>
        {'✓' if txt_exists  else '✗'} metrics.txt<br/>
        {'✓' if met_exists  else '✗'} model_metrics.json<br/>
        {'✓' if dq_exists   else '✗'} data_quality.json<br/><br/>
        <b>🤖 Models</b><br/>
        {'✓' if mdl_exists  else '✗'} models/yelp_best_model/ (Spark PipelineModel)
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="info-box">
        <b>📋 Data</b><br/>
        ✓ Train Set ({fmt(metrics.get('train_fraction', 0.8), 0) if metrics else '80'}%)<br/>
        ✓ Test Set ({fmt(1 - metrics.get('train_fraction', 0.8), 0) if metrics else '20'}%)<br/>
        ✓ Balanced Classes (undersampled)<br/><br/>
        <b>Dataset:</b> Yelp Open Dataset
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-label">📋 Pipeline Execution Guide</p>', unsafe_allow_html=True)
    st.code("""# Step 1 — clean data (prepare_data.py)
python prepare_data.py
# or fast mode for large review.json:
PREPARE_FAST_SAMPLE=1 python prepare_data.py

# Step 2 (optional) — Cassandra EDA charts (data_visualization.py)
# Requires Cassandra running with Yelp keyspace loaded
python data_visualization.py

# Step 3 — Spark training + auto-runs ml_visualizations.py (advanced_yelp_ml.py)
python advanced_yelp_ml.py

# Step 4 — regenerate visualizations only (ml_visualizations.py)
python ml_visualizations.py

# Step 5 — launch this dashboard
streamlit run dashboard_professional.py""", language="bash")

    if metrics_txt:
        st.markdown("---")
        st.markdown('<p class="section-label">📄 Plain-text Report (metrics.txt)</p>', unsafe_allow_html=True)
        st.code(metrics_txt, language="text")

    st.markdown("---")
    st.markdown('<p class="section-label">📥 Raw Metrics JSON</p>', unsafe_allow_html=True)
    if metrics:
        st.json(metrics, expanded=False)
    else:
        st.info("No `outputs/model_metrics.json` found. Run `advanced_yelp_ml.py` first.")

    st.markdown("---")
    st.markdown('<p class="section-label">📖 Model Summary</p>', unsafe_allow_html=True)
    st.markdown(f"""
**Algorithm:** {metrics.get('model_type', '—')}  
**Embedding:** {metrics.get('embedding_type', 'Word2Vec')} ({metrics.get('embedding_dimension', '—')}-dim)

**Performance (Test Set):**
- **AUC-ROC:** {fmt(metrics.get('auc', 0))}
- **Accuracy:** {fmt(metrics.get('accuracy', 0))}
- **F1-Score:** {fmt(metrics.get('f1', 0))}
- **Precision:** {fmt(metrics.get('precision', 0))}
- **Recall:** {fmt(metrics.get('recall', 0))}

**Generalization:**
- **Overfitting Status:** {get_overfitting_status(metrics)[0]}
- **AUC Gap (Train-Test):** {metrics.get('auc_diff', 0):+.4f}
- **Accuracy Gap:** {metrics.get('acc_diff', 0):+.4f}

**Configuration:**
- **Trees (RF):** {metrics.get('num_trees', 'N/A')}  | **Max Depth (RF):** {metrics.get('max_depth', 'N/A')}
- **Max Iter (LR):** {metrics.get('max_iter', 'N/A')} | **Reg / ElasticNet:** {metrics.get('reg_param', 'N/A')} / {metrics.get('elastic_net', 'N/A')}
- **Dataset Size:** {fmt(metrics.get('dataset_size', 0))}
- **Train Seed:** {metrics.get('train_test_seed', '—')}
- **Validation:** TrainValidationSplit
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8A8FA8; font-size: 0.8rem">
  🍽️ Yelp ML Analytics Dashboard &nbsp;·&nbsp; Spark ML &nbsp;·&nbsp;
  prepare_data + data_visualization + advanced_yelp_ml + ml_visualizations
</div>
""", unsafe_allow_html=True)
