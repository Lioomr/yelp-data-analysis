"""
Shared utilities for Yelp ML Pipeline
"""

import json
import os
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsManager:
    """Manage model metrics and configurations"""
    
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    METRICS_FILE = os.path.join(ROOT_DIR, "outputs", "model_metrics.json")
    METRICS_TXT = os.path.join(ROOT_DIR, "outputs", "metrics.txt")
    DATA_QUALITY_FILE = os.path.join(ROOT_DIR, "outputs", "data_quality.json")
    EDA_SUMMARY_FILE = os.path.join(ROOT_DIR, "outputs", "eda_summary.json")
    
    @staticmethod
    def load_metrics():
        """Load metrics from JSON file"""
        if os.path.exists(MetricsManager.METRICS_FILE):
            try:
                with open(MetricsManager.METRICS_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metrics: {e}")
                return {}
        return {}
    
    @staticmethod
    def default_metrics():
        """Default metrics when no file exists"""
        return {
            "auc": 0.75,
            "accuracy": 0.75,
            "f1": 0.75,
            "precision": 0.75,
            "recall": 0.75,
            "train_auc": 0.78,
            "train_accuracy": 0.78,
            "auc_diff": 0.03,
            "acc_diff": 0.03,
            "dataset_size": 10465,
            "model_type": "Logistic Regression",
            "num_trees": None,
            "max_depth": None,
            "max_iter": 30,
            "reg_param": 0.1,
            "min_instances": 1,
        }

    @staticmethod
    def load_data_quality():
        """Load prepare_data outputs for dashboard EDA."""
        if os.path.exists(MetricsManager.DATA_QUALITY_FILE):
            try:
                with open(MetricsManager.DATA_QUALITY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load data quality: {e}")
        return {}

    @staticmethod
    def load_eda_summary():
        if os.path.exists(MetricsManager.EDA_SUMMARY_FILE):
            try:
                with open(MetricsManager.EDA_SUMMARY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load eda summary: {e}")
        return {}
    
    @staticmethod
    def save_metrics(metrics):
        """Save metrics to JSON file"""
        os.makedirs(os.path.dirname(MetricsManager.METRICS_FILE), exist_ok=True)
        with open(MetricsManager.METRICS_FILE, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {MetricsManager.METRICS_FILE}")
    
    @staticmethod
    def get_overfitting_status(metrics):
        """Determine overfitting status"""
        auc_diff = abs(metrics.get('auc_diff', 0))
        acc_diff = abs(metrics.get('acc_diff', 0))
        
        if auc_diff > 0.15 or acc_diff > 0.15:
            return "❌ Warning: Possible Overfitting", "red"
        elif auc_diff > 0.08 or acc_diff > 0.08:
            return "⚠️ Monitor Overfitting", "orange"
        else:
            return "✅ Good Generalization", "green"
    
    @staticmethod
    def get_performance_rating(auc):
        """Get rating based on AUC score"""
        if auc >= 0.95:
            return "🏆 Excellent", "#2A9D8F"
        elif auc >= 0.90:
            return "⭐ Very Good", "#F4A261"
        elif auc >= 0.80:
            return "👍 Good", "#FFB703"
        elif auc >= 0.70:
            return "📊 Fair", "#FB5607"
        else:
            return "❌ Poor", "#E63946"


class DataUtils:
    """Utilities for data handling"""
    
    @staticmethod
    def format_number(value, decimals=4):
        """Format number with specified decimals"""
        if isinstance(value, float):
            return f"{value:.{decimals}f}"
        return str(value)
    
    @staticmethod
    def format_percentage(value, decimals=2):
        """Format as percentage"""
        return f"{value*100:.{decimals}f}%"
    
    @staticmethod
    def get_confusion_matrix_stats(tn, fp, fn, tp):
        """Calculate statistics from confusion matrix"""
        total = tn + fp + fn + tp
        
        return {
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,  # Recall
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,  # Precision
            "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
            "accuracy": (tp + tn) / total if total > 0 else 0,
            "f1": 2 * (tp / (tp + fp) * tp / (tp + fn)) / (tp / (tp + fp) + tp / (tp + fn)) if (tp + fp) > 0 and (tp + fn) > 0 else 0,
            "total": total
        }


class FileManager:
    """Manage file paths and creation"""
    
    DIRS = {
        "outputs": "outputs",
        "visualizations": "outputs/visualizations",
        "models": "models",
        "data": "data"
    }
    
    @staticmethod
    def ensure_dirs():
        """Create all required directories"""
        for dir_name, dir_path in FileManager.DIRS.items():
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Ensured directory: {dir_path}")
    
    @staticmethod
    def get_viz_path(filename):
        """Get visualization file path"""
        return os.path.join(FileManager.DIRS["visualizations"], filename)
    
    @staticmethod
    def get_model_path(filename):
        """Get model file path"""
        return os.path.join(FileManager.DIRS["models"], filename)


class ReportGenerator:
    """Generate text reports"""
    
    @staticmethod
    def create_metrics_report(metrics):
        """Create formatted metrics report"""
        report = "="*60 + "\n"
        report += "YELP ML MODEL - PERFORMANCE METRICS\n"
        report += "="*60 + "\n\n"
        
        report += "TEST SET METRICS:\n"
        report += f"  AUC-ROC:   {metrics.get('auc', 0):.4f}\n"
        report += f"  Accuracy:  {metrics.get('accuracy', 0):.4f}\n"
        report += f"  F1 Score:  {metrics.get('f1', 0):.4f}\n"
        report += f"  Precision: {metrics.get('precision', 0):.4f}\n"
        report += f"  Recall:    {metrics.get('recall', 0):.4f}\n\n"
        
        report += "TRAINING SET METRICS:\n"
        report += f"  AUC-ROC:   {metrics.get('train_auc', 0):.4f}\n"
        report += f"  Accuracy:  {metrics.get('train_accuracy', 0):.4f}\n\n"
        
        report += "OVERFITTING ANALYSIS:\n"
        report += f"  AUC Difference:      {metrics.get('auc_diff', 0):+.4f}\n"
        report += f"  Accuracy Difference: {metrics.get('acc_diff', 0):+.4f}\n\n"
        
        report += "MODEL CONFIGURATION:\n"
        report += f"  Algorithm:       {metrics.get('model_type', 'N/A')}\n"
        nt = metrics.get("num_trees")
        report += f"  Number of Trees: {nt if nt is not None else 'N/A (not RF)'}\n"
        md = metrics.get("max_depth")
        report += f"  Max Depth:       {md if md is not None else 'N/A (not RF)'}\n"
        mi = metrics.get("max_iter")
        report += f"  Max Iter (LR):   {mi if mi is not None else 'N/A'}\n"
        rp = metrics.get("reg_param")
        report += f"  Reg Param:       {rp if rp is not None else 'N/A'}\n"
        report += f"  Min Instances:   {metrics.get('min_instances', 1)}\n"
        report += f"  Dataset Size:    {metrics.get('dataset_size', 0):,}\n\n"
        
        report += "="*60 + "\n"
        
        return report
    
    @staticmethod
    def save_report(metrics, filename="outputs/metrics.txt"):
        """Save metrics report to file"""
        report = ReportGenerator.create_metrics_report(metrics)
        with open(filename, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {filename}")


if __name__ == "__main__":
    # Test utilities
    FileManager.ensure_dirs()
    metrics = MetricsManager.load_metrics()
    print(MetricsManager.get_overfitting_status(metrics))
    print(MetricsManager.get_performance_rating(metrics.get('auc', 0.7)))
