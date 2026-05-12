"""
Yelp ML Pipeline Orchestrator
Runs the complete pipeline: data preparation, model training, and visualization generation
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from utils import FileManager, MetricsManager, ReportGenerator, logger


def run_prepare_data():
    """Run prepare_data.py in the project root (rename, clean JSONL, build review_small)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prep = os.path.join(script_dir, "prepare_data.py")
    if not os.path.isfile(prep):
        print("⚠️  prepare_data.py not found; skipping data preparation.")
        return True
    print("\n⏳ Data preparation (prepare_data.py)")
    try:
        r = subprocess.run(
            [sys.executable, prep],
            cwd=script_dir,
            timeout=7200,
        )
        if r.returncode != 0:
            print(f"❌ prepare_data.py failed with code {r.returncode}")
            return False
        print("✅ Data preparation completed")
        return True
    except Exception as e:
        print(f"❌ prepare_data.py error: {e}")
        return False

class PipelineOrchestrator:
    """Orchestrate the entire ML pipeline"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.results = {}
    
    def log_section(self, title):
        """Print formatted section header"""
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70)
    
    def run_step(self, step_name, script_path, description=""):
        """Run a pipeline step"""
        print(f"\n⏳ {step_name}")
        if description:
            print(f"   {description}")
        
        try:
            if script_path.endswith('.py'):
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=False,
                    timeout=3600  # 1 hour timeout
                )
                if result.returncode != 0:
                    print(f"❌ {step_name} failed with code {result.returncode}")
                    return False
            print(f"✅ {step_name} completed")
            return True
        except subprocess.TimeoutExpired:
            print(f"❌ {step_name} timed out")
            return False
        except Exception as e:
            print(f"❌ {step_name} error: {e}")
            return False
    
    def setup(self):
        """Setup pipeline environment"""
        self.log_section("🔧 PIPELINE SETUP")
        FileManager.ensure_dirs()
        print("✅ Directories created")
        print("✅ Environment ready")
    
    def train_model(self):
        """Train the ML model"""
        self.log_section("🤖 MODEL TRAINING")
        
        success = self.run_step(
            "Training Advanced ML Model",
            "advanced_yelp_ml.py",
            "PySpark: Word2Vec + TF-IDF + categories, LR vs RF with TrainValidationSplit",
        )
        
        if success:
            metrics = MetricsManager.load_metrics()
            self.results['metrics'] = metrics
            return True
        return False
    
    def generate_visualizations(self):
        """Generate visualization charts"""
        self.log_section("📊 VISUALIZATION GENERATION")
        
        if not self.results.get('metrics'):
            metrics = MetricsManager.load_metrics()
            self.results['metrics'] = metrics
        
        success = self.run_step(
            "Creating ML Visualizations",
            "ml_visualizations.py",
            "Generating interactive Plotly charts and dashboards"
        )
        
        return success
    
    def generate_report(self):
        """Generate text reports"""
        self.log_section("📄 REPORT GENERATION")
        
        metrics = self.results.get('metrics') or MetricsManager.load_metrics()
        
        try:
            ReportGenerator.save_report(metrics)
            print("✅ Metrics report generated")
            
            # Print report to console
            report = ReportGenerator.create_metrics_report(metrics)
            print(report)
            
            return True
        except Exception as e:
            print(f"❌ Report generation failed: {e}")
            return False
    
    def print_summary(self):
        """Print pipeline summary"""
        self.log_section("📈 PIPELINE SUMMARY")
        
        metrics = self.results.get('metrics') or MetricsManager.load_metrics()
        
        print("\n🎯 KEY METRICS:")
        print(f"  AUC-ROC:       {metrics.get('auc', 0):.4f}")
        print(f"  Accuracy:      {metrics.get('accuracy', 0):.4f}")
        print(f"  F1 Score:      {metrics.get('f1', 0):.4f}")
        print(f"  Precision:     {metrics.get('precision', 0):.4f}")
        print(f"  Recall:        {metrics.get('recall', 0):.4f}")
        
        print("\n📊 OVERFITTING ANALYSIS:")
        status, _ = MetricsManager.get_overfitting_status(metrics)
        print(f"  {status}")
        print(f"  AUC Gap:       {metrics.get('auc_diff', 0):+.4f}")
        print(f"  Accuracy Gap:  {metrics.get('acc_diff', 0):+.4f}")
        
        rating, _ = MetricsManager.get_performance_rating(metrics.get('auc', 0.7))
        print(f"\n⭐ MODEL RATING: {rating}")
        
        print("\n📁 OUTPUT FILES:")
        print("  ✓ outputs/model_metrics.json      - Metrics data")
        print("  ✓ outputs/metrics.txt             - Formatted report")
        print("  ✓ outputs/visualizations/         - Interactive charts")
        print("    - metrics_gauges.html")
        print("    - train_test_comparison.html")
        print("    - metrics_breakdown.html")
        print("    - overfitting_analysis.html")
        print("    - hyperparameters.html")
        print("    - roc_curve.html")
        print("  ✓ models/yelp_best_model/          - Trained Spark ML pipeline")
        
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            print(f"\n⏱️  TOTAL EXECUTION TIME: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    def run_full_pipeline(self):
        """Execute the complete pipeline"""
        self.start_time = time.time()
        
        print("\n")
        print("╔" + "="*68 + "╗")
        print("║" + " "*15 + "🍽️  YELP ML PIPELINE - COMPLETE WORKFLOW" + " "*13 + "║")
        print("╚" + "="*68 + "╝")
        print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run pipeline steps
        steps = [
            ("Setup", lambda: self.setup()),
            ("Prepare Data", lambda: run_prepare_data()),
            ("Train Model", lambda: self.train_model()),
            ("Generate Visualizations", lambda: self.generate_visualizations()),
            ("Generate Reports", lambda: self.generate_report()),
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            if not step_func():
                failed_steps.append(step_name)
                print(f"\n⚠️  Continuing despite {step_name} issues...")
        
        self.end_time = time.time()
        
        # Print final summary
        self.print_summary()
        
        print("\n" + "="*70)
        if failed_steps:
            print(f"⚠️  COMPLETED WITH ISSUES: {', '.join(failed_steps)}")
        else:
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
        return len(failed_steps) == 0


def main():
    """Main entry point"""
    orchestrator = PipelineOrchestrator()
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    success = orchestrator.run_full_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
