"""
ML Visualizations Module
Creates comprehensive visualizations for model evaluation and insights
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

class MLVisualizer:
    def __init__(self, metrics_file="outputs/model_metrics.json"):
        """Initialize visualizer with model metrics"""
        self.metrics_file = metrics_file
        self.metrics = self._load_metrics()
        self.plots_dir = "outputs/visualizations"
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def _load_metrics(self):
        """Load metrics from JSON file"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {}
    
    def create_metrics_gauge_charts(self):
        """Create gauge charts for key metrics"""
        metrics = self.metrics
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=("AUC-ROC", "Accuracy", "Precision", "Recall")
        )
        
        # AUC Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=metrics.get("auc", 0.7),
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#F4A261"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "#E76F51"},
                        {'range': [0.5, 0.8], 'color': "#F4A261"},
                        {'range': [0.8, 1], 'color': "#2A9D8F"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.75
                    }
                },
                title={'text': "AUC-ROC"}
            ),
            row=1, col=1
        )
        
        # Accuracy Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.get("accuracy", 0.7),
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#2A9D8F"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "#E76F51"},
                        {'range': [0.5, 0.8], 'color': "#F4A261"},
                        {'range': [0.8, 1], 'color': "#2A9D8F"}
                    ]
                },
                title={'text': "Accuracy"}
            ),
            row=1, col=2
        )
        
        # Precision Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.get("precision", 0.7),
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#E76F51"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "#E76F51"},
                        {'range': [0.5, 0.8], 'color': "#F4A261"},
                        {'range': [0.8, 1], 'color': "#2A9D8F"}
                    ]
                },
                title={'text': "Precision"}
            ),
            row=2, col=1
        )
        
        # Recall Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.get("recall", 0.7),
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#264653"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "#E76F51"},
                        {'range': [0.5, 0.8], 'color': "#F4A261"},
                        {'range': [0.8, 1], 'color': "#2A9D8F"}
                    ]
                },
                title={'text': "Recall"}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Model Performance Metrics",
            font={"size": 12, "color": "#E8E3D5"},
            paper_bgcolor="#0D0F14",
            plot_bgcolor="#0D0F14",
            height=700,
            showlegend=False
        )
        
        fig.write_html(f"{self.plots_dir}/metrics_gauges.html")
        print(f"✅ Saved: metrics_gauges.html")
    
    def create_train_test_comparison(self):
        """Create train vs test comparison chart"""
        metrics = self.metrics
        
        comparison_data = {
            'Metric': ['AUC-ROC', 'Accuracy'],
            'Train': [metrics.get('train_auc', 0), metrics.get('train_accuracy', 0)],
            'Test': [metrics.get('auc', 0), metrics.get('accuracy', 0)]
        }
        df = pd.DataFrame(comparison_data)
        
        fig = go.Figure(data=[
            go.Bar(x=df['Metric'], y=df['Train'], name='Train', marker_color='#F4A261'),
            go.Bar(x=df['Metric'], y=df['Test'], name='Test', marker_color='#2A9D8F')
        ])
        
        fig.update_layout(
            title="Train vs Test Performance (Overfitting Check)",
            xaxis_title="Metric",
            yaxis_title="Score",
            barmode='group',
            font={"size": 12, "color": "#E8E3D5"},
            paper_bgcolor="#0D0F14",
            plot_bgcolor="#13161E",
            hovermode='x unified',
            template="plotly_dark",
            height=500
        )
        
        fig.write_html(f"{self.plots_dir}/train_test_comparison.html")
        print(f"✅ Saved: train_test_comparison.html")
    
    def create_metrics_breakdown(self):
        """Create detailed metrics breakdown"""
        metrics = self.metrics
        
        metrics_data = {
            'Metric': ['AUC-ROC', 'Accuracy', 'F1 Score', 'Precision', 'Recall'],
            'Score': [
                metrics.get('auc', 0),
                metrics.get('accuracy', 0),
                metrics.get('f1', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0)
            ]
        }
        df = pd.DataFrame(metrics_data)
        df = df.sort_values('Score', ascending=True)
        
        fig = px.bar(
            df,
            x='Score',
            y='Metric',
            orientation='h',
            color='Score',
            color_continuous_scale=['#E76F51', '#F4A261', '#2A9D8F'],
            title="Model Evaluation Metrics",
            labels={'Score': 'Score (0-1)', 'Metric': 'Metric'}
        )
        
        fig.update_layout(
            font={"size": 12, "color": "#E8E3D5"},
            paper_bgcolor="#0D0F14",
            plot_bgcolor="#13161E",
            height=400,
            showlegend=False,
            template="plotly_dark"
        )
        
        # Add value labels on bars
        fig.update_traces(text=df['Score'].round(4), textposition='outside')
        
        fig.write_html(f"{self.plots_dir}/metrics_breakdown.html")
        print(f"✅ Saved: metrics_breakdown.html")
    
    def create_overfitting_analysis(self):
        """Create overfitting gap visualization"""
        metrics = self.metrics
        
        auc_gap = metrics.get('auc_diff', 0)
        acc_gap = metrics.get('acc_diff', 0)
        
        data = {
            'Metric': ['AUC-ROC', 'Accuracy'],
            'Overfitting Gap': [auc_gap, acc_gap],
            'Status': ['✅ OK' if abs(auc_gap) < 0.1 else '⚠️ Warning',
                      '✅ OK' if abs(acc_gap) < 0.1 else '⚠️ Warning']
        }
        df = pd.DataFrame(data)
        
        colors = ['#2A9D8F' if abs(gap) < 0.1 else '#E76F51' for gap in df['Overfitting Gap']]
        
        fig = px.bar(df, x='Metric', y='Overfitting Gap',
                    color_discrete_sequence=colors,
                    title="Overfitting Analysis (Train - Test Gap)",
                    labels={'Overfitting Gap': 'Gap (lower is better)'})
        
        fig.add_hline(y=0.1, line_dash="dash", line_color="red", 
                     annotation_text="Warning Threshold (0.1)")
        fig.add_hline(y=-0.1, line_dash="dash", line_color="red")
        
        fig.update_layout(
            font={"size": 12, "color": "#E8E3D5"},
            paper_bgcolor="#0D0F14",
            plot_bgcolor="#13161E",
            height=400,
            showlegend=False,
            template="plotly_dark"
        )
        
        fig.write_html(f"{self.plots_dir}/overfitting_analysis.html")
        print(f"✅ Saved: overfitting_analysis.html")
    
    def create_hyperparameter_summary(self):
        """Create hyperparameter configuration display"""
        metrics = self.metrics
        
        fig = go.Figure(data=[go.Table(
            header=dict(values=['<b>Hyperparameter</b>', '<b>Value</b>'],
                       fill_color='#F4A261',
                       font=dict(color='#0D0F14', size=12),
                       align='left'),
            cells=dict(values=[
                ['Number of Trees', 'Max Depth', 'Min Instances per Node', 'Dataset Size'],
                [
                    f"{metrics.get('num_trees', 'N/A')}", 
                    f"{metrics.get('max_depth', 'N/A')}",
                    f"{metrics.get('min_instances', 'N/A')}",
                    f"{metrics.get('dataset_size', 'N/A'):,}" if isinstance(metrics.get('dataset_size', None), (int, float)) \
                    else f"{metrics.get('dataset_size', 'N/A')}"
                ]
            ],
            fill_color='#13161E',
            font=dict(color='#E8E3D5', size=11),
            align='left',
            height=30)
        )])
        
        fig.update_layout(
            title_text="Model Configuration & Best Hyperparameters",
            paper_bgcolor="#0D0F14",
            font=dict(color="#E8E3D5"),
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        fig.write_html(f"{self.plots_dir}/hyperparameters.html")
        print(f"✅ Saved: hyperparameters.html")
    
    def create_roc_simulation(self):
        """Create simulated ROC curve based on AUC"""
        import numpy as np
        
        auc = self.metrics.get('auc', 0.75)
        
        # Generate smooth ROC curve approximation
        fpr = np.linspace(0, 1, 100)
        tpr = np.where(fpr < 0.3, 2 * auc * fpr,
                      np.where(fpr < 0.7, auc + (fpr - 0.3) * 0.5,
                              1.0 - (1 - auc) * (1 - fpr)))
        tpr = np.clip(tpr, 0, 1)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC={auc:.4f})',
            line=dict(color='#F4A261', width=3),
            fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier (AUC=0.5)',
            line=dict(color='#E76F51', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f"ROC Curve (AUC = {auc:.4f})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            font={"size": 12, "color": "#E8E3D5"},
            paper_bgcolor="#0D0F14",
            plot_bgcolor="#13161E",
            hovermode='closest',
            template="plotly_dark",
            height=500
        )
        
        fig.write_html(f"{self.plots_dir}/roc_curve.html")
        print(f"✅ Saved: roc_curve.html")

    def create_metrics_radar_chart(self):
        """Create a radar chart for core model metrics"""
        metrics = self.metrics
        categories = ["AUC-ROC", "Accuracy", "Precision", "Recall", "F1 Score"]
        values = [
            metrics.get("auc", 0),
            metrics.get("accuracy", 0),
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("f1", 0)
        ]
        values += values[:1]
        categories += categories[:1]

        fig = go.Figure(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Model Metrics',
                line=dict(color='#F4A261', width=3)
            )
        )

        fig.update_layout(
            title="Model Performance Radar",
            polar=dict(
                bgcolor="#0D0F14",
                radialaxis=dict(range=[0, 1], visible=True, tickfont=dict(color="#E8E3D5")),
                angularaxis=dict(tickfont=dict(color="#E8E3D5"))
            ),
            font=dict(color="#E8E3D5", size=12),
            paper_bgcolor="#0D0F14",
            plot_bgcolor="#13161E",
            template="plotly_dark",
            height=550
        )

        fig.write_html(f"{self.plots_dir}/metrics_radar.html")
        print(f"✅ Saved: metrics_radar.html")

    def create_metric_strength_donut(self):
        """Create a donut chart showing relative metric strength"""
        metrics = self.metrics
        labels = ["AUC-ROC", "Accuracy", "Precision", "Recall", "F1 Score"]
        values = [
            metrics.get("auc", 0),
            metrics.get("accuracy", 0),
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("f1", 0)
        ]
        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=['#F4A261', '#2A9D8F', '#E76F51', '#264653', '#8AB17D']),
                textinfo='label+percent',
                hoverinfo='label+value'
            )
        )

        fig.update_layout(
            title="Metric Strength Distribution",
            font=dict(color="#E8E3D5", size=12),
            paper_bgcolor="#0D0F14",
            plot_bgcolor="#13161E",
            template="plotly_dark",
            height=520
        )

        fig.write_html(f"{self.plots_dir}/metric_strength_donut.html")
        print(f"✅ Saved: metric_strength_donut.html")

    def create_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "="*50)
        print("📊 GENERATING ML VISUALIZATIONS")
        print("="*50)
        
        try:
            self.create_metrics_gauge_charts()
            self.create_train_test_comparison()
            self.create_metrics_breakdown()
            self.create_overfitting_analysis()
            self.create_hyperparameter_summary()
            self.create_roc_simulation()
            self.create_metrics_radar_chart()
            self.create_metric_strength_donut()
            
            print("\n✅ All visualizations created successfully!")
            print(f"📁 Visualizations saved to: {self.plots_dir}/")
            return True
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            return False


if __name__ == "__main__":
    visualizer = MLVisualizer()
    visualizer.create_all_visualizations()
