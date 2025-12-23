"""
Visualization Module for Network Anomaly Detection

Generate plots for model evaluation, feature analysis, and data exploration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, List, Optional, Tuple, Any
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """
    Visualization class for network anomaly detection results.
    """
    
    def __init__(self, output_dir: str = "./visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        normalize: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        save: bool = True
    ) -> plt.Figure:
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Class names for axis labels
            title: Plot title
            normalize: Whether to normalize values
            figsize: Figure size
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            fmt = 'd'
            
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'confusion_matrix.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {filepath}")
            
        return fig
        
    def plot_roc_curves(
        self,
        roc_data: Dict[str, Dict],
        title: str = "ROC Curves",
        figsize: Tuple[int, int] = (10, 8),
        save: bool = True
    ) -> plt.Figure:
        """
        Plot ROC curves for multiple classes or models.
        
        Args:
            roc_data: Dictionary with ROC data (fpr, tpr, auc per class/model)
            title: Plot title
            figsize: Figure size
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(roc_data)))
        
        for (name, data), color in zip(roc_data.items(), colors):
            fpr = np.array(data['fpr'])
            tpr = np.array(data['tpr'])
            roc_auc = data['auc']
            
            ax.plot(
                fpr, tpr,
                color=color,
                lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})'
            )
            
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'roc_curves.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved ROC curves to {filepath}")
            
        return fig
        
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        figsize: Tuple[int, int] = (14, 5),
        save: bool = True
    ) -> plt.Figure:
        """
        Plot neural network training history.
        
        Args:
            history: Training history dictionary with loss and metrics
            title: Plot title
            figsize: Figure size
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss plot
        ax1 = axes[0]
        if 'loss' in history:
            ax1.plot(history['loss'], label='Training Loss', color='#2ecc71', lw=2)
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='Validation Loss', color='#e74c3c', lw=2)
            
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Model Loss', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2 = axes[1]
        if 'accuracy' in history:
            ax2.plot(history['accuracy'], label='Training Accuracy', color='#2ecc71', lw=2)
        if 'val_accuracy' in history:
            ax2.plot(history['val_accuracy'], label='Validation Accuracy', color='#e74c3c', lw=2)
            
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Model Accuracy', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'training_history.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved training history to {filepath}")
            
        return fig
        
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        title: str = "Feature Importance",
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8),
        save: bool = True
    ) -> plt.Figure:
        """
        Plot feature importance bar chart.
        
        Args:
            feature_names: Names of features
            importances: Importance scores
            title: Plot title
            top_n: Number of top features to show
            figsize: Figure size
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
        
        y_pos = np.arange(top_n)
        ax.barh(
            y_pos,
            importances[indices][::-1],
            color=colors[::-1],
            edgecolor='none'
        )
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'feature_importance.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved feature importance to {filepath}")
            
        return fig
        
    def plot_attack_distribution(
        self,
        labels: pd.Series,
        title: str = "Attack Type Distribution",
        figsize: Tuple[int, int] = (12, 6),
        save: bool = True
    ) -> plt.Figure:
        """
        Plot distribution of attack types.
        
        Args:
            labels: Attack labels
            title: Plot title
            figsize: Figure size
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Bar chart
        ax1 = axes[0]
        counts = labels.value_counts()
        colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
        
        bars = ax1.bar(range(len(counts)), counts.values, color=colors, edgecolor='white')
        ax1.set_xticks(range(len(counts)))
        ax1.set_xticklabels(counts.index, rotation=45, ha='right')
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Count Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts.values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f'{count:,}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        # Pie chart
        ax2 = axes[1]
        explode = [0.02] * len(counts)
        ax2.pie(
            counts.values,
            labels=counts.index,
            autopct='%1.1f%%',
            colors=colors,
            explode=explode,
            startangle=90
        )
        ax2.set_title('Percentage Distribution', fontsize=12, fontweight='bold')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'attack_distribution.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved attack distribution to {filepath}")
            
        return fig
        
    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
        title: str = "Model Comparison",
        figsize: Tuple[int, int] = (12, 6),
        save: bool = True
    ) -> plt.Figure:
        """
        Plot comparison of multiple models.
        
        Args:
            comparison_df: DataFrame with model metrics
            metrics: Metrics to compare
            title: Plot title
            figsize: Figure size
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(comparison_df))
        width = 0.2
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                offset = (i - len(metrics) / 2 + 0.5) * width
                bars = ax.bar(
                    x + offset,
                    comparison_df[metric],
                    width,
                    label=metric.replace('_', ' ').title(),
                    color=colors[i % len(colors)]
                )
                
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['model'], rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'model_comparison.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved model comparison to {filepath}")
            
        return fig
        
    def plot_precision_recall_curve(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        avg_precision: float,
        title: str = "Precision-Recall Curve",
        figsize: Tuple[int, int] = (8, 6),
        save: bool = True
    ) -> plt.Figure:
        """
        Plot precision-recall curve.
        
        Args:
            precision: Precision values
            recall: Recall values
            avg_precision: Average precision score
            title: Plot title
            figsize: Figure size
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(recall, precision, color='#3498db', lw=2,
                label=f'PR Curve (AP = {avg_precision:.3f})')
        ax.fill_between(recall, precision, alpha=0.2, color='#3498db')
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'precision_recall_curve.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved precision-recall curve to {filepath}")
            
        return fig
        
    def create_dashboard_summary(
        self,
        evaluation_results: Dict[str, Any],
        figsize: Tuple[int, int] = (16, 12),
        save: bool = True
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple plots.
        
        Args:
            evaluation_results: Complete evaluation results dictionary
            figsize: Figure size
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        
        # Define grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Metrics bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = evaluation_results.get('basic_metrics', {})
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0)
        ]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        bars = ax1.bar(metric_names, metric_values, color=colors)
        ax1.set_ylim(0, 1.1)
        ax1.set_title('Performance Metrics', fontweight='bold')
        for bar, val in zip(bars, metric_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Confusion Matrix Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        cm = np.array(evaluation_results.get('confusion_matrix_normalized', [[0]]))
        sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', ax=ax2)
        ax2.set_title('Normalized Confusion Matrix', fontweight='bold')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        # ROC Curve
        ax3 = fig.add_subplot(gs[0, 2])
        if 'roc' in evaluation_results:
            roc = evaluation_results['roc']
            if 'fpr' in roc:
                ax3.plot(roc['fpr'], roc['tpr'], color='#3498db', lw=2,
                        label=f'AUC = {roc["auc"]:.3f}')
            ax3.plot([0, 1], [0, 1], 'k--', lw=1)
            ax3.set_xlim([0, 1])
            ax3.set_ylim([0, 1.05])
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Curve', fontweight='bold')
            ax3.legend(loc='lower right')
        else:
            ax3.text(0.5, 0.5, 'ROC data not available', ha='center', va='center')
            
        # Per-class metrics
        ax4 = fig.add_subplot(gs[1, :2])
        per_class = evaluation_results.get('per_class_metrics', [])
        if per_class:
            df = pd.DataFrame(per_class)
            x = np.arange(len(df))
            width = 0.25
            ax4.bar(x - width, df['precision'], width, label='Precision', color='#3498db')
            ax4.bar(x, df['recall'], width, label='Recall', color='#2ecc71')
            ax4.bar(x + width, df['f1_score'], width, label='F1-Score', color='#e74c3c')
            ax4.set_xticks(x)
            ax4.set_xticklabels(df['class'], rotation=45, ha='right')
            ax4.set_ylim(0, 1.1)
            ax4.set_title('Per-Class Metrics', fontweight='bold')
            ax4.legend()
            
        # Summary text
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        summary_text = f"""
        Model Performance Summary
        ─────────────────────────
        
        Accuracy:     {metrics.get('accuracy', 0):.4f}
        Precision:    {metrics.get('precision', 0):.4f}
        Recall:       {metrics.get('recall', 0):.4f}
        F1-Score:     {metrics.get('f1_score', 0):.4f}
        
        MCC:          {metrics.get('mcc', 0):.4f}
        Cohen Kappa:  {metrics.get('cohen_kappa', 0):.4f}
        """
        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle('Network Anomaly Detection - Model Evaluation Dashboard',
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save:
            filepath = os.path.join(self.output_dir, 'evaluation_dashboard.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved evaluation dashboard to {filepath}")
            
        return fig


def quick_visualize(y_true, y_pred, y_prob=None, class_names=None, output_dir='./visualizations'):
    """
    Quick visualization of model results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        class_names: Class names
        output_dir: Output directory for plots
    """
    viz = Visualizer(output_dir)
    
    if class_names is None:
        class_names = ['Normal', 'Attack']
        
    viz.plot_confusion_matrix(y_true, y_pred, class_names)
    
    if y_prob is not None:
        if y_prob.ndim == 1 or y_prob.shape[1] == 2:
            prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            fpr, tpr, _ = roc_curve(y_true, prob)
            roc_auc = auc(fpr, tpr)
            viz.plot_roc_curves({'Model': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}})
            
    plt.show()
