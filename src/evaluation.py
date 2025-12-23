"""
Model Evaluation Module for Network Anomaly Detection

Comprehensive evaluation metrics, confusion matrices, and ROC analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation for intrusion detection.
    
    Provides multiple metrics, confusion matrix analysis, and ROC curves.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize the evaluator.
        
        Args:
            class_names: Names of the classes for display
        """
        self.class_names = class_names
        self.evaluation_results = {}
        
    def calculate_basic_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method for multi-class
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        return metrics
        
    def calculate_per_class_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """
        Calculate metrics for each class separately.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            DataFrame with per-class metrics
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        
        results = []
        for cls in classes:
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)
            
            result = {
                'class': self.class_names[cls] if self.class_names and cls < len(self.class_names) else str(cls),
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                'support': int(np.sum(y_true == cls))
            }
            results.append(result)
            
        return pd.DataFrame(results)
        
    def get_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        normalize: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization mode ('true', 'pred', 'all', None)
            
        Returns:
            Confusion matrix as numpy array
        """
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        return cm
        
    def get_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        output_dict: bool = False
    ) -> Union[str, Dict]:
        """
        Generate sklearn classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_dict: Whether to return as dictionary
            
        Returns:
            Classification report string or dictionary
        """
        return classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names,
            output_dict=output_dict,
            zero_division=0
        )
        
    def calculate_roc_data(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        n_classes: int = 2
    ) -> Dict[str, Any]:
        """
        Calculate ROC curve data for all classes.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities (n_samples, n_classes)
            n_classes: Number of classes
            
        Returns:
            Dictionary with FPR, TPR, and AUC for each class
        """
        # Binarize the true labels
        if n_classes == 2:
            y_true_bin = y_true
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]
                
            fpr, tpr, thresholds = roc_curve(y_true_bin, y_prob)
            roc_auc = auc(fpr, tpr)
            
            return {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': roc_auc
            }
        else:
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            roc_data = {}
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                
                class_name = self.class_names[i] if self.class_names else f"class_{i}"
                roc_data[class_name] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'auc': roc_auc
                }
                
            # Compute micro-average ROC
            fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
            roc_data['micro_avg'] = {
                'fpr': fpr_micro,
                'tpr': tpr_micro,
                'auc': auc(fpr_micro, tpr_micro)
            }
            
            return roc_data
            
    def calculate_precision_recall_data(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate precision-recall curve data.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary with precision, recall, and thresholds
        """
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
            
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'average_precision': avg_precision
        }
        
    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Complete model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            model_name: Name of the model for logging
            
        Returns:
            Dictionary with all evaluation results
        """
        logger.info(f"Evaluating {model_name}...")
        
        results = {
            'model_name': model_name,
            'basic_metrics': self.calculate_basic_metrics(y_true, y_pred),
            'per_class_metrics': self.calculate_per_class_metrics(y_true, y_pred).to_dict('records'),
            'confusion_matrix': self.get_confusion_matrix(y_true, y_pred).tolist(),
            'confusion_matrix_normalized': self.get_confusion_matrix(y_true, y_pred, normalize='true').tolist(),
            'classification_report': self.get_classification_report(y_true, y_pred, output_dict=True)
        }
        
        # Add ROC data if probabilities available
        if y_prob is not None:
            n_classes = y_prob.shape[1] if y_prob.ndim == 2 else 2
            roc_data = self.calculate_roc_data(y_true, y_prob, n_classes)
            
            # Convert numpy arrays to lists for JSON serialization
            if n_classes == 2:
                results['roc'] = {
                    'fpr': roc_data['fpr'].tolist(),
                    'tpr': roc_data['tpr'].tolist(),
                    'auc': roc_data['auc']
                }
            else:
                results['roc'] = {}
                for key, data in roc_data.items():
                    results['roc'][key] = {
                        'fpr': data['fpr'].tolist(),
                        'tpr': data['tpr'].tolist(),
                        'auc': data['auc']
                    }
                    
            # Precision-Recall curve for binary classification
            if n_classes == 2:
                pr_data = self.calculate_precision_recall_data(y_true, y_prob)
                results['precision_recall'] = {
                    'precision': pr_data['precision'].tolist(),
                    'recall': pr_data['recall'].tolist(),
                    'average_precision': pr_data['average_precision']
                }
                
        self.evaluation_results[model_name] = results
        
        # Log summary
        metrics = results['basic_metrics']
        logger.info(
            f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
            f"F1: {metrics['f1_score']:.4f}, "
            f"MCC: {metrics['mcc']:.4f}"
        )
        
        return results
        
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all evaluated models.
        
        Returns:
            DataFrame comparing model metrics
        """
        if not self.evaluation_results:
            return pd.DataFrame()
            
        comparison = []
        for model_name, results in self.evaluation_results.items():
            row = {'model': model_name}
            row.update(results['basic_metrics'])
            
            # Add AUC if available
            if 'roc' in results:
                if 'auc' in results['roc']:
                    row['auc_roc'] = results['roc']['auc']
                elif 'micro_avg' in results['roc']:
                    row['auc_roc'] = results['roc']['micro_avg']['auc']
                    
            comparison.append(row)
            
        df = pd.DataFrame(comparison)
        return df.sort_values('f1_score', ascending=False)
        
    def get_best_model(self, metric: str = 'f1_score') -> str:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Name of the best model
        """
        comparison = self.compare_models()
        if comparison.empty:
            return ""
            
        best_idx = comparison[metric].idxmax()
        return comparison.loc[best_idx, 'model']
        
    def save_results(self, filepath: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            filepath: Path to save results
        """
        with open(filepath, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        logger.info(f"Saved evaluation results to {filepath}")
        
    def load_results(self, filepath: str):
        """
        Load evaluation results from JSON file.
        
        Args:
            filepath: Path to load results from
        """
        with open(filepath, 'r') as f:
            self.evaluation_results = json.load(f)
        logger.info(f"Loaded evaluation results from {filepath}")
        
        
def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Quick evaluation for binary classifiers.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = ModelEvaluator(class_names=['Normal', 'Attack'])
    return evaluator.evaluate_model(y_true, y_pred, y_prob, 'binary_classifier')
    
    
def evaluate_multiclass_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Quick evaluation for multi-class classifiers.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        class_names: Names of classes
        
    Returns:
        Dictionary of evaluation metrics
    """
    if class_names is None:
        class_names = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
        
    evaluator = ModelEvaluator(class_names=class_names)
    return evaluator.evaluate_model(y_true, y_pred, y_prob, 'multiclass_classifier')
