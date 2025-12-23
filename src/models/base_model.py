"""
Base Model Class for Network Anomaly Detection
Provides a common interface for all ML models
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from ..utils import save_model, load_model, setup_logging

logger = setup_logging(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all anomaly detection models.
    Defines the common interface and shared functionality.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize the base model.
        
        Args:
            name: Name of the model
            **kwargs: Additional model parameters
        """
        self.name = name
        self.model = None
        self.is_trained = False
        self.training_history: Dict[str, Any] = {}
        self.params = kwargs
    
    @abstractmethod
    def build(self, input_dim: int, n_classes: int) -> None:
        """
        Build the model architecture.
        
        Args:
            input_dim: Number of input features
            n_classes: Number of output classes
        """
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training history/metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Handle binary vs multi-class
        if y_proba.ndim > 1 and y_proba.shape[1] == 2:
            y_proba_positive = y_proba[:, 1]
            average = 'binary'
        else:
            y_proba_positive = y_proba
            average = 'weighted'
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average=average, zero_division=0),
            'recall': recall_score(y, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y, y_pred, average=average, zero_division=0)
        }
        
        # ROC-AUC
        try:
            if len(np.unique(y)) == 2:
                metrics['roc_auc'] = roc_auc_score(y, y_proba_positive)
            else:
                metrics['roc_auc'] = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
        except Exception:
            metrics['roc_auc'] = 0.0
        
        return metrics
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save the trained model.
        
        Args:
            filepath: Optional path to save to
            
        Returns:
            Path to saved model
        """
        metadata = {
            'name': self.name,
            'params': self.params,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        return save_model(self.model, self.name, metadata)
    
    @classmethod
    def load(cls, model_name: str, version: str = 'latest') -> 'BaseModel':
        """
        Load a saved model.
        
        Args:
            model_name: Name of the model
            version: Version to load
            
        Returns:
            Loaded model instance
        """
        model, metadata = load_model(model_name, version)
        
        instance = cls(name=metadata['name'], **metadata.get('params', {}))
        instance.model = model
        instance.training_history = metadata.get('training_history', {})
        instance.is_trained = metadata.get('is_trained', True)
        
        return instance
    
    def summary(self) -> str:
        """
        Get a summary of the model.
        
        Returns:
            Model summary string
        """
        lines = [
            f"Model: {self.name}",
            f"Trained: {self.is_trained}",
            f"Parameters: {self.params}"
        ]
        
        if self.training_history:
            lines.append(f"Training History: {list(self.training_history.keys())}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', trained={self.is_trained})"
