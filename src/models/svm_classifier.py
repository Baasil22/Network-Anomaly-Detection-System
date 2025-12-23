"""
Support Vector Machine Classifier for Network Anomaly Detection
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import time

from .base_model import BaseModel
from ..config import SVM_PARAMS
from ..utils import setup_logging, print_section

logger = setup_logging(__name__)


class SVMClassifier(BaseModel):
    """
    Support Vector Machine based network anomaly detector.
    Effective for binary classification with clear margins.
    """
    
    def __init__(self, name: str = "svm", **kwargs):
        """
        Initialize SVM classifier.
        
        Args:
            name: Model name
            **kwargs: Override default parameters
        """
        # Merge default params with any overrides
        params = {**SVM_PARAMS, **kwargs}
        super().__init__(name, **params)
    
    def build(self, input_dim: int, n_classes: int) -> None:
        """
        Build the SVM model.
        
        Args:
            input_dim: Number of input features
            n_classes: Number of output classes
        """
        print_section(f"Building {self.name}")
        
        self.model = SVC(
            C=self.params.get('C', 1.0),
            kernel=self.params.get('kernel', 'rbf'),
            gamma=self.params.get('gamma', 'scale'),
            random_state=self.params.get('random_state', 42),
            probability=True,  # Enable probability estimates
            class_weight='balanced'  # Handle imbalanced classes
        )
        
        print(f"  Kernel: {self.params.get('kernel', 'rbf')}")
        print(f"  C: {self.params.get('C', 1.0)}")
        print(f"  Gamma: {self.params.get('gamma', 'scale')}")
        print(f"  Input Features: {input_dim}")
        print(f"  Output Classes: {n_classes}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the SVM model.
        
        Note: SVM can be slow on large datasets. 
        Use sample_size to limit training data if needed.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            sample_size: Optional limit on training samples
            
        Returns:
            Training history with metrics
        """
        print_section(f"Training {self.name}")
        
        # Sample if dataset is large
        if sample_size and len(X_train) > sample_size:
            print(f"  Sampling {sample_size} from {len(X_train)} samples...")
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]
        
        start_time = time.time()
        
        # Train the model
        print(f"  Fitting SVM on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        train_score = self.model.score(X_train, y_train)
        
        self.training_history = {
            'training_time': training_time,
            'train_accuracy': train_score,
            'n_samples': len(X_train),
            'n_support_vectors': self.model.n_support_.sum()
        }
        
        print(f"  Training Accuracy: {train_score:.4f}")
        print(f"  Support Vectors: {self.model.n_support_.sum()}")
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            self.training_history['val_accuracy'] = val_score
            print(f"  Validation Accuracy: {val_score:.4f}")
        
        print(f"  Training Time: {training_time:.2f}s")
        
        self.is_trained = True
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def hyperparameter_search(self, X: np.ndarray, y: np.ndarray,
                               param_grid: Optional[Dict] = None,
                               sample_size: int = 10000) -> Dict[str, Any]:
        """
        Perform hyperparameter search using GridSearchCV.
        
        Args:
            X: Features
            y: Labels
            param_grid: Parameter grid to search
            sample_size: Number of samples to use
            
        Returns:
            Best parameters and scores
        """
        print_section("Hyperparameter Search (SVM)")
        
        # Sample for faster search
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X = X[indices]
            y = y[indices]
        
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        
        grid_search = GridSearchCV(
            SVC(random_state=42, probability=True),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        print(f"  Searching on {len(X)} samples...")
        grid_search.fit(X, y)
        
        print(f"  Best Accuracy: {grid_search.best_score_:.4f}")
        print(f"  Best Parameters: {grid_search.best_params_}")
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.params.update(grid_search.best_params_)
        
        return {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_
        }
