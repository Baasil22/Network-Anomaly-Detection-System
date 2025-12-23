"""
Random Forest Classifier for Network Anomaly Detection
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import time

from .base_model import BaseModel
from ..config import RANDOM_FOREST_PARAMS, TRAINING_CONFIG
from ..utils import setup_logging, print_section

logger = setup_logging(__name__)


class RandomForestAnomalyDetector(BaseModel):
    """
    Random Forest based network anomaly detector.
    Excellent for handling mixed feature types and providing feature importance.
    """
    
    def __init__(self, name: str = "random_forest", **kwargs):
        """
        Initialize Random Forest detector.
        
        Args:
            name: Model name
            **kwargs: Override default parameters
        """
        # Merge default params with any overrides
        params = {**RANDOM_FOREST_PARAMS, **kwargs}
        super().__init__(name, **params)
        
        self.feature_importances_: Optional[np.ndarray] = None
    
    def build(self, input_dim: int, n_classes: int) -> None:
        """
        Build the Random Forest model.
        
        Args:
            input_dim: Number of input features
            n_classes: Number of output classes
        """
        print_section(f"Building {self.name}")
        
        self.model = RandomForestClassifier(
            n_estimators=self.params.get('n_estimators', 100),
            max_depth=self.params.get('max_depth', 20),
            min_samples_split=self.params.get('min_samples_split', 5),
            min_samples_leaf=self.params.get('min_samples_leaf', 2),
            random_state=self.params.get('random_state', 42),
            n_jobs=self.params.get('n_jobs', -1),
            class_weight='balanced'  # Handle imbalanced classes
        )
        
        print(f"  Estimators: {self.params.get('n_estimators', 100)}")
        print(f"  Max Depth: {self.params.get('max_depth', 20)}")
        print(f"  Input Features: {input_dim}")
        print(f"  Output Classes: {n_classes}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training history with metrics
        """
        print_section(f"Training {self.name}")
        
        start_time = time.time()
        
        # Train the model
        print("  Fitting Random Forest...")
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Store feature importances
        self.feature_importances_ = self.model.feature_importances_
        
        # Calculate metrics
        train_score = self.model.score(X_train, y_train)
        
        self.training_history = {
            'training_time': training_time,
            'train_accuracy': train_score,
            'n_samples': len(X_train)
        }
        
        print(f"  Training Accuracy: {train_score:.4f}")
        
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
    
    def get_feature_importance(self, feature_names: Optional[list] = None,
                                top_n: int = 20) -> Dict[str, float]:
        """
        Get top feature importances.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature names and importances
        """
        if self.feature_importances_ is None:
            raise ValueError("Model not trained. Call train() first.")
        
        indices = np.argsort(self.feature_importances_)[::-1][:top_n]
        
        if feature_names:
            return {feature_names[i]: self.feature_importances_[i] for i in indices}
        else:
            return {f"feature_{i}": self.feature_importances_[i] for i in indices}
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        print_section(f"Cross-Validation ({cv} folds)")
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        results = {
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_scores': scores.tolist()
        }
        
        print(f"  Mean Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        
        return results
    
    def hyperparameter_search(self, X: np.ndarray, y: np.ndarray,
                               param_grid: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter search using GridSearchCV.
        
        Args:
            X: Features
            y: Labels
            param_grid: Parameter grid to search
            
        Returns:
            Best parameters and scores
        """
        print_section("Hyperparameter Search")
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        print("  Searching...")
        grid_search.fit(X, y)
        
        print(f"  Best Accuracy: {grid_search.best_score_:.4f}")
        print(f"  Best Parameters: {grid_search.best_params_}")
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.params.update(grid_search.best_params_)
        
        return {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_
        }
