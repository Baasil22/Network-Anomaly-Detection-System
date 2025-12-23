"""
Ensemble Model for Network Anomaly Detection
Combines multiple classifiers for improved accuracy
"""

import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.ensemble import (
    VotingClassifier, 
    GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import time

from .base_model import BaseModel
from ..config import ENSEMBLE_PARAMS, RANDOM_FOREST_PARAMS, SVM_PARAMS
from ..utils import setup_logging, print_section

logger = setup_logging(__name__)


class EnsembleDetector(BaseModel):
    """
    Ensemble model combining multiple classifiers.
    Supports voting and stacking strategies.
    """
    
    def __init__(self, name: str = "ensemble", method: str = "voting", **kwargs):
        """
        Initialize Ensemble detector.
        
        Args:
            name: Model name
            method: 'voting', 'stacking', or 'gradient_boosting'
            **kwargs: Override default parameters
        """
        params = {**ENSEMBLE_PARAMS, **kwargs}
        params['method'] = method
        super().__init__(name, **params)
        
        self.method = method
        self.estimators: List = []
    
    def build(self, input_dim: int, n_classes: int) -> None:
        """
        Build the ensemble model.
        
        Args:
            input_dim: Number of input features
            n_classes: Number of output classes
        """
        print_section(f"Building {self.name} ({self.method})")
        
        if self.method == "voting":
            self._build_voting_classifier()
        elif self.method == "stacking":
            self._build_stacking_classifier()
        elif self.method == "gradient_boosting":
            self._build_gradient_boosting()
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        print(f"  Method: {self.method}")
        print(f"  Input Features: {input_dim}")
        print(f"  Output Classes: {n_classes}")
    
    def _build_voting_classifier(self) -> None:
        """Build a voting classifier ensemble."""
        
        # Define base estimators
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        svm = SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        self.estimators = [
            ('rf', rf),
            ('svm', svm),
            ('gb', gb)
        ]
        
        voting = self.params.get('voting', 'soft')
        weights = self.params.get('weights', [2, 1, 2])
        
        self.model = VotingClassifier(
            estimators=self.estimators,
            voting=voting,
            weights=weights,
            n_jobs=-1
        )
        
        print(f"  Estimators: {[name for name, _ in self.estimators]}")
        print(f"  Voting: {voting}")
        print(f"  Weights: {weights}")
    
    def _build_stacking_classifier(self) -> None:
        """Build a stacking classifier ensemble."""
        
        # Define base estimators
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        svm = SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=42
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
        
        self.estimators = [
            ('rf', rf),
            ('svm', svm),
            ('gb', gb)
        ]
        
        # Meta-learner
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        self.model = StackingClassifier(
            estimators=self.estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
        
        print(f"  Base Estimators: {[name for name, _ in self.estimators]}")
        print(f"  Meta-Learner: LogisticRegression")
    
    def _build_gradient_boosting(self) -> None:
        """Build a Gradient Boosting classifier."""
        
        self.model = GradientBoostingClassifier(
            n_estimators=self.params.get('n_estimators', 200),
            max_depth=self.params.get('max_depth', 5),
            learning_rate=self.params.get('learning_rate', 0.1),
            subsample=self.params.get('subsample', 0.8),
            random_state=42
        )
        
        print(f"  Estimators: {self.params.get('n_estimators', 200)}")
        print(f"  Max Depth: {self.params.get('max_depth', 5)}")
        print(f"  Learning Rate: {self.params.get('learning_rate', 0.1)}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            sample_size: Optional limit on training samples (for SVM)
            
        Returns:
            Training history
        """
        print_section(f"Training {self.name}")
        
        # Sample for faster training if needed
        if sample_size and len(X_train) > sample_size:
            print(f"  Sampling {sample_size} from {len(X_train)} samples...")
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_train_fit = X_train[indices]
            y_train_fit = y_train[indices]
        else:
            X_train_fit = X_train
            y_train_fit = y_train
        
        start_time = time.time()
        
        print(f"  Fitting ensemble on {len(X_train_fit)} samples...")
        self.model.fit(X_train_fit, y_train_fit)
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        train_score = self.model.score(X_train_fit, y_train_fit)
        
        self.training_history = {
            'training_time': training_time,
            'train_accuracy': train_score,
            'n_samples': len(X_train_fit),
            'method': self.method
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
        
        # Gradient Boosting doesn't have predict_proba by default for some versions
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fallback: create one-hot probabilities from predictions
            predictions = self.model.predict(X)
            proba = np.zeros((len(predictions), 2))
            proba[np.arange(len(predictions)), predictions] = 1.0
            return proba
    
    def get_estimator_scores(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Get individual scores for each estimator in the ensemble.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dictionary of estimator names and scores
        """
        if self.method == "gradient_boosting":
            return {"gradient_boosting": self.model.score(X, y)}
        
        scores = {}
        for name, estimator in self.estimators:
            if hasattr(estimator, 'score'):
                scores[name] = estimator.score(X, y)
        
        return scores
