"""
Deep Neural Network for Network Anomaly Detection using TensorFlow/Keras
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import time
import warnings

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from .base_model import BaseModel
from ..config import NEURAL_NETWORK_PARAMS, MODELS_DIR
from ..utils import setup_logging, print_section

logger = setup_logging(__name__)


class NeuralNetworkDetector(BaseModel):
    """
    Deep Neural Network based network anomaly detector.
    Uses TensorFlow/Keras for flexible architecture and GPU acceleration.
    """
    
    def __init__(self, name: str = "neural_network", **kwargs):
        """
        Initialize Neural Network detector.
        
        Args:
            name: Model name
            **kwargs: Override default parameters
        """
        # Merge default params with any overrides
        params = {**NEURAL_NETWORK_PARAMS, **kwargs}
        super().__init__(name, **params)
        
        self.input_dim: Optional[int] = None
        self.n_classes: Optional[int] = None
        self.history = None
    
    def build(self, input_dim: int, n_classes: int) -> None:
        """
        Build the Neural Network architecture.
        
        Args:
            input_dim: Number of input features
            n_classes: Number of output classes
        """
        print_section(f"Building {self.name}")
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        hidden_layers = self.params.get('hidden_layers', [128, 64, 32, 16])
        dropout_rate = self.params.get('dropout_rate', 0.3)
        
        # Build model
        model = keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.BatchNormalization()
        ])
        
        # Add hidden layers
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001),
                name=f'dense_{i}'
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate if i < len(hidden_layers) - 1 else dropout_rate * 0.5))
        
        # Output layer
        if n_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy', keras.metrics.AUC(name='auc')]
        else:
            model.add(layers.Dense(n_classes, activation='softmax', name='output'))
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.params.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        self.model = model
        
        # Print summary
        print(f"  Input Features: {input_dim}")
        print(f"  Hidden Layers: {hidden_layers}")
        print(f"  Dropout Rate: {dropout_rate}")
        print(f"  Output Classes: {n_classes}")
        print(f"  Total Parameters: {model.count_params():,}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              class_weight: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Train the Neural Network.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            class_weight: Optional class weights for imbalanced data
            
        Returns:
            Training history
        """
        print_section(f"Training {self.name}")
        
        batch_size = self.params.get('batch_size', 256)
        epochs = self.params.get('epochs', 50)
        patience = self.params.get('early_stopping_patience', 10)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Calculate class weights if not provided
        if class_weight is None and self.n_classes == 2:
            from ..utils import calculate_class_weights
            class_weight = calculate_class_weights(y_train)
            print(f"  Class Weights: {class_weight}")
        
        print(f"  Batch Size: {batch_size}")
        print(f"  Max Epochs: {epochs}")
        print(f"  Early Stopping Patience: {patience}")
        print(f"  Training Samples: {len(X_train)}")
        
        start_time = time.time()
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Store training history
        self.training_history = {
            'training_time': training_time,
            'epochs_trained': len(self.history.history['loss']),
            'final_loss': self.history.history['loss'][-1],
            'final_accuracy': self.history.history['accuracy'][-1],
            'history': {k: [float(v) for v in vals] for k, vals in self.history.history.items()}
        }
        
        if 'val_loss' in self.history.history:
            self.training_history['final_val_loss'] = self.history.history['val_loss'][-1]
            self.training_history['final_val_accuracy'] = self.history.history['val_accuracy'][-1]
        
        print(f"\n  Training Complete!")
        print(f"  Epochs: {self.training_history['epochs_trained']}")
        print(f"  Final Accuracy: {self.training_history['final_accuracy']:.4f}")
        if 'final_val_accuracy' in self.training_history:
            print(f"  Final Val Accuracy: {self.training_history['final_val_accuracy']:.4f}")
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
        
        predictions = self.model.predict(X, verbose=0)
        
        if self.n_classes == 2:
            return (predictions.flatten() > 0.5).astype(int)
        else:
            return np.argmax(predictions, axis=1)
    
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
        
        predictions = self.model.predict(X, verbose=0)
        
        if self.n_classes == 2:
            # Return [prob_0, prob_1] for binary classification
            prob_1 = predictions.flatten()
            return np.column_stack([1 - prob_1, prob_1])
        else:
            return predictions
    
    def get_training_curves(self) -> Dict[str, List[float]]:
        """
        Get training history curves for plotting.
        
        Returns:
            Dictionary of metric histories
        """
        if self.history is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return dict(self.history.history)
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the Keras model.
        
        Args:
            filepath: Optional path to save to
            
        Returns:
            Path to saved model
        """
        if filepath is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = str(MODELS_DIR / f"{self.name}_model.keras")
        
        self.model.save(filepath)
        logger.info(f"Neural network saved to: {filepath}")
        
        return filepath
    
    @classmethod
    def load_keras_model(cls, filepath: str, name: str = "neural_network") -> 'NeuralNetworkDetector':
        """
        Load a saved Keras model.
        
        Args:
            filepath: Path to the saved model
            name: Name for the loaded model
            
        Returns:
            Loaded NeuralNetworkDetector instance
        """
        instance = cls(name=name)
        instance.model = keras.models.load_model(filepath)
        instance.is_trained = True
        
        # Infer dimensions from loaded model
        instance.input_dim = instance.model.input_shape[1]
        output_shape = instance.model.output_shape
        instance.n_classes = 2 if output_shape[-1] == 1 else output_shape[-1]
        
        return instance
