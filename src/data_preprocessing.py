"""
Data Preprocessing Module for Network Anomaly Detection
Handles loading, cleaning, and preparing the NSL-KDD dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

from .config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURE_NAMES,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, ATTACK_CATEGORIES,
    TRAINING_CONFIG, DATASET_CONFIG
)
from .utils import setup_logging, print_banner, print_section

logger = setup_logging(__name__)


class DataPreprocessor:
    """
    Preprocessor for NSL-KDD dataset.
    Handles loading, encoding, scaling, and splitting.
    """
    
    def __init__(self, use_binary_labels: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            use_binary_labels: If True, use binary (normal/attack) labels.
                             If False, use multi-class (attack category) labels.
        """
        self.use_binary_labels = use_binary_labels
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: list = []
        self._is_fitted = False
    
    def load_dataset(self, train: bool = True) -> pd.DataFrame:
        """
        Load the NSL-KDD dataset.
        
        Args:
            train: If True, load training set. Otherwise, load test set.
            
        Returns:
            DataFrame with the loaded data
        """
        filename = DATASET_CONFIG['train_file'] if train else DATASET_CONFIG['test_file']
        filepath = RAW_DATA_DIR / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found: {filepath}\n"
                "Please run 'python data/download_data.py' first."
            )
        
        # Load without header (NSL-KDD has no header row)
        df = pd.read_csv(filepath, names=FEATURE_NAMES, header=None)
        
        logger.info(f"Loaded {'training' if train else 'test'} set: {len(df)} records")
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding.
        
        Args:
            df: Input DataFrame
            fit: If True, fit the encoders. Otherwise, use existing encoders.
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()
        
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                continue
                
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                le = self.label_encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        return df
    
    def _map_attack_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map attack labels to categories or binary.
        
        Args:
            df: Input DataFrame with 'label' column
            
        Returns:
            DataFrame with mapped labels
        """
        df = df.copy()
        
        if self.use_binary_labels:
            # Binary classification: normal (0) vs attack (1)
            df['label'] = df['label'].apply(
                lambda x: 0 if x == 'normal' else 1
            )
            df['label_name'] = df['label'].apply(
                lambda x: 'normal' if x == 0 else 'attack'
            )
        else:
            # Multi-class: map to attack categories
            df['attack_category'] = df['label'].map(
                lambda x: ATTACK_CATEGORIES.get(x, 'unknown')
            )
            
            # Encode attack categories
            if 'attack_category' not in self.label_encoders:
                self.label_encoders['attack_category'] = LabelEncoder()
                df['label'] = self.label_encoders['attack_category'].fit_transform(
                    df['attack_category']
                )
            else:
                df['label'] = self.label_encoders['attack_category'].transform(
                    df['attack_category']
                )
        
        return df
    
    def _scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X: Feature matrix
            fit: If True, fit the scaler. Otherwise, use existing scaler.
            
        Returns:
            Scaled feature matrix
        """
        if fit:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call fit_transform first.")
            return self.scaler.transform(X)
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features, labels)
        """
        print_section("Data Preprocessing (Fit)")
        
        # Store original attack labels for analysis
        original_labels = df['label'].copy()
        
        # Encode categorical features
        df = self._encode_categorical(df, fit=True)
        
        # Map attack labels
        df = self._map_attack_labels(df)
        
        # Get feature columns (exclude label and difficulty)
        feature_cols = [c for c in df.columns if c not in ['label', 'label_name', 'attack_category', 'difficulty']]
        self.feature_names = feature_cols
        
        # Extract features and labels
        X = df[feature_cols].values.astype(np.float32)
        y = df['label'].values.astype(np.int32)
        
        # Scale features
        X = self._scale_features(X, fit=True)
        
        self._is_fitted = True
        
        # Print statistics
        print(f"  Features: {X.shape[1]}")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Classes: {np.unique(y)}")
        
        if self.use_binary_labels:
            normal_count = (y == 0).sum()
            attack_count = (y == 1).sum()
            print(f"  Normal: {normal_count:,} ({100*normal_count/len(y):.1f}%)")
            print(f"  Attack: {attack_count:,} ({100*attack_count/len(y):.1f}%)")
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features, labels)
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        print_section("Data Preprocessing (Transform)")
        
        # Encode categorical features
        df = self._encode_categorical(df, fit=False)
        
        # Map attack labels
        df = self._map_attack_labels(df)
        
        # Extract features
        X = df[self.feature_names].values.astype(np.float32)
        y = df['label'].values.astype(np.int32)
        
        # Scale features
        X = self._scale_features(X, fit=False)
        
        print(f"  Features: {X.shape[1]}")
        print(f"  Samples: {X.shape[0]}")
        
        return X, y
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save the preprocessor to disk.
        
        Args:
            filepath: Path to save to. If None, uses default location.
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            filepath = PROCESSED_DATA_DIR / "preprocessor.joblib"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'use_binary_labels': self.use_binary_labels,
            '_is_fitted': self._is_fitted
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Preprocessor saved to: {filepath}")
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> 'DataPreprocessor':
        """
        Load a preprocessor from disk.
        
        Args:
            filepath: Path to load from. If None, uses default location.
            
        Returns:
            Loaded DataPreprocessor instance
        """
        if filepath is None:
            filepath = PROCESSED_DATA_DIR / "preprocessor.joblib"
        
        save_data = joblib.load(filepath)
        
        preprocessor = cls(use_binary_labels=save_data['use_binary_labels'])
        preprocessor.label_encoders = save_data['label_encoders']
        preprocessor.scaler = save_data['scaler']
        preprocessor.feature_names = save_data['feature_names']
        preprocessor._is_fitted = save_data['_is_fitted']
        
        return preprocessor


def prepare_data(use_binary_labels: bool = True, 
                 test_size: float = 0.2) -> Dict[str, Any]:
    """
    Prepare the complete dataset for training.
    
    Args:
        use_binary_labels: Use binary (normal/attack) classification
        test_size: Proportion of data to use for validation
        
    Returns:
        Dictionary with processed data and preprocessor
    """
    print_banner("Preparing NSL-KDD Dataset")
    
    preprocessor = DataPreprocessor(use_binary_labels=use_binary_labels)
    
    # Load and process training data
    train_df = preprocessor.load_dataset(train=True)
    X_train, y_train = preprocessor.fit_transform(train_df)
    
    # Load and process test data
    test_df = preprocessor.load_dataset(train=False)
    X_test, y_test = preprocessor.transform(test_df)
    
    # Create validation split from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=test_size,
        random_state=TRAINING_CONFIG['random_state'],
        stratify=y_train
    )
    
    print_section("Final Dataset Sizes")
    print(f"  Training: {X_train.shape[0]:,} samples")
    print(f"  Validation: {X_val.shape[0]:,} samples")
    print(f"  Test: {X_test.shape[0]:,} samples")
    
    # Save preprocessor
    preprocessor.save()
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': preprocessor.feature_names,
        'n_classes': len(np.unique(y_train))
    }


if __name__ == "__main__":
    # Test the preprocessor
    data = prepare_data(use_binary_labels=True)
    
    print("\n" + "="*60)
    print("  Data Preparation Complete!")
    print("="*60)
    print(f"\n  Training samples: {data['X_train'].shape}")
    print(f"  Validation samples: {data['X_val'].shape}")
    print(f"  Test samples: {data['X_test'].shape}")
    print(f"  Number of features: {len(data['feature_names'])}")
    print(f"  Number of classes: {data['n_classes']}")
