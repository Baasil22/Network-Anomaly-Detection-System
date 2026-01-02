"""
Model Predictor Service for Network Anomaly Detection
Handles model loading, preprocessing, and predictions.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyPredictor:
    """Predictor class for network anomaly detection."""
    
    FEATURE_NAMES = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]
    
    CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']
    BINARY_CLASS_NAMES = ['Normal', 'Attack']
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'models', 'saved', 'best_model.joblib'
        )
        self.model_type = 'unknown'
        self.class_names = self.BINARY_CLASS_NAMES
        
    def load_model(self, model_path: Optional[str] = None):
        """Load a trained model from disk."""
        path = model_path or self.model_path
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        logger.info(f"Loading model from {path}")
        bundle = joblib.load(path)
        
        if isinstance(bundle, dict):
            self.model = bundle.get('model')
            self.scaler = bundle.get('scaler')
            self.feature_names = bundle.get('feature_names', [])
            self.model_type = bundle.get('model_type', 'unknown')
            self.is_multiclass = bundle.get('is_multiclass', False)
            self.class_names = bundle.get('class_names', self.BINARY_CLASS_NAMES)
        else:
            self.model = bundle
            self.model_type = type(bundle).__name__
            self.is_multiclass = False
            
        logger.info(f"Loaded {self.model_type} model")
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features to match training pipeline."""
        df = df.copy()
        
        # Byte-based features
        df['byte_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)
        df['total_bytes'] = df['src_bytes'] + df['dst_bytes']
        df['log_src_bytes'] = np.log1p(df['src_bytes'])
        df['log_dst_bytes'] = np.log1p(df['dst_bytes'])
        df['byte_diff'] = df['src_bytes'] - df['dst_bytes']
        df['byte_product'] = np.log1p(df['src_bytes'] * df['dst_bytes'])
        
        # Connection rate features
        df['srv_per_host'] = df['srv_count'] / (df['count'] + 1)
        df['error_rate_sum'] = df['serror_rate'] + df['rerror_rate']
        df['srv_error_sum'] = df['srv_serror_rate'] + df['srv_rerror_rate']
        df['total_error_rate'] = df['error_rate_sum'] + df['srv_error_sum']
        
        # Host-based features
        df['host_same_srv_diff'] = df['dst_host_same_srv_rate'] - df['dst_host_diff_srv_rate']
        df['host_srv_ratio'] = df['dst_host_srv_count'] / (df['dst_host_count'] + 1)
        df['host_error_total'] = df['dst_host_serror_rate'] + df['dst_host_rerror_rate']
        df['host_srv_error_total'] = df['dst_host_srv_serror_rate'] + df['dst_host_srv_rerror_rate']
        
        # DoS detection features
        df['dos_indicator'] = (df['count'] * df['serror_rate']) + (df['srv_count'] * df['srv_serror_rate'])
        df['syn_flood_indicator'] = df['count'] * (1 - df['same_srv_rate'])
        
        # Probe detection features
        df['probe_indicator'] = df['dst_host_count'] * df['dst_host_diff_srv_rate']
        df['scan_indicator'] = (1 - df['dst_host_same_srv_rate']) * df['dst_host_count']
        
        # R2L/U2R detection features
        df['intrusion_indicator'] = df['num_failed_logins'] + df['num_compromised'] + df['root_shell'] * 10
        df['privilege_indicator'] = df['su_attempted'] * 5 + df['num_root'] + df['num_shells']
        
        # Duration-based features
        df['duration_bytes'] = np.log1p(df['duration'] * df['total_bytes'])
        df['duration_count'] = np.log1p(df['duration']) * np.log1p(df['count'])
        
        # Connection pattern features
        df['same_diff_ratio'] = df['same_srv_rate'] / (df['diff_srv_rate'] + 0.01)
        df['srv_diff_host_indicator'] = df['srv_diff_host_rate'] * df['srv_count']
        
        # Binary indicators for suspicious activity
        df['has_errors'] = ((df['serror_rate'] > 0) | (df['rerror_rate'] > 0)).astype(int)
        df['high_count'] = (df['count'] > 100).astype(int)
        df['zero_bytes'] = ((df['src_bytes'] == 0) & (df['dst_bytes'] == 0)).astype(int)
        
        return df
        
    def preprocess_input(self, data: Dict[str, Any]) -> np.ndarray:
        """Preprocess input data for prediction - matches training pipeline."""
        
        # Handle features array format
        if 'features' in data:
            features = data['features']
            if isinstance(features, list):
                # Create dict from array
                row = {}
                for i, val in enumerate(features):
                    if i < len(self.FEATURE_NAMES):
                        row[self.FEATURE_NAMES[i]] = val
                data = row
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Ensure numeric columns are numeric
        numeric_cols = [c for c in self.FEATURE_NAMES if c not in self.CATEGORICAL_COLS]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Engineer features (same as training)
        df = self.engineer_features(df)
        
        # One-hot encode categorical columns (only if they exist)
        cat_cols_present = [c for c in self.CATEGORICAL_COLS if c in df.columns]
        if cat_cols_present:
            df = pd.get_dummies(df, columns=cat_cols_present)
        
        # Align with training feature names
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[[c for c in self.feature_names if c in df.columns]]
            
            # Add any missing columns
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]
        
        features = df.values.astype(np.float64)
        
        # Apply scaler
        if self.scaler is not None:
            features = self.scaler.transform(features)
            
        return features
        
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions on input features."""
        if self.model is None:
            logger.warning("No model loaded. Using random predictions.")
            predictions = np.random.randint(0, 2, size=features.shape[0])
            confidence = np.random.random(size=features.shape[0])
            return predictions, confidence
            
        predictions = self.model.predict(features)
        
        confidence = None
        if hasattr(self.model, 'predict_proba'):
            try:
                proba = self.model.predict_proba(features)
                confidence = np.max(proba, axis=1)
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")
                
        return predictions, confidence
        
    def get_label_name(self, label: int) -> str:
        """Get the name for a prediction label."""
        if label < len(self.class_names):
            return self.class_names[label]
        return f'Unknown_{label}'


def create_predictor(model_path: Optional[str] = None) -> AnomalyPredictor:
    """Factory function to create a predictor."""
    predictor = AnomalyPredictor(model_path)
    
    try:
        predictor.load_model()
    except FileNotFoundError:
        logger.warning("No saved model found. Predictor will use random predictions.")
        
    return predictor
