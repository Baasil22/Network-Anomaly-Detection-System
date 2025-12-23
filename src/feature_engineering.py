"""
Feature Engineering Module for Network Anomaly Detection

Advanced feature engineering including scaling, selection, and dimensionality reduction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.decomposition import PCA
from typing import Tuple, Optional, List, Dict, Any
import logging

from .config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for network traffic data.
    
    Handles scaling, selection, and dimensionality reduction.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the feature engineer."""
        self.config = config or Config()
        self.scaler = None
        self.selector = None
        self.pca = None
        self.selected_features = None
        self.feature_importances = None
        
    def fit_scaler(self, X: pd.DataFrame, method: str = 'standard') -> 'FeatureEngineer':
        """
        Fit the scaler to the training data.
        
        Args:
            X: Training features
            method: Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            self for chaining
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
            
        self.scaler.fit(X)
        logger.info(f"Fitted {method} scaler on {X.shape[1]} features")
        return self
        
    def transform_scale(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using the fitted scaler.
        
        Args:
            X: Features to scale
            
        Returns:
            Scaled features as numpy array
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        return self.scaler.transform(X)
        
    def fit_transform_scale(self, X: pd.DataFrame, method: str = 'standard') -> np.ndarray:
        """Fit scaler and transform in one step."""
        self.fit_scaler(X, method)
        return self.transform_scale(X)
        
    def select_features_mutual_info(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        k: int = 30
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top k features using mutual information.
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            
        Returns:
            Selected features DataFrame and list of selected feature names
        """
        k = min(k, X.shape[1])
        
        self.selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = self.selector.fit_transform(X, y)
        
        # Get selected feature names
        mask = self.selector.get_support()
        self.selected_features = X.columns[mask].tolist()
        
        # Store feature importances
        self.feature_importances = dict(zip(X.columns, self.selector.scores_))
        
        logger.info(f"Selected {k} features using mutual information")
        
        return pd.DataFrame(X_selected, columns=self.selected_features), self.selected_features
        
    def select_features_f_classif(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        k: int = 30
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top k features using F-classification (ANOVA F-value).
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            
        Returns:
            Selected features DataFrame and list of selected feature names
        """
        k = min(k, X.shape[1])
        
        self.selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.selector.fit_transform(X, y)
        
        mask = self.selector.get_support()
        self.selected_features = X.columns[mask].tolist()
        
        self.feature_importances = dict(zip(X.columns, self.selector.scores_))
        
        logger.info(f"Selected {k} features using F-classification")
        
        return pd.DataFrame(X_selected, columns=self.selected_features), self.selected_features
        
    def apply_pca(
        self, 
        X: np.ndarray, 
        n_components: float = 0.95,
        fit: bool = True
    ) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            X: Feature matrix (scaled)
            n_components: Number of components or variance ratio to preserve
            fit: Whether to fit the PCA (False for transform only)
            
        Returns:
            Transformed features
        """
        if fit:
            self.pca = PCA(n_components=n_components)
            X_pca = self.pca.fit_transform(X)
            logger.info(
                f"Applied PCA: {X.shape[1]} -> {X_pca.shape[1]} components "
                f"({self.pca.explained_variance_ratio_.sum():.2%} variance)"
            )
        else:
            if self.pca is None:
                raise ValueError("PCA not fitted. Call with fit=True first.")
            X_pca = self.pca.transform(X)
            
        return X_pca
        
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features to improve detection.
        
        Args:
            df: Original feature DataFrame
            
        Returns:
            DataFrame with additional derived features
        """
        df = df.copy()
        
        # Byte ratio features
        if 'src_bytes' in df.columns and 'dst_bytes' in df.columns:
            df['byte_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)
            df['total_bytes'] = df['src_bytes'] + df['dst_bytes']
            df['byte_diff'] = abs(df['src_bytes'] - df['dst_bytes'])
            
        # Error rate features
        if 'serror_rate' in df.columns and 'rerror_rate' in df.columns:
            df['total_error_rate'] = df['serror_rate'] + df['rerror_rate']
            
        # Connection rate features
        if 'count' in df.columns and 'srv_count' in df.columns:
            df['srv_ratio'] = df['srv_count'] / (df['count'] + 1)
            
        # Host-based ratio features
        if 'dst_host_count' in df.columns and 'dst_host_srv_count' in df.columns:
            df['dst_host_srv_ratio'] = df['dst_host_srv_count'] / (df['dst_host_count'] + 1)
            
        # Aggregate error features
        error_cols = [col for col in df.columns if 'error' in col.lower()]
        if error_cols:
            df['mean_error_rate'] = df[error_cols].mean(axis=1)
            
        logger.info(f"Created derived features. New shape: {df.shape}")
        
        return df
        
    def get_feature_importance_df(self) -> Optional[pd.DataFrame]:
        """
        Get feature importances as a sorted DataFrame.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_importances is None:
            return None
            
        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.feature_importances.items()
        ])
        
        return df.sort_values('importance', ascending=False).reset_index(drop=True)
        
    def get_top_features(self, n: int = 20) -> List[str]:
        """
        Get top n most important features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of top feature names
        """
        if self.feature_importances is None:
            return []
            
        sorted_features = sorted(
            self.feature_importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [f[0] for f in sorted_features[:n]]
        

def engineer_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    config: Optional[Config] = None,
    scale_method: str = 'standard',
    select_k: int = 30,
    use_pca: bool = False,
    pca_variance: float = 0.95,
    create_derived: bool = True
) -> Tuple[np.ndarray, np.ndarray, FeatureEngineer]:
    """
    Complete feature engineering pipeline.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        config: Configuration object
        scale_method: Scaling method
        select_k: Number of features to select
        use_pca: Whether to apply PCA
        pca_variance: Variance ratio to preserve with PCA
        create_derived: Whether to create derived features
        
    Returns:
        Processed training features, test features, and fitted engineer
    """
    engineer = FeatureEngineer(config)
    
    # Create derived features
    if create_derived:
        X_train = engineer.create_derived_features(X_train)
        X_test = engineer.create_derived_features(X_test)
        
    # Feature selection
    X_train_selected, selected_features = engineer.select_features_mutual_info(
        X_train, y_train, k=select_k
    )
    X_test_selected = X_test[selected_features]
    
    # Scale features
    X_train_scaled = engineer.fit_transform_scale(X_train_selected, scale_method)
    X_test_scaled = engineer.transform_scale(X_test_selected)
    
    # Optional PCA
    if use_pca:
        X_train_scaled = engineer.apply_pca(X_train_scaled, pca_variance, fit=True)
        X_test_scaled = engineer.apply_pca(X_test_scaled, fit=False)
        
    return X_train_scaled, X_test_scaled, engineer
