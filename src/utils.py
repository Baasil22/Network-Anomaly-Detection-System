"""
Utility Functions for Network Anomaly Detection
"""

import logging
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config import MODELS_DIR, LOGGING_CONFIG


def setup_logging(name: str = __name__) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format']
    )
    return logging.getLogger(name)


def save_model(model: Any, model_name: str, metadata: Optional[Dict] = None) -> Path:
    """
    Save a trained model to disk.
    
    Args:
        model: The trained model object
        model_name: Name for the saved model
        metadata: Optional metadata to save with the model
        
    Returns:
        Path to the saved model
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.joblib"
    filepath = MODELS_DIR / filename
    
    # Create save package
    save_data = {
        'model': model,
        'metadata': metadata or {},
        'timestamp': timestamp,
        'model_name': model_name
    }
    
    joblib.dump(save_data, filepath)
    
    # Also save as latest
    latest_path = MODELS_DIR / f"{model_name}_latest.joblib"
    joblib.dump(save_data, latest_path)
    
    return filepath


def load_model(model_name: str, version: str = 'latest') -> Tuple[Any, Dict]:
    """
    Load a trained model from disk.
    
    Args:
        model_name: Name of the model to load
        version: Version to load ('latest' or timestamp)
        
    Returns:
        Tuple of (model, metadata)
    """
    if version == 'latest':
        filepath = MODELS_DIR / f"{model_name}_latest.joblib"
    else:
        filepath = MODELS_DIR / f"{model_name}_{version}.joblib"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")
    
    save_data = joblib.load(filepath)
    return save_data['model'], save_data['metadata']


def list_saved_models() -> List[Dict]:
    """
    List all saved models.
    
    Returns:
        List of model information dictionaries
    """
    models = []
    
    if not MODELS_DIR.exists():
        return models
    
    for filepath in MODELS_DIR.glob("*.joblib"):
        if '_latest' not in filepath.name:
            try:
                save_data = joblib.load(filepath)
                models.append({
                    'name': save_data['model_name'],
                    'timestamp': save_data['timestamp'],
                    'path': str(filepath),
                    'metadata': save_data.get('metadata', {})
                })
            except Exception:
                pass
    
    return sorted(models, key=lambda x: x['timestamp'], reverse=True)


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    Format metrics dictionary as a readable string.
    
    Args:
        metrics: Dictionary of metric names and values
        prefix: Optional prefix for each line
        
    Returns:
        Formatted string
    """
    lines = []
    for name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{prefix}{name}: {value:.4f}")
        else:
            lines.append(f"{prefix}{name}: {value}")
    return "\n".join(lines)


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y: Array of labels
        
    Returns:
        Dictionary mapping class labels to weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return dict(zip(classes, weights))


def print_banner(title: str, width: int = 60) -> None:
    """Print a formatted banner."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")


def print_section(title: str, width: int = 60) -> None:
    """Print a section header."""
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width)
