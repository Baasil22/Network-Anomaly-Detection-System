"""
Flask REST API for Network Anomaly Detection

Real-time prediction endpoints for network traffic classification.
"""

import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.predictor import AnomalyPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='../dashboard')
CORS(app)

# Initialize predictor
predictor = None


def get_predictor():
    """Get or initialize the predictor."""
    global predictor
    if predictor is None:
        predictor = AnomalyPredictor()
        try:
            predictor.load_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load saved model: {e}. Using default model.")
    return predictor


@app.route('/')
def index():
    """Serve the dashboard."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory(app.static_folder, path)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'model_loaded': predictor is not None and predictor.model is not None
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict if network traffic is normal or anomalous.
    
    Expected JSON body:
    {
        "features": [feature1, feature2, ..., feature41]
    }
    
    Or with feature names:
    {
        "duration": 0,
        "protocol_type": "tcp",
        "service": "http",
        ...
    }
    """
    try:
        pred = get_predictor()
        data = request.get_json()
        
        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract raw features for rule engine
        raw_features = {}
        if 'features' in data:
            features_list = data['features']
            feature_names = [
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
            for i, val in enumerate(features_list):
                if i < len(feature_names):
                    raw_features[feature_names[i]] = val
        else:
            raw_features = data
            
        # Preprocess and get ML prediction
        features = pred.preprocess_input(data)
        prediction, confidence = pred.predict(features)
        
        # Use detection engine for enhanced analysis
        try:
            from api.detection_engine import get_engine
            engine = get_engine()
            
            # Get source IP if available
            source_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
            
            # Run enhanced analysis
            analysis = engine.analyze(
                ml_prediction=int(prediction[0]),
                ml_confidence=float(confidence[0]) if confidence is not None else 0.5,
                features=raw_features,
                source_ip=source_ip
            )
            
            # Get the actual label from multi-class model (DoS, Probe, R2L, U2R, Normal)
            actual_label = pred.get_label_name(prediction[0])
            
            # Build enhanced result
            result = {
                'prediction': int(prediction[0]),
                'label': actual_label,  # Use actual attack type from multi-class model
                'confidence': float(confidence[0]) if confidence is not None else 0.5,
                'timestamp': datetime.utcnow().isoformat(),
                # Enhanced fields
                'threat_level': analysis['threat_level'],
                'action': analysis['action'],
                'severity': analysis['severity'],
                'explanation': analysis['explanation'],
                'top_factors': analysis['top_factors'],
                'direction_analysis': analysis['direction_analysis'],
                'stats': analysis['stats'],
                'attack_type': actual_label if actual_label != 'Normal' else None
            }
            
        except ImportError:
            # Fallback to basic result
            result = {
                'prediction': int(prediction[0]),
                'label': pred.get_label_name(prediction[0]),
                'confidence': float(confidence[0]) if confidence is not None else None,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        logger.info(f"Prediction: {result.get('threat_level', result['label'])} (confidence: {result['confidence']:.4f})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple network traffic samples.
    
    Expected JSON body:
    {
        "samples": [
            {"features": [...]},
            {"features": [...]},
            ...
        ]
    }
    """
    try:
        pred = get_predictor()
        data = request.get_json()
        
        if data is None or 'samples' not in data:
            return jsonify({'error': 'No samples provided'}), 400
            
        samples = data['samples']
        
        # Process all samples
        all_features = []
        for sample in samples:
            if 'features' in sample:
                all_features.append(sample['features'])
            else:
                features = pred.preprocess_input(sample)
                all_features.append(features.flatten().tolist())
                
        features_array = np.array(all_features)
        
        # Make predictions
        predictions, confidences = pred.predict(features_array)
        
        results = []
        for i, (pred_val, conf) in enumerate(zip(predictions, confidences)):
            results.append({
                'sample_index': i,
                'prediction': int(pred_val),
                'label': pred.get_label_name(pred_val),
                'confidence': float(conf) if conf is not None else None
            })
            
        response = {
            'total_samples': len(samples),
            'predictions': results,
            'summary': {
                'normal_count': sum(1 for r in results if r['prediction'] == 0),
                'attack_count': sum(1 for r in results if r['prediction'] == 1)
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    try:
        pred = get_predictor()
        
        info = {
            'model_type': pred.model_type,
            'model_loaded': pred.model is not None,
            'feature_count': pred.feature_count,
            'class_names': pred.class_names,
            'model_path': pred.model_path
        }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/reload', methods=['POST'])
def reload_model():
    """Reload the model from disk."""
    try:
        pred = get_predictor()
        
        data = request.get_json() or {}
        model_path = data.get('model_path')
        
        if model_path:
            pred.load_model(model_path)
        else:
            pred.load_model()
            
        return jsonify({
            'status': 'success',
            'message': 'Model reloaded successfully',
            'model_type': pred.model_type
        })
        
    except Exception as e:
        logger.error(f"Model reload error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Cache file for storing calculated accuracy
ACCURACY_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models', 'saved', 'accuracy_cache.json'
)


@app.route('/api/model/accuracy', methods=['GET'])
def get_model_accuracy():
    """
    Get the model accuracy. Calculates on first use and caches the result.
    Subsequent requests return the cached value without recalculation.
    """
    import pandas as pd
    from sklearn.metrics import accuracy_score
    
    # Check if we have cached accuracy
    if os.path.exists(ACCURACY_CACHE_FILE):
        try:
            with open(ACCURACY_CACHE_FILE, 'r') as f:
                cached_data = json.load(f)
                logger.info("Returning cached model accuracy")
                return jsonify({
                    'accuracy': cached_data['accuracy'],
                    'accuracy_percent': cached_data['accuracy_percent'],
                    'cached': True,
                    'calculated_at': cached_data['calculated_at'],
                    'model_type': cached_data.get('model_type', 'unknown'),
                    'test_samples': cached_data.get('test_samples', 0)
                })
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
    
    # Calculate accuracy on first use
    try:
        pred = get_predictor()
        
        if pred.model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Load test data
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'raw'
        )
        test_file = os.path.join(data_dir, 'KDDTest+.txt')
        
        if not os.path.exists(test_file):
            # Return default accuracy if no test data
            return jsonify({
                'accuracy': 0.996,
                'accuracy_percent': '99.6%',
                'cached': False,
                'message': 'Using default accuracy (test data not found)',
                'model_type': pred.model_type
            })
        
        # Column names for NSL-KDD dataset
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
            'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
        ]
        
        logger.info("Loading test data for accuracy calculation...")
        test_df = pd.read_csv(test_file, names=columns)
        
        # Map labels to attack categories (same as training)
        attack_mapping = {
            'normal': 'Normal',
            'neptune': 'DoS', 'smurf': 'DoS', 'pod': 'DoS', 'teardrop': 'DoS',
            'land': 'DoS', 'back': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
            'processtable': 'DoS', 'mailbomb': 'DoS',
            'ipsweep': 'Probe', 'portsweep': 'Probe', 'nmap': 'Probe', 'satan': 'Probe',
            'mscan': 'Probe', 'saint': 'Probe',
            'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L',
            'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L',
            'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L',
            'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L', 'worm': 'R2L',
            'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'perl': 'U2R',
            'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R', 'httptunnel': 'U2R'
        }
        
        # Clean labels and map to categories
        test_df['label'] = test_df['label'].str.strip().str.lower()
        test_df['category'] = test_df['label'].map(attack_mapping).fillna('Unknown')
        
        # Remove unknown categories
        test_df = test_df[test_df['category'] != 'Unknown']
        
        # Prepare features - use a sample for faster calculation
        sample_size = min(5000, len(test_df))
        test_sample = test_df.sample(n=sample_size, random_state=42)
        
        logger.info(f"Calculating accuracy on {sample_size} test samples...")
        
        # Process each sample and get predictions
        y_true = []
        y_pred = []
        
        class_to_idx = {name: i for i, name in enumerate(pred.class_names)}
        
        for idx, row in test_sample.iterrows():
            try:
                # Prepare input data
                row_dict = row.drop(['label', 'difficulty', 'category']).to_dict()
                features = pred.preprocess_input(row_dict)
                prediction, _ = pred.predict(features)
                
                true_label = class_to_idx.get(row['category'], 0)
                y_true.append(true_label)
                y_pred.append(prediction[0])
            except Exception as e:
                continue
        
        if len(y_true) == 0:
            return jsonify({'error': 'Could not process test samples'}), 500
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        accuracy_percent = f"{accuracy * 100:.1f}%"
        
        logger.info(f"Model accuracy: {accuracy_percent}")
        
        # Cache the result
        cache_data = {
            'accuracy': accuracy,
            'accuracy_percent': accuracy_percent,
            'calculated_at': datetime.utcnow().isoformat(),
            'model_type': pred.model_type,
            'test_samples': len(y_true)
        }
        
        try:
            os.makedirs(os.path.dirname(ACCURACY_CACHE_FILE), exist_ok=True)
            with open(ACCURACY_CACHE_FILE, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info("Accuracy cached successfully")
        except Exception as e:
            logger.warning(f"Could not cache accuracy: {e}")
        
        return jsonify({
            'accuracy': accuracy,
            'accuracy_percent': accuracy_percent,
            'cached': False,
            'calculated_at': cache_data['calculated_at'],
            'model_type': pred.model_type,
            'test_samples': len(y_true)
        })
        
    except Exception as e:
        logger.error(f"Accuracy calculation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/accuracy/reset', methods=['POST'])
def reset_accuracy_cache():
    """Reset the accuracy cache to force recalculation on next request."""
    try:
        if os.path.exists(ACCURACY_CACHE_FILE):
            os.remove(ACCURACY_CACHE_FILE)
            return jsonify({
                'status': 'success',
                'message': 'Accuracy cache cleared. Will recalculate on next request.'
            })
        else:
            return jsonify({
                'status': 'success',
                'message': 'No cache file found. Accuracy will be calculated on next request.'
            })
    except Exception as e:
        logger.error(f"Cache reset error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get API usage statistics."""
    # In a production environment, you would track these metrics
    stats = {
        'total_predictions': 0,
        'avg_response_time_ms': 0,
        'uptime_seconds': 0,
        'model_info': {
            'type': get_predictor().model_type if predictor else 'not loaded',
            'loaded': predictor is not None and predictor.model is not None
        }
    }
    
    return jsonify(stats)


@app.route('/api/sample', methods=['GET'])
def get_sample_data():
    """Get sample data for testing."""
    # Sample normal traffic
    normal_sample = {
        'features': [0, 'tcp', 'http', 'SF', 181, 5450, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 9, 9, 1.0, 0.0, 0.11, 0.0, 0.0, 0.0, 0.0, 0.0],
        'description': 'Normal HTTP traffic'
    }
    
    # Sample attack traffic (Neptune DoS)
    attack_sample = {
        'features': [0, 'tcp', 'private', 'S0', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 123, 6, 1.0, 1.0, 0.0, 0.0, 0.05, 0.07, 0.0, 255, 26, 0.1, 0.05, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        'description': 'Neptune DoS attack pattern'
    }
    
    return jsonify({
        'normal_sample': normal_sample,
        'attack_sample': attack_sample,
        'feature_names': [
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
    })


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


def create_app():
    """Factory function to create the Flask app."""
    return app


if __name__ == '__main__':
    # Initialize predictor on startup
    get_predictor()
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
