"""
Multi-Class Attack Type Classification
Identifies specific attack types: Normal, DoS, Probe, R2L, U2R
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved')
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_NAMES = [
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
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# Specific attack types to detect (23 classes)
# Only include attacks with sufficient samples in training data
SPECIFIC_ATTACKS = {
    # Normal traffic
    'normal': {'name': 'Normal', 'category': 'Normal', 'severity': 'safe'},
    
    # DoS attacks (Denial of Service)
    'neptune': {'name': 'Neptune', 'category': 'DoS', 'severity': 'critical'},
    'smurf': {'name': 'Smurf', 'category': 'DoS', 'severity': 'critical'},
    'back': {'name': 'Back', 'category': 'DoS', 'severity': 'high'},
    'teardrop': {'name': 'Teardrop', 'category': 'DoS', 'severity': 'high'},
    'pod': {'name': 'Pod', 'category': 'DoS', 'severity': 'high'},
    'land': {'name': 'Land', 'category': 'DoS', 'severity': 'medium'},
    
    # Probe attacks (Reconnaissance)
    'satan': {'name': 'Satan', 'category': 'Probe', 'severity': 'medium'},
    'ipsweep': {'name': 'IPSweep', 'category': 'Probe', 'severity': 'medium'},
    'nmap': {'name': 'Nmap', 'category': 'Probe', 'severity': 'medium'},
    'portsweep': {'name': 'Portsweep', 'category': 'Probe', 'severity': 'medium'},
    
    # R2L attacks (Remote to Local)
    'warezclient': {'name': 'WarezClient', 'category': 'R2L', 'severity': 'high'},
    'guess_passwd': {'name': 'GuessPasswd', 'category': 'R2L', 'severity': 'high'},
    'warezmaster': {'name': 'WarezMaster', 'category': 'R2L', 'severity': 'high'},
    'imap': {'name': 'Imap', 'category': 'R2L', 'severity': 'medium'},
    'ftp_write': {'name': 'FTPWrite', 'category': 'R2L', 'severity': 'high'},
    'multihop': {'name': 'Multihop', 'category': 'R2L', 'severity': 'high'},
    'phf': {'name': 'Phf', 'category': 'R2L', 'severity': 'medium'},
    'spy': {'name': 'Spy', 'category': 'R2L', 'severity': 'critical'},
    
    # U2R attacks (User to Root - Privilege Escalation)
    'buffer_overflow': {'name': 'BufferOverflow', 'category': 'U2R', 'severity': 'critical'},
    'rootkit': {'name': 'Rootkit', 'category': 'U2R', 'severity': 'critical'},
    'loadmodule': {'name': 'Loadmodule', 'category': 'U2R', 'severity': 'critical'},
    'perl': {'name': 'Perl', 'category': 'U2R', 'severity': 'high'},
}

# Keep category mapping for backwards compatibility
ATTACK_CATEGORIES = {k: v['category'] for k, v in SPECIFIC_ATTACKS.items()}


def engineer_features(df):
    """Enhanced feature engineering for better attack detection."""
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
    
    # DoS detection features (high connection counts with errors)
    df['dos_indicator'] = (df['count'] * df['serror_rate']) + (df['srv_count'] * df['srv_serror_rate'])
    df['syn_flood_indicator'] = df['count'] * (1 - df['same_srv_rate'])
    
    # Probe detection features (scanning patterns)
    df['probe_indicator'] = df['dst_host_count'] * df['dst_host_diff_srv_rate']
    df['scan_indicator'] = (1 - df['dst_host_same_srv_rate']) * df['dst_host_count']
    
    # R2L/U2R detection features (failed logins, root access)
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


def main():
    print("\n" + "="*70)
    print("  ADVANCED 23-CLASS ATTACK DETECTION")
    print("  Detects specific attack types: Neptune, Smurf, Nmap, etc.")
    print("="*70)
    
    # Load data
    print("\n[1/6] Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'KDDTrain+.txt'), header=None, names=FEATURE_NAMES)
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'KDDTest+.txt'), header=None, names=FEATURE_NAMES)
    
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
    
    # Map to specific attack names
    print("\n[2/6] Mapping to 23 specific attack types...")
    
    def get_attack_name(label):
        if label in SPECIFIC_ATTACKS:
            return SPECIFIC_ATTACKS[label]['name']
        return 'Unknown'
    
    train_df['attack_type'] = train_df['label'].map(get_attack_name)
    test_df['attack_type'] = test_df['label'].map(get_attack_name)
    
    # Remove unknown attacks (not in our 23 classes)
    train_df = train_df[train_df['attack_type'] != 'Unknown']
    test_df = test_df[test_df['attack_type'] != 'Unknown']
    
    # Get unique attack types
    attack_names = sorted(list(set(train_df['attack_type'].unique()) | set(test_df['attack_type'].unique())))
    print(f"  Detected {len(attack_names)} unique attack types")
    
    # Count distribution
    print("\n  Attack Type Distribution (Training):")
    for atype in attack_names:
        count = sum(train_df['attack_type'] == atype)
        pct = 100 * count / len(train_df) if len(train_df) > 0 else 0
        if count > 0:
            print(f"    {atype}: {count:,} ({pct:.1f}%)")
    
    # Feature engineering
    print("\n[3/6] Engineering features...")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    # Encode labels with all attack names
    label_encoder = LabelEncoder()
    label_encoder.fit(attack_names)
    
    y_train = label_encoder.transform(train_df['attack_type'])
    y_test = label_encoder.transform(test_df['attack_type'])
    
    # Prepare features
    exclude = ['label', 'difficulty', 'label_binary', 'attack_type']
    feat_cols = [c for c in train_df.columns if c not in exclude]
    
    combined = pd.concat([train_df[feat_cols], test_df[feat_cols]], ignore_index=True)
    combined_encoded = pd.get_dummies(combined, columns=CATEGORICAL_COLS)
    all_cols = sorted(combined_encoded.columns.tolist())
    
    train_encoded = combined_encoded.iloc[:len(train_df)][all_cols]
    test_encoded = combined_encoded.iloc[len(train_df):][all_cols]
    
    X_train = train_encoded.values
    X_test = test_encoded.values
    
    # Scale
    print("\n[4/6] Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"  Features: {X_train.shape[1]}")
    
    # Train multi-class model
    print("\n[5/6] Training multi-class RandomForest...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n[6/6] Evaluating on test set...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "="*70)
    print("  RESULTS")
    print("="*70)
    print(f"\n  ★ Overall Accuracy: {acc*100:.2f}%")
    print(f"  ★ Weighted F1-Score: {f1*100:.2f}%")
    
    print("\n  Classification Report:")
    # Get unique labels in both test predictions and actual
    unique_labels = sorted(list(set(y_test) | set(y_pred)))
    target_names_filtered = [label_encoder.classes_[i] for i in unique_labels]
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names_filtered, zero_division=0))
    
    # Per-class accuracy
    print("  Per-Class Accuracy:")
    for i, cls in enumerate(label_encoder.classes_):
        mask = y_test == i
        if sum(mask) > 0:
            cls_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"    {cls}: {cls_acc*100:.1f}%")
    
    # Save
    print("\n  Saving model...")
    
    # Build attack info for API
    attack_info = {name: info for name, info in SPECIFIC_ATTACKS.items()}
    
    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_names': list(all_cols),
        'label_encoder': label_encoder,
        'model_type': 'RandomForest-300-23Class',
        'is_multiclass': True,
        'class_names': list(label_encoder.classes_),
        'attack_info': attack_info,
        'num_classes': len(label_encoder.classes_),
        'accuracy': acc,
        'f1_score': f1,
        'timestamp': datetime.now().isoformat()
    }
    joblib.dump(bundle, os.path.join(MODELS_DIR, 'best_model.joblib'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    
    print("\n" + "="*70)
    print(f"  MODEL SAVED - Advanced 23-Class Attack Detection")
    print(f"  Classes: {len(label_encoder.classes_)} specific attack types")
    print(f"  ★ Accuracy: {acc*100:.2f}%")
    print("  ★ Restart API: python api/app.py")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
