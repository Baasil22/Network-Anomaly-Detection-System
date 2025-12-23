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

# Attack type mapping (NSL-KDD specific attacks to categories)
ATTACK_CATEGORIES = {
    'normal': 'Normal',
    # DoS attacks
    'apache2': 'DoS', 'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS',
    'mailbomb': 'DoS', 'pod': 'DoS', 'processtable': 'DoS', 'smurf': 'DoS',
    'teardrop': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',
    # Probe attacks
    'ipsweep': 'Probe', 'mscan': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'saint': 'Probe', 'satan': 'Probe',
    # R2L attacks
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'httptunnel': 'R2L', 'imap': 'R2L',
    'multihop': 'R2L', 'named': 'R2L', 'phf': 'R2L', 'sendmail': 'R2L',
    'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L',
    'warezmaster': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L',
    # U2R attacks
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'ps': 'U2R',
    'rootkit': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
}


def engineer_features(df):
    df = df.copy()
    df['byte_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)
    df['total_bytes'] = df['src_bytes'] + df['dst_bytes']
    df['log_src_bytes'] = np.log1p(df['src_bytes'])
    df['log_dst_bytes'] = np.log1p(df['dst_bytes'])
    df['srv_per_host'] = df['srv_count'] / (df['count'] + 1)
    df['error_rate_sum'] = df['serror_rate'] + df['rerror_rate']
    df['srv_error_sum'] = df['srv_serror_rate'] + df['srv_rerror_rate']
    df['host_same_srv_diff'] = df['dst_host_same_srv_rate'] - df['dst_host_diff_srv_rate']
    df['host_srv_ratio'] = df['dst_host_srv_count'] / (df['dst_host_count'] + 1)
    df['host_error_total'] = df['dst_host_serror_rate'] + df['dst_host_rerror_rate']
    return df


def main():
    print("\n" + "="*70)
    print("  MULTI-CLASS ATTACK TYPE CLASSIFICATION")
    print("  Categories: Normal, DoS, Probe, R2L, U2R")
    print("="*70)
    
    # Load data
    print("\n[1/6] Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'KDDTrain+.txt'), header=None, names=FEATURE_NAMES)
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'KDDTest+.txt'), header=None, names=FEATURE_NAMES)
    
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
    
    # Map attacks to categories
    print("\n[2/6] Mapping attack types...")
    train_df['attack_type'] = train_df['label'].map(lambda x: ATTACK_CATEGORIES.get(x, 'Unknown'))
    test_df['attack_type'] = test_df['label'].map(lambda x: ATTACK_CATEGORIES.get(x, 'Unknown'))
    
    # Remove unknown (very rare attacks not in our mapping)
    train_df = train_df[train_df['attack_type'] != 'Unknown']
    test_df = test_df[test_df['attack_type'] != 'Unknown']
    
    # Count distribution
    print("\n  Attack Type Distribution (Training):")
    for atype in ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']:
        count = sum(train_df['attack_type'] == atype)
        pct = 100 * count / len(train_df)
        print(f"    {atype}: {count:,} ({pct:.1f}%)")
    
    # Feature engineering
    print("\n[3/6] Engineering features...")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(['Normal', 'DoS', 'Probe', 'R2L', 'U2R'])
    
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
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Per-class accuracy
    print("  Per-Class Accuracy:")
    for i, cls in enumerate(label_encoder.classes_):
        mask = y_test == i
        if sum(mask) > 0:
            cls_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"    {cls}: {cls_acc*100:.1f}%")
    
    # Save
    print("\n  Saving model...")
    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_names': list(all_cols),
        'label_encoder': label_encoder,
        'model_type': 'RandomForest-300-MultiClass',
        'is_multiclass': True,
        'class_names': list(label_encoder.classes_),
        'accuracy': acc,
        'f1_score': f1,
        'timestamp': datetime.now().isoformat()
    }
    joblib.dump(bundle, os.path.join(MODELS_DIR, 'best_model.joblib'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    
    print("\n" + "="*70)
    print(f"  ★ MODEL SAVED - Multi-Class Attack Detection")
    print(f"  ★ Classes: Normal, DoS, Probe, R2L, U2R")
    print(f"  ★ Accuracy: {acc*100:.2f}%")
    print("  ★ Restart API: python api/app.py")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
