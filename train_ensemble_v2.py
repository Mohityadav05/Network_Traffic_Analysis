import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os
from datetime import datetime

# --- 1. Load and Preprocessing ---
print("Loading data...")
df = pd.read_csv('Cleaned_Darknet.csv')
df['is_vpn'] = (df['Label'] == 'VPN').astype(np.float32)

X = df.drop(['Label', 'is_vpn'], axis=1)
y = df['is_vpn']

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

print("Applying PowerTransformer...")
scaler = PowerTransformer(method='yeo-johnson')
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- 2. Train XGBoost ---
print("Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    tree_method='hist',
    early_stopping_rounds=50
)

xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

xgb_preds_prob = xgb.predict_proba(X_test)[:, 1]
xgb_acc = accuracy_score(y_test, (xgb_preds_prob > 0.5).astype(int))
print(f"XGBoost Test Accuracy: {xgb_acc:.4f}")

# Save XGBoost
xgb_name = f"vpn_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
joblib.dump(xgb, xgb_name)

# --- 3. Load MLP and Ensemble ---
mlp_files = [f for f in os.listdir('.') if f.startswith('vpn_residual_mlp_') and f.endswith('.keras')]
mlp_files.sort(reverse=True)
if mlp_files:
    mlp_path = mlp_files[0]
    print(f"Loading MLP: {mlp_path}")
    mlp = tf.keras.models.load_model(mlp_path)
    mlp_preds_prob = mlp.predict(X_test, verbose=0).flatten()
    
    # Simple Weighting (Tune these if needed)
    ensemble_preds_prob = (0.6 * xgb_preds_prob) + (0.4 * mlp_preds_prob)
    ensemble_acc = accuracy_score(y_test, (ensemble_preds_prob > 0.5).astype(int))
    ensemble_auc = roc_auc_score(y_test, ensemble_preds_prob)
    
    print(f"\n--- ENSEMBLE RESULTS ---")
    print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
    print(f"Ensemble AUC: {ensemble_auc:.4f}")
    print(classification_report(y_test, (ensemble_preds_prob > 0.5).astype(int)))
    
    if ensemble_acc >= 0.98:
        print("SUCCESS: Targeted accuracy reached!")
    else:
        print("ALMOST THERE: Consider further tuning or stacking.")
else:
    print("No MLP found for ensemble.")

# Save Scaler
joblib.dump(scaler, 'scaler_v2.joblib')
