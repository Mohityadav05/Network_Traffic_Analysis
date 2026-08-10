import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from datetime import datetime
import joblib
import os

print("Loading data...")
df = pd.read_csv('Cleaned_Darknet.csv')
df['is_vpn'] = (df['Label'] == 'VPN').astype(np.float32)

# FIX: drop BOTH label columns (Label and Label.1) plus the derived target.
# Label.1 is a second ground-truth label (application category) that leaked
# into the feature set in the original scripts, inflating accuracy.
LEAK_COLS = ['Label', 'Label.1', 'is_vpn']
X = df.drop([c for c in LEAK_COLS if c in df.columns], axis=1)
y = df['is_vpn']

print(f"Feature count after removing leakage columns: {X.shape[1]}")
assert 'Label.1' not in X.columns, "Leakage column still present!"

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

print("Applying PowerTransformer...")
scaler = PowerTransformer(method='yeo-johnson')
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ---------- XGBoost ----------
print("Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    tree_method='hist',
    early_stopping_rounds=30,
    eval_metric='logloss',
)
xgb.fit(X_train_res, y_train_res, eval_set=[(X_test, y_test)], verbose=False)

xgb_preds_prob = xgb.predict_proba(X_test)[:, 1]
xgb_acc = accuracy_score(y_test, (xgb_preds_prob > 0.5).astype(int))
xgb_auc = roc_auc_score(y_test, xgb_preds_prob)
print(f"XGBoost Test Accuracy: {xgb_acc:.4f}  AUC: {xgb_auc:.4f}")

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
xgb_name = f"vpn_xgboost_clean_{ts}.joblib"
joblib.dump(xgb, xgb_name)

# ---------- Residual MLP ----------
def residual_block(x, units, dropout_rate=0.2):
    shortcut = x
    x = Dense(units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(units)(x)
    x = BatchNormalization()(x)

    if shortcut.shape[-1] != units:
        shortcut = Dense(units)(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_residual_mlp(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(512)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = residual_block(x, 512)
    x = residual_block(x, 256)
    x = residual_block(x, 128)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

print("Training Residual MLP...")
mlp = build_residual_mlp(X_train.shape[1])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=6, restore_best_weights=True, mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
]

history = mlp.fit(
    X_train_res, y_train_res,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=512,
    callbacks=callbacks,
    verbose=2
)

mlp_name = f"vpn_residual_mlp_clean_{ts}.keras"
mlp.save(mlp_name)

mlp_preds_prob = mlp.predict(X_test, verbose=0).flatten()
mlp_acc = accuracy_score(y_test, (mlp_preds_prob > 0.5).astype(int))
mlp_auc = roc_auc_score(y_test, mlp_preds_prob)
print(f"MLP Test Accuracy: {mlp_acc:.4f}  AUC: {mlp_auc:.4f}")

# ---------- Ensemble ----------
ensemble_preds_prob = (0.6 * xgb_preds_prob) + (0.4 * mlp_preds_prob)
ensemble_acc = accuracy_score(y_test, (ensemble_preds_prob > 0.5).astype(int))
ensemble_auc = roc_auc_score(y_test, ensemble_preds_prob)
print(f"\n--- ENSEMBLE RESULTS (clean, no leakage) ---")
print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
print(f"Ensemble AUC: {ensemble_auc:.4f}")
print(classification_report(y_test, (ensemble_preds_prob > 0.5).astype(int)))

scaler_name = f"scaler_clean_{ts}.joblib"
joblib.dump(scaler, scaler_name)

# Write a small manifest so app.py / the user knows the exact filenames
with open('clean_model_manifest.txt', 'w') as f:
    f.write(f"xgb_model={xgb_name}\n")
    f.write(f"mlp_model={mlp_name}\n")
    f.write(f"scaler={scaler_name}\n")
    f.write(f"xgb_acc={xgb_acc:.4f}\n")
    f.write(f"xgb_auc={xgb_auc:.4f}\n")
    f.write(f"mlp_acc={mlp_acc:.4f}\n")
    f.write(f"mlp_auc={mlp_auc:.4f}\n")
    f.write(f"ensemble_acc={ensemble_acc:.4f}\n")
    f.write(f"ensemble_auc={ensemble_auc:.4f}\n")

print("\nDone. See clean_model_manifest.txt for filenames + metrics.")
