import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from datetime import datetime
import os

print("Loading data for Final V2 training...")
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

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

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

model = build_residual_mlp(X_train.shape[1])
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
]

print("Starting training...")
history = model.fit(
    X_train_res, y_train_res,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=512,
    callbacks=callbacks,
    verbose=1
)

m_name = f"vpn_residual_mlp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
model.save(m_name)
print(f"Model saved as {m_name}")

loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Test Accuracy: {acc:.4f}")
print(f"Final Test AUC: {auc:.4f}")

import joblib
joblib.dump(scaler, 'scaler_v2.joblib')
