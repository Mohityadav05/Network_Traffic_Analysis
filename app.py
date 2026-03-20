import streamlit as st
import pandas as pd
import numpy as np
import joblib
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
import os
import glob

# Set page config
st.set_page_config(page_title="Network Traffic Classifier", layout="wide")

st.title("Network Traffic Classifier (VPN / Non-VPN)")

@st.cache_resource
def load_models():
    # Find models in directory
    keras_models = glob.glob('*.keras')
    joblib_models = [m for m in glob.glob('*.joblib') if 'scaler' not in m.lower()]
    scaler_file = glob.glob('scaler*.joblib')
    
    scaler = None
    if scaler_file:
        scaler = joblib.load(scaler_file[-1])
        
    return keras_models, joblib_models, scaler

keras_models, joblib_models, scaler = load_models()

@st.cache_data
def load_sample_data():
    try:
        # Load just a few rows to get features
        df = pd.read_csv('Cleaned_Darknet.csv', nrows=100)
        # Drop label columns
        features = df.drop(['Label', 'is_vpn'], axis=1, errors='ignore')
        return df, features.columns.tolist()
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None, []

df_sample, feature_cols = load_sample_data()

st.sidebar.header("Configuration")
model_type = st.sidebar.radio("Select Model Type", ["Neural Network (.keras)", "Machine Learning (.joblib)"], index=1)

selected_model = None
model_instance = None

if model_type == "Neural Network (.keras)" and keras_models:
    if not TF_AVAILABLE:
        st.sidebar.error("TensorFlow is not available. Please install it to use .keras models.")
    else:
        selected_model = st.sidebar.selectbox("Select Model", keras_models)
        if selected_model:
            model_instance = tf.keras.models.load_model(selected_model)
elif model_type == "Machine Learning (.joblib)" and joblib_models:
    selected_model = st.sidebar.selectbox("Select Model", joblib_models)
    if selected_model:
        model_instance = joblib.load(selected_model)
else:
    st.sidebar.warning("No models found of this type.")

if not scaler:
    st.warning("No Scaler found! Predictions might be inaccurate.")

st.markdown("### Input Parameters")
st.write("Since there are many features, you can select a sample row to populate the inputs, and then tweak them.")

if df_sample is not None and len(feature_cols) > 0:
    sample_idx = st.selectbox("Select Sample Network Flow", range(len(df_sample)), format_func=lambda x: f"Sample {x} (Original Label: {df_sample.iloc[x].get('Label', 'Unknown')})")
    
    # Initialize session state for features
    if 'current_features' not in st.session_state or st.session_state.get('last_sample_idx') != sample_idx:
        st.session_state['current_features'] = df_sample.iloc[sample_idx][feature_cols].to_dict()
        st.session_state['last_sample_idx'] = sample_idx
        
    # Group inputs into categories for better UI
    
    # Show a few key features directly
    key_features = ['Src Port', 'Dst Port', 'Protocol', 'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets']
    
    st.markdown("#### Key Features")
    k_col1, k_col2, k_col3 = st.columns(3)
    cols = [k_col1, k_col2, k_col3]
    for i, feature in enumerate(key_features):
        if feature in feature_cols:
            val = st.session_state['current_features'][feature]
            new_val = cols[i%3].number_input(feature, value=float(val), key=f"key_{feature}")
            st.session_state['current_features'][feature] = new_val

    with st.expander("Show all 70+ Advanced Network Features"):
        st.write("Modify any underlying packet statistics here:")
        a_col1, a_col2, a_col3, a_col4 = st.columns(4)
        a_cols = [a_col1, a_col2, a_col3, a_col4]
        
        # Don't show key features again
        adv_features = [f for f in feature_cols if f not in key_features]
        for i, feature in enumerate(adv_features):
            val = st.session_state['current_features'][feature]
            new_val = a_cols[i%4].number_input(feature, value=float(val), key=f"adv_{feature}", label_visibility="visible")
            st.session_state['current_features'][feature] = new_val

    st.markdown("---")
    if st.button("Predict Network Traffic Type", type="primary", use_container_width=True):
        if model_instance is None:
            st.error("Please load a model first.")
        else:
            with st.spinner("Analyzing parameters..."):
                # Prepare input
                input_df = pd.DataFrame([st.session_state['current_features']])
                
                # Replace inf and nan
                input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                input_df.fillna(0, inplace=True) # basic fill
                
                # Scale
                if scaler:
                    try:
                        input_scaled = scaler.transform(input_df)
                    except Exception as e:
                        st.warning(f"Scaler failed, using unscaled normal: {e}")
                        input_scaled = input_df.values
                else:
                    input_scaled = input_df.values
                
                # Predict
                if model_type == "Neural Network (.keras)":
                    if not TF_AVAILABLE:
                        st.error("TensorFlow is not available.")
                        st.stop()
                    pred_prob = model_instance.predict(input_scaled, verbose=0)[0][0]
                    is_vpn = pred_prob > 0.5
                    confidence = pred_prob if is_vpn else 1 - pred_prob
                else:
                    # Joblib model
                    if hasattr(model_instance, "predict_proba"):
                        pred_prob = model_instance.predict_proba(input_scaled)[0]
                        # Assume class 1 is VPN if two classes
                        prob = pred_prob[1] if len(pred_prob) > 1 else pred_prob[0]
                        is_vpn = prob > 0.5
                        confidence = prob if is_vpn else 1 - prob
                    else:
                        pred = model_instance.predict(input_scaled)[0]
                        is_vpn = bool(pred)
                        confidence = 1.0
                
                # Display output
                st.markdown("### Result")
                if is_vpn:
                    st.error(f"🚨 **MALICIOUS / VPN TRAFFIC DETECTED** (Confidence: {confidence:.2%})")
                else:
                    st.success(f"✅ **NORMAL TRAFFIC** (Confidence: {confidence:.2%})")
                    
else:
    st.info("No data available. Please ensure Cleaned_Darknet.csv is in the directory.")
