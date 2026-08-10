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

st.set_page_config(page_title="Network Traffic Classifier", layout="wide")
st.title("Network Traffic Classifier (VPN / Non-VPN)")
st.caption(
    "Ensemble of XGBoost + a residual MLP, trained on CIC-Darknet-style flow "
    "statistics (packet timing, size, and inter-arrival features)."
)

# Columns that must never be used as model inputs: they are ground-truth
# labels from the source dataset, not observable network features.
LABEL_COLUMNS = ["Label", "Label.1", "is_vpn"]


@st.cache_resource
def load_models():
    keras_models = sorted(glob.glob("*.keras"))
    joblib_models = sorted(
        [m for m in glob.glob("*.joblib") if "scaler" not in m.lower() and "feature_names" not in m.lower()]
    )
    scaler_file = sorted(glob.glob("scaler*.joblib"))
    feature_file = sorted(glob.glob("feature_names*.joblib"))

    scaler = joblib.load(scaler_file[-1]) if scaler_file else None
    feature_names = joblib.load(feature_file[-1]) if feature_file else None

    return keras_models, joblib_models, scaler, feature_names


keras_models, joblib_models, scaler, feature_names = load_models()


@st.cache_data
def load_sample_data(feature_names):
    try:
        df_full = pd.read_csv("Cleaned_Darknet.csv")
        df_vpn = df_full[df_full["Label"] == "VPN"].head(50)
        df_non = df_full[df_full["Label"] != "VPN"].head(50)
        df = pd.concat([df_non, df_vpn]).reset_index(drop=True)

        if feature_names is not None:
            cols = [c for c in feature_names if c in df.columns]
        else:
            cols = [c for c in df.columns if c not in LABEL_COLUMNS]

        return df, cols
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None, []


df_sample, feature_cols = load_sample_data(feature_names)

st.sidebar.header("Configuration")
model_type = st.sidebar.radio(
    "Select Model Type", ["Neural Network (.keras)", "Machine Learning (.joblib)"], index=1
)

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


def prepare_and_predict(input_df):
    """Scale + run the selected model, return (is_vpn array, confidence array)."""
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)
    input_df = input_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    if scaler:
        try:
            X = scaler.transform(input_df)
        except Exception as e:
            st.warning(f"Scaler failed, using unscaled input: {e}")
            X = input_df.values
    else:
        X = input_df.values

    if model_type == "Neural Network (.keras)":
        if not TF_AVAILABLE:
            st.error("TensorFlow is not available.")
            st.stop()
        probs = model_instance.predict(X, verbose=0).flatten()
    else:
        if hasattr(model_instance, "predict_proba"):
            proba = model_instance.predict_proba(X)
            probs = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        else:
            probs = model_instance.predict(X).astype(float)

    is_vpn = probs > 0.5
    confidence = np.where(is_vpn, probs, 1 - probs)
    return is_vpn, confidence


tab_single, tab_batch = st.tabs(["Single Flow (interactive)", "Batch CSV Upload"])

with tab_single:
    st.markdown("### Input Parameters")
    st.write("Select a sample row to populate the inputs, then tweak them.")

    if df_sample is not None and len(feature_cols) > 0:
        sample_idx = st.selectbox(
            "Select Sample Network Flow",
            range(len(df_sample)),
            format_func=lambda x: f"Sample {x} (Original Label: {df_sample.iloc[x].get('Label', 'Unknown')})",
        )

        if "current_features" not in st.session_state or st.session_state.get("last_sample_idx") != sample_idx:
            st.session_state["current_features"] = df_sample.iloc[sample_idx][feature_cols].to_dict()
            st.session_state["last_sample_idx"] = sample_idx

        key_features = ["Src Port", "Dst Port", "Protocol", "Flow Duration", "Total Fwd Packet", "Total Bwd packets"]

        st.markdown("#### Key Features")
        k_cols = st.columns(3)
        for i, feature in enumerate(key_features):
            if feature in feature_cols:
                val = st.session_state["current_features"][feature]
                new_val = k_cols[i % 3].number_input(feature, value=float(val), key=f"key_{feature}")
                st.session_state["current_features"][feature] = new_val

        with st.expander(f"Show all {len(feature_cols)} network features"):
            a_cols = st.columns(4)
            adv_features = [f for f in feature_cols if f not in key_features]
            for i, feature in enumerate(adv_features):
                val = st.session_state["current_features"][feature]
                new_val = a_cols[i % 4].number_input(feature, value=float(val), key=f"adv_{feature}")
                st.session_state["current_features"][feature] = new_val

        st.markdown("---")
        if st.button("Predict Network Traffic Type", type="primary", use_container_width=True):
            if model_instance is None:
                st.error("Please load a model first.")
            else:
                with st.spinner("Analyzing parameters..."):
                    input_df = pd.DataFrame([st.session_state["current_features"]])
                    is_vpn, confidence = prepare_and_predict(input_df)

                st.markdown("### Result")
                if is_vpn[0]:
                    st.error(f"🔒 **VPN TRAFFIC DETECTED** (Confidence: {confidence[0]:.2%})")
                else:
                    st.success(f"✅ **NON-VPN TRAFFIC** (Confidence: {confidence[0]:.2%})")
    else:
        st.info("No data available. Please ensure Cleaned_Darknet.csv is in the directory.")

with tab_batch:
    st.markdown("### Upload a CSV of flow records")
    st.write(
        "Upload a CSV with the same flow-statistic columns as the training data "
        "(no `Label` / `Label.1` columns needed — they're ignored if present)."
    )
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            missing = [c for c in feature_cols if c not in batch_df.columns]
            if missing:
                st.error(f"Uploaded CSV is missing {len(missing)} expected columns, e.g.: {missing[:5]}")
            elif model_instance is None:
                st.error("Please select a model in the sidebar first.")
            else:
                with st.spinner(f"Scoring {len(batch_df)} flows..."):
                    is_vpn, confidence = prepare_and_predict(batch_df[feature_cols])
                result_df = batch_df.copy()
                result_df["Predicted"] = np.where(is_vpn, "VPN", "Non-VPN")
                result_df["Confidence"] = confidence
                st.success(f"Scored {len(result_df)} flows.")
                st.dataframe(result_df[["Predicted", "Confidence"] + feature_cols[:5]], use_container_width=True)
                st.download_button(
                    "Download full results as CSV",
                    result_df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Failed to process file: {e}")
