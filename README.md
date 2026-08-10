# Network Traffic Analysis — VPN Traffic Classifier

Classifies network flows as **VPN** or **Non-VPN** using flow-level statistics
(packet timing, size, and inter-arrival features) from a CIC-Darknet-style
dataset. Ensemble of XGBoost + a residual MLP (Keras), served through a
Streamlit app.

## Live demo
Run locally or deploy on Streamlit Community Cloud (see below).

## Repository structure
- `app.py` — Streamlit app: pick a sample flow or upload a CSV, get a VPN / Non-VPN prediction with confidence.
- `Cleaned_Darknet.csv` — flow-level dataset (CIC-Darknet2020-style) used for training and for sample flows in the app.
- `train_clean.py` — training script (XGBoost + residual MLP + SMOTE).
- `research.ipynb` — exploration / experiments notebook.
- `vpn_xgboost_clean_v3.joblib`, `vpn_residual_mlp_clean_v3.keras` — trained models.
- `scaler_v3_clean.joblib` — PowerTransformer fit on the training features.
- `feature_names_clean.joblib` — exact ordered list of 79 input feature columns the models expect.

## A note on a bug that was fixed
The dataset has **two label columns**: `Label` (Non-Tor/NonVPN/Tor/VPN) and
`Label.1` (an encoded application category — Browsing, Chat, VOIP, etc.).
Earlier versions of the training scripts only dropped `Label`, so `Label.1`
leaked into the model as an input feature — in several categories it predicted
the traffic type with 99–100% purity. Both label columns are now dropped
before training (`train_clean.py`), and the shipped models were retrained
clean. Ensemble test performance after the fix: **98.16% accuracy, 0.998 AUC**
on a held-out 20% split — legitimate numbers based only on real flow
statistics.

## Quick start (local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. Go to https://share.streamlit.io, sign in with GitHub.
3. "New app" → select this repo → branch `main` → main file path `app.py`.
4. Deploy. First boot installs `tensorflow-cpu`, which can take a few minutes.

## Retraining
```bash
python train_clean.py
```
Produces a timestamped scaler + XGBoost + Keras model and a
`clean_model_manifest.txt` with metrics.
