# Final Research — Project Index

Short index and quick start for this repository containing dataset files, trained models, and the analysis notebook.

## Overview
- This repo holds Darknet-related datasets, processed splits, trained LSTM models, and an analysis notebook used for experiments.

## Repository structure
- `Cleaned_Darknet.csv` — cleaned CSV dataset used for model training and analysis.
- `research.ipynb` — Jupyter notebook containing data exploration and training/evaluation experiments.
- `vpn_binary_lstm_20251110_095646.keras` + `_meta.json` — saved Keras model and metadata (older run).
- `vpn_binary_lstm_20260201_112709.keras` + `_meta.json` — saved Keras model and metadata (later run).
- `.gitignore` — rules for files that should not be tracked (large raw datasets, checkpoints).

> Note: Some large/raw files (e.g., full raw `Darknet.csv` or large .npz splits) may be intentionally untracked.

## Quick start
1. Create and activate a Python virtual environment:

```bash
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1   # PowerShell
```

2. Install common dependencies (adjust versions as needed):

```bash
pip install -U pip
pip install numpy pandas jupyter tensorflow
```

3. Open the notebook for exploration and to re-run experiments:

```bash
jupyter notebook research.ipynb
```

4. Load a saved Keras model (example):

```python
from tensorflow.keras.models import load_model
model = load_model('vpn_binary_lstm_20260201_112709.keras')
```

## Data notes
- If you need raw datasets that are not tracked, check `.gitignore` and restore them from backups or a data storage location before running the notebook.

## Commit & push
- To add this README and push to a remote:

```bash
git add README.md
git commit -m "Add README with project index"
git push origin main
```

## License
- Add a `LICENSE` file if you want to publish this repository publicly.

---
Generated: README index for quick GitHub presentation.
