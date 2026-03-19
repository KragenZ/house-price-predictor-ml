"""
utils.py — Shared helper functions used across the project.
"""

import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# ── Paths ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pkl"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
RAW_DATA_PATH = ROOT / "Data" / "Raw" / "housing.csv"
PROCESSED_DATA_PATH = ROOT / "Data" / "processed" / "housing_clean.csv"


# ── Model I/O ─────────────────────────────────────────────────
def save_model(model, path=MODEL_PATH):
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"No model found at {path}. Run train.py first.")
    return joblib.load(path)


# ── Metadata ──────────────────────────────────────────────────
def save_metadata(metadata: dict, path=METADATA_PATH):
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to {path}")


def load_metadata(path=METADATA_PATH):
    with open(path, "r") as f:
        return json.load(f)


# ── Metrics ───────────────────────────────────────────────────
def rmse(y_true, y_pred):
    """Root Mean Squared Error on log-transformed predictions."""
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

def mae(y_true, y_pred):
    """Mean Absolute Error ($)."""
    return mean_absolute_error(y_true, y_pred)
