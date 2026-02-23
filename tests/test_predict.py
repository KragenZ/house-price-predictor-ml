"""
tests/test_predict.py — Basic sanity tests for the prediction pipeline.
Run: pytest tests/
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import numpy as np
from utils import rmse


# ── Test 1: RMSE calculation ──────────────────────────────────
def test_rmse_perfect_prediction():
    y = np.array([100000, 200000, 300000])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_is_positive():
    y_true = np.array([100000, 200000])
    y_pred = np.array([110000, 190000])
    assert rmse(y_true, y_pred) > 0


def test_rmse_worse_prediction_is_higher():
    y_true = np.array([200000, 200000])
    y_close = np.array([210000, 190000])
    y_far   = np.array([300000, 100000])
    assert rmse(y_true, y_close) < rmse(y_true, y_far)


# ── Test 2: Input dict has expected keys ──────────────────────
def test_input_dict_has_required_keys():
    required = ["Overall Qual", "Gr Liv Area", "TotalSF", "TotalBath"]
    sample = {
        "Overall Qual": 7,
        "Gr Liv Area": 1500,
        "TotalSF": 2500,
        "TotalBath": 2.5,
    }
    for key in required:
        assert key in sample, f"Missing key: {key}"


# ── Test 3: Prediction is a reasonable dollar amount ──────────
# This test only runs if a trained model exists
def test_prediction_in_reasonable_range():
    try:
        from predict import predict_price
        price = predict_price({
            "Overall Qual": 6,
            "Gr Liv Area": 1400,
            "TotalSF": 2100,
            "TotalBath": 2.0,
            "HouseAge": 25,
        })
        assert 50_000 < price < 1_000_000, f"Prediction ${price:,.0f} is out of range"
    except FileNotFoundError:
        pytest.skip("Model not trained yet — run train.py first")
