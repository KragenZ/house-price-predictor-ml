"""
predict.py — Load trained model and make predictions.
Can be used standalone or imported by streamlit_app.py
"""

import numpy as np
import pandas as pd
from utils import load_model, load_metadata


def predict_price(input_dict: dict) -> float:
    """
    Predict house price from a dictionary of features.

    Args:
        input_dict: dict of feature_name → value

    Returns:
        Predicted price in dollars (float)
    """
    model = load_model()
    metadata = load_metadata()

    # Build a dataframe with the right columns (in the right order)
    features = metadata["features"]
    row = pd.DataFrame([input_dict]).reindex(columns=features, fill_value=0)

    log_pred = model.predict(row)[0]
    return float(np.expm1(log_pred))


def predict_from_csv(csv_path: str) -> pd.DataFrame:
    """Batch predictions from a CSV file."""
    model = load_model()
    metadata = load_metadata()

    df = pd.read_csv(csv_path)
    features = metadata["features"]
    X = df.reindex(columns=features, fill_value=0)

    log_preds = model.predict(X)
    df["PredictedPrice"] = np.expm1(log_preds)
    return df


if __name__ == "__main__":
    # Quick sanity check — replace values with your own
    sample = {
        "Overall Qual": 7,
        "Gr Liv Area": 1500,
        "TotalSF": 2500,
        "TotalBath": 2.5,
        "Garage Cars": 2,
        "HouseAge": 20,
    }
    price = predict_price(sample)
    print(f"💰 Predicted Price: ${price:,.0f}")
