"""
train.py — Train, evaluate, and save the House Price Predictor model.
Usage: python train.py
"""

import numpy as np
import pandas as pd
from datetime import date
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from utils import (
    PROCESSED_DATA_PATH, save_model, save_metadata, rmse
)


def load_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]
    return X, y


def evaluate(model, X_train, y_train, X_test, y_test, name):
    """Train, cross-validate and print results for a model."""
    model.fit(X_train, np.log1p(y_train))

    # Cross-validation score on training set
    cv_scores = cross_val_score(
        model, X_train, np.log1p(y_train),
        cv=5, scoring="neg_root_mean_squared_error"
    )
    cv_rmse = -cv_scores.mean()

    # Test set score
    preds = np.expm1(model.predict(X_test))
    test_rmse = rmse(y_test, preds)
    r2 = model.score(X_test, np.log1p(y_test))

    print(f"\n📊 {name}")
    print(f"   CV RMSE (log):   {cv_rmse:.4f}")
    print(f"   Test RMSE ($):   ${np.expm1(test_rmse):,.0f}")
    print(f"   Test R²:         {r2:.4f}")

    return {"rmse_log": cv_rmse, "test_rmse": float(np.expm1(test_rmse)), "r2": float(r2)}


def train():
    print("🏠 House Price Predictor — Training\n" + "="*40)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # ── Models to compare ──────────────────────────────────────
    models = {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0
        ),
    }

    results = {}
    for name, model in models.items():
        results[name] = evaluate(model, X_train, y_train, X_test, y_test, name)

    # ── Pick best model by CV RMSE ─────────────────────────────
    best_name = min(results, key=lambda k: results[k]["rmse_log"])
    best_model = models[best_name]

    print(f"\n🏆 Best model: {best_name}")

    # ── Retrain best model on full data ───────────────────────
    best_model.fit(X, np.log1p(y))
    save_model(best_model)

    # ── Save metadata ──────────────────────────────────────────
    save_metadata({
        "model": best_name,
        "rmse": results[best_name]["test_rmse"],
        "r2": results[best_name]["r2"],
        "trained_on": str(date.today()),
        "n_features": X.shape[1],
        "features": X.columns.tolist(),
    })

    print("\n✅ Training complete! Model and metadata saved.")
    return best_model


if __name__ == "__main__":
    train()
