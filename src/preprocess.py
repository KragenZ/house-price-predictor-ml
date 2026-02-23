"""
preprocess.py — Data cleaning and feature engineering for Ames Housing dataset.
Run this before train.py to generate data/processed/housing_clean.csv
"""

import pandas as pd
import numpy as np
from utils import RAW_DATA_PATH, PROCESSED_DATA_PATH


def load_raw_data():
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/AmesHousing.csv"
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"✅ Loaded from local: {RAW_DATA_PATH}")
    except FileNotFoundError:
        print("Local file not found. Downloading from URL...")
        df = pd.read_csv(url)
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"✅ Saved to {RAW_DATA_PATH}")
    return df


def drop_outliers(df):
    """Remove extreme outliers that hurt model performance."""
    # Known outliers in Ames dataset: large houses sold cheap
    df = df[~((df["Gr Liv Area"] > 4000) & (df["SalePrice"] < 300000))]
    return df


def fill_missing(df):
    """Fill missing values with domain knowledge."""

    # These NaN values actually mean "None" (no basement, no garage, etc.)
    none_cols = [
        "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2",
        "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond",
        "Fireplace Qu", "Pool QC", "Fence", "Misc Feature", "Alley", "Mas Vnr Type",
    ]
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Numeric NaNs that mean 0
    zero_cols = [
        "Garage Yr Blt", "Garage Area", "Garage Cars",
        "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF",
        "Bsmt Full Bath", "Bsmt Half Bath", "Mas Vnr Area",
    ]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill remaining with mode (most common value)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Fill remaining numeric with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    return df


def engineer_features(df):
    """Create new features that improve model performance."""

    # Total square footage (one of the strongest predictors)
    df["TotalSF"] = df["Total Bsmt SF"] + df["1st Flr SF"] + df["2nd Flr SF"]

    # Total bathrooms
    df["TotalBath"] = (
        df["Full Bath"] + df["Half Bath"] * 0.5 +
        df["Bsmt Full Bath"] + df["Bsmt Half Bath"] * 0.5
    )

    # House age and remodel age
    df["HouseAge"] = df["Yr Sold"] - df["Year Built"]
    df["RemodAge"] = df["Yr Sold"] - df["Year Remod/Add"]

    # Was the house remodeled?
    df["WasRemodeled"] = (df["Year Built"] != df["Year Remod/Add"]).astype(int)

    # Has a garage?
    df["HasGarage"] = (df["Garage Area"] > 0).astype(int)

    # Has a basement?
    df["HasBasement"] = (df["Total Bsmt SF"] > 0).astype(int)

    # Has a pool?
    df["HasPool"] = (df["Pool Area"] > 0).astype(int)

    return df


def encode_categoricals(df):
    """Label-encode ordinal features, one-hot encode nominal features."""

    # Ordinal quality mappings
    quality_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    ordinal_cols = [
        "Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond",
        "Heating QC", "Kitchen Qual", "Fireplace Qu",
        "Garage Qual", "Garage Cond", "Pool QC",
    ]
    for col in ordinal_cols:
        if col in df.columns:
            df[col] = df[col].map(quality_map).fillna(0)

    # One-hot encode remaining categoricals
    df = pd.get_dummies(df, drop_first=True)
    return df


def preprocess(save=True):
    print("🔄 Starting preprocessing pipeline...\n")

    df = load_raw_data()
    print(f"   Raw shape: {df.shape}")

    df = drop_outliers(df)
    print(f"   After outlier removal: {df.shape}")

    df = fill_missing(df)
    print(f"   Missing values filled")

    df = engineer_features(df)
    print(f"   Features engineered")

    # Drop ID/irrelevant columns
    drop_cols = ["Order", "PID", "Mo Sold", "Yr Sold"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    df = encode_categoricals(df)
    print(f"   Categoricals encoded")
    print(f"   Final shape: {df.shape}")

    if save:
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"\n✅ Saved to {PROCESSED_DATA_PATH}")

    return df


if __name__ == "__main__":
    preprocess()
