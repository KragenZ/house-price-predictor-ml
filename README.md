# 🏠 House Price Predictor
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://house-price-predictor-ml-nh9zrtguvhpmhwsys9qexr.streamlit.app/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Driven-orange.svg)](https://xgboost.readthedocs.io/)

An end-to-end Machine Learning project that predicts residential real estate prices. It includes a complete data pipeline and a live Streamlit app that explains its own predictions.

## 🔗 Live Demo
**[Try the Web App!](https://house-price-predictor-ml-nh9zrtguvhpmhwsys9qexr.streamlit.app/)**

![Streamlit App Walkthrough](streamlit_demo.gif)

---

## 🎯 Motivation
Pricing a house properly is tough for both buyers and sellers. This project aims to make the process more transparent. Instead of just spitting out a number, the model uses SHAP-like feature attributions to show exactly *why* a house is priced the way it is (e.g., how much the overall condition or living area adds to the value).

---

## 🛠️ The Pipeline

### 1. Data Processing
- Used the Ames Housing Dataset (~1,500 properties, 80 features).
- Applied a `log1p` transformation to the `SalePrice` to fix the heavily skewed target variable.
- Cleaned up missing values (e.g., properly imputing zeros for houses without basements or garages).

### 2. Feature Engineering
- Added 10 custom features (`TotalSF`, `HouseAge`, `TotalBath`, etc.) to capture real-world aspects of a home.
- One-hot encoded categorical variables to get them model-ready without assuming an arbitrary order.

### 3. Model Selection
Tested Ridge, Random Forest, and XGBoost using 5-fold Cross-Validation. **XGBoost** won out thanks to its built-in regularization and ability to handle complex interactions.

| Model | CV RMSE (log) | Test MAE ($) | Test R² |
|-------|--------------|--------------|---------|
| Ridge Regression | ~0.14 | ~$15,500 | ~0.87 |
| Random Forest | ~0.13 | ~$14,200 | ~0.89 |
| **XGBoost** | **~0.12** | **~$12,800** | **~0.91** |

---

## 🗂️ Project Structure

```text
house-price-predictor-ml/
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Cleaned data ready for modeling
├── notebooks/
│   ├── 01_eda.ipynb      # EDA and visualizations
│   └── 02_training.ipynb # Model design experiments
├── src/
│   ├── preprocess.py     # Data cleaning pipeline
│   ├── train.py          # Training script
│   ├── predict.py        # Inference logic
│   └── utils.py          # Shared helpers
├── app/
│   └── streamlit_app.py  # Streamlit frontend
└── models/               # Saved .pkl model & schema
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/KragenZ/house-price-predictor-ml
cd house-price-predictor-ml
pip install -r requirements.txt

# Run the pipeline
python src/preprocess.py
python src/train.py

# Start the UI
streamlit run app/streamlit_app.py
```

---

## 📦 Tech Stack
- `pandas`, `numpy`, `scikit-learn`, `xgboost`
- `matplotlib`, `seaborn`
- `streamlit`

---

## 📁 Dataset
[Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) — Includes 79 explanatory variables representing almost every aspect of a residential home in Ames, Iowa.
