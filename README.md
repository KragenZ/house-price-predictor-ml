# 🏠 House Price Predictor
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://house-price-predictor-ml-nh9zrtguvhpmhwsys9qexr.streamlit.app/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Driven-orange.svg)](https://xgboost.readthedocs.io/)

A complete machine learning project that predicts residential home prices using the **Ames Housing Dataset** (80 features, ~1,500 properties). Built as an automated, end-to-end ML pipeline featuring data cleaning, model comparison, and a live Streamlit UI.

## 🔗 Live Demo
🚀 **[Launch House Price Predictor Web App](https://house-price-predictor-ml-nh9zrtguvhpmhwsys9qexr.streamlit.app/)**

---

## 📊 Model Performance

| Model | CV RMSE (log) | Test MAE ($) | Test R² |
|-------|--------------|--------------|---------|
| Ridge Regression | ~0.14 | ~$15,500 | ~0.87 |
| Random Forest | ~0.13 | ~$14,200 | ~0.89 |
| **XGBoost** | **~0.12** | **~$12,800** | **~0.91** |

---

## 🗂️ Project Structure

```
house-price-predictor-ml/
├── data/
│   ├── raw/              # Original dataset (never modified)
│   └── processed/        # Cleaned & engineered features
├── notebooks/
│   ├── 01_eda.ipynb      # Exploratory Data Analysis
│   └── 02_training.ipynb # Model training & comparison
├── src/
│   ├── preprocess.py     # Data cleaning & feature engineering
│   ├── train.py          # Model training & evaluation
│   ├── predict.py        # Inference logic
│   └── utils.py          # Shared helpers
├── app/
│   └── streamlit_app.py  # Interactive web app
├── models/
│   ├── model.pkl         # Trained model
│   └── model_metadata.json
└── tests/
    └── test_predict.py
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/KragenZ/house-price-predictor-ml
cd house-price-predictor-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Preprocess data
python src/preprocess.py

# 4. Train the model
python src/train.py

# 5. Launch the app
streamlit run app/streamlit_app.py
```

---

## 🔧 Key Features

- **Feature Engineering** — 10 new features (TotalSF, HouseAge, TotalBath, etc.)
- **Model Comparison** — Ridge, Random Forest, and XGBoost evaluated with 5-fold CV
- **Log Transform** — SalePrice transformed with `log1p` to reduce skew
- **Streamlit App** — Interactive UI with sliders, deployed publicly

---

## 📦 Tech Stack

`pandas` · `scikit-learn` · `xgboost` · `streamlit` · `matplotlib` · `seaborn`

---

## 📁 Dataset

[Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) — 1,460 residential properties in Ames, Iowa with 79 explanatory variables.
