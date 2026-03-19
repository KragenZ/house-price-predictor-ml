# 🏠 House Price Predictor
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://house-price-predictor-ml-nh9zrtguvhpmhwsys9qexr.streamlit.app/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Driven-orange.svg)](https://xgboost.readthedocs.io/)

An end-to-end Machine Learning project to predict residential real estate prices, deployed as a live Streamlit application with interactive model interpretability.

## 🔗 Live Application & Demo
> 🚀 **[Try the House Price Predictor Web App here!](https://house-price-predictor-ml-nh9zrtguvhpmhwsys9qexr.streamlit.app/)**

<img src="streamlit_demo.webp" width="800" alt="Streamlit App Walkthrough" />

---

## 💡 Business Value
Accurate real-estate valuation is critical for buyers, sellers, and agencies. This application tackles the problem by providing a data-driven approach to house pricing. Through advanced feature engineering and an optimized XGBoost model, the system predicts precise market values based on ~80 property features while providing users with clear insights into *why* a specific price was estimated using SHAP-like feature attributions.

---

## 🛠️ The Data Science & ML Pipeline

### 1. Exploratory Data Analysis (EDA)
- Analyzed the Ames Housing Dataset comprising ~1,500 properties.
- Addressed skewed target variable (`SalePrice`) using a `log1p` transformation to stabilize variance and normalize the distribution.
- Handled missing data logically (e.g., zeroing out missing basement parameters where no basement exists).

### 2. Feature Engineering
- **Domain-specific features:** Created 10 new predictive features (e.g., `TotalSF`, `HouseAge`, `TotalBath`) to capture compound effects missed by raw variables.
- **Categorical Encoding:** One-hot encoded nominal variables to preserve non-ordinal distinct categories.

### 3. Model Selection & Optimization
Evaluated multiple algorithms using 5-fold Cross-Validation to ensure out-of-sample generalization. **XGBoost** outperformed other regressors by combining gradient-boosting capabilities with aggressive regularization.

| Model | CV RMSE (log) | Test MAE ($) | Test R² |
|-------|--------------|--------------|---------|
| Ridge Regression | ~0.14 | ~$15,500 | ~0.87 |
| Random Forest | ~0.13 | ~$14,200 | ~0.89 |
| **XGBoost (Selected)** | **~0.12** | **~$12,800** | **~0.91** |

---

## 🗂️ Project Architecture & Structure

```text
house-price-predictor-ml/
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Post-feature-engineering data
├── notebooks/
│   ├── 01_eda.ipynb      # EDA and visualizations
│   └── 02_training.ipynb # Model design experiments
├── src/
│   ├── preprocess.py     # Automated data cleaning pipeline
│   ├── train.py          # Model training & optimization script
│   ├── predict.py        # Inference module
│   └── utils.py          # Helper functions
├── app/
│   └── streamlit_app.py  # Interactive frontend (Streamlit)
└── models/               # Serialized model (.pkl) & metadata
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/KragenZ/house-price-predictor-ml
cd house-price-predictor-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Preprocess data and train the model
python src/preprocess.py
python src/train.py

# 4. Launch the local Streamlit application
streamlit run app/streamlit_app.py
```

---

## 📦 Tech Stack
- **Data Engineering:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `xgboost`
- **Visualization:** `matplotlib`, `seaborn`
- **Web App / Deployment:** `streamlit`

---

## 📁 Dataset Reference
[Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) — Ground truth dataset documenting residential properties in Ames, Iowa. Contains ~80 explanatory variables representing almost every aspect of a home.
