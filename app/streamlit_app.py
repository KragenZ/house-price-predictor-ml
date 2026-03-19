"""
streamlit_app.py — Interactive House Price Predictor web app.
Run: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from predict import predict_price
from utils import load_metadata, load_model, PROCESSED_DATA_PATH, ROOT

# Page Config 
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Data & Model Loading (Cached) ----
@st.cache_data
def get_dataset():
    # Attempt multiple path casings because Streamlit Cloud can cache old Linux Git topologies
    possible_paths = [
        PROCESSED_DATA_PATH,
        ROOT / "Data" / "processed" / "housing_clean.csv",
        ROOT / "data" / "Processed" / "housing_clean.csv"
    ]
    for p in possible_paths:
        if p.exists():
            return pd.read_csv(p)
            
    # Try finding it dynamically just in case
    for p in ROOT.rglob("housing_clean.csv"):
        return pd.read_csv(p)
        
    st.error(f"Dataset loading error. Looked in {PROCESSED_DATA_PATH} but could not find the file.")
    return None

@st.cache_resource
def get_model():
    try:
        return load_model()
    except Exception:
        return None

@st.cache_data
def get_metadata():
    try:
        return load_metadata()
    except Exception:
        return None

df = get_dataset()
model = get_model()
meta = get_metadata()

# ---- Sidebar ----
st.sidebar.title("🏠 House Price Predictor")
st.sidebar.markdown("Predict residential home prices using advanced machine learning models built on the Ames Housing Dataset.")

if meta:
    st.sidebar.subheader("🤖 Model Specs")
    st.sidebar.info(f"**Algorithm:** {meta.get('model', 'XGBoost')}")
    st.sidebar.metric("Test R² Score", f"{meta.get('r2', 0):.3f}")
    if 'mae' in meta:
        st.sidebar.metric("Mean Absolute Error", f"${meta['mae']:,.0f}")
    st.sidebar.metric("RMSE ($)", f"${meta.get('rmse', 0):,.0f}")
else:
    st.sidebar.warning("Model not found. Run `python src/train.py` first.")

# ---- Main UI ----
st.title("🏠 Advanced Property Valuation Engine")
st.markdown("Enter property details below to get an AI-driven valuation, along with market analytics and feature importance.")

tab_pred, tab_analytics, tab_model = st.tabs(["🔮 Predictor", "📊 Analytics & Visuals", "🤖 About Model"])

# ==========================================
# TAB 1: PREDICTOR
# ==========================================
with tab_pred:
    st.subheader("🏗️ Quick Estimate")
    col_a, col_r = st.columns(2)
    with col_a:
        area = st.number_input("Total Area (sq ft)", min_value=500, max_value=10000, value=1500, step=100)
    with col_r:
        rooms = st.number_input("Total Rooms (Bedrooms & Bathrooms)", min_value=1, max_value=15, value=6, step=1)

    with st.expander("⚙️ Advanced House Details (Optional)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 6)
            gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 500, 6000, value=area)
            total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 3000, value=int(area * 0.3))
            first_flr_sf = st.number_input("1st Floor Area (sq ft)", 500, 4000, value=gr_liv_area)
            second_flr_sf = st.number_input("2nd Floor Area (sq ft)", 0, 2000, value=0)

        with col2:
            garage_cars = st.selectbox("Garage Capacity (cars)", [0, 1, 2, 3, 4], index=2)
            full_bath = st.selectbox("Full Bathrooms", [1, 2, 3, 4], index=min(4, max(1, rooms // 2)))
            half_bath = st.selectbox("Half Bathrooms", [0, 1, 2], index=0)
            year_built = st.number_input("Year Built", 1872, 2023, value=2000)
            year_remod = st.number_input("Year Remodeled", 1872, 2023, value=2000)

    # Derived features calculation
    yr_sold = 2010  
    input_data = {
        "Overall Qual": overall_qual,
        "Gr Liv Area": gr_liv_area,
        "Total Bsmt SF": total_bsmt_sf,
        "1st Flr SF": first_flr_sf,
        "2nd Flr SF": second_flr_sf,
        "Garage Cars": garage_cars,
        "Full Bath": full_bath,
        "Half Bath": half_bath,
        "Bsmt Full Bath": 0,
        "Bsmt Half Bath": 0,
        "Year Built": year_built,
        "Year Remod/Add": year_remod,
        "Mas Vnr Area": 0,
        "TotalSF": total_bsmt_sf + first_flr_sf + second_flr_sf,
        "TotalBath": full_bath + half_bath * 0.5,
        "HouseAge": yr_sold - year_built,
        "RemodAge": yr_sold - year_remod,
        "WasRemodeled": int(year_built != year_remod),
        "HasGarage": int(garage_cars > 0),
        "HasBasement": int(total_bsmt_sf > 0),
        "HasPool": 0,
        "Garage Area": garage_cars * 200,
        "Pool Area": 0,
    }

    st.markdown("---")
    
    if st.button("💰 Predict Price", type="primary", use_container_width=True):
        if meta and model:
            price = predict_price(input_data)
            
            # Confidence Range using MAE
            mae_val = meta.get('mae', price * 0.10)
            low, high = price - mae_val, price + mae_val
            
            st.success(f"### Estimated Price: **${price:,.0f}**")
            st.caption(f"**Confidence Range (± MAE):** ${low:,.0f} – ${high:,.0f}")
            st.balloons()
            
            # Explainability logic
            st.markdown("#### 🔍 Why this price?")
            reasons = []
            
            if df is not None:
                median_area = df["TotalSF"].median()
                median_age = df["HouseAge"].median()
                median_qual = df["Overall Qual"].median()
                
                # Area impact
                area_diff = input_data["TotalSF"] - median_area
                if area_diff > 400:
                    reasons.append(f"✅ **Larger than average living area** (+{area_diff:,.0f} sq ft over median).")
                elif area_diff < -400:
                    reasons.append(f"📉 **Smaller living area** ({abs(area_diff):,.0f} sq ft below median).")
                
                # Age impact
                age_diff = median_age - input_data["HouseAge"]
                if age_diff > 15:
                    reasons.append(f"✅ **Newer construction** ({input_data['HouseAge']} years old vs {median_age:.0f} median) raises the premium.")
                elif age_diff < -15:
                    reasons.append(f"📉 **Older property** ({input_data['HouseAge']} years old) decreases the baseline valuation.")
                
                # Quality
                if input_data["Overall Qual"] > median_qual + 1:
                    reasons.append("✅ **Premium overall material and finish** quality boosts the price significantly.")
                elif input_data["Overall Qual"] < median_qual - 1:
                    reasons.append("📉 **Below average finish quality** heavily discounts the property.")

                if input_data["WasRemodeled"] == 1 and (input_data["HouseAge"] > 20):
                    reasons.append("✅ **Recent remodeling** successfully preserves the home's value despite its age.")

            if reasons:
                for r in reasons:
                    st.markdown("- " + r)
            else:
                st.markdown("- Your inputs are closely aligned with typical market averages in Ames, Iowa.")
                
        else:
            st.error("⚠️ Model not found. Please run `python src/train.py` locally first.")

# ==========================================
# TAB 2: ANALYTICS & VISUALS
# ==========================================
with tab_analytics:
    st.subheader("📊 Market Analysis & Distributions")
    if df is not None and model is not None and meta is not None:
        
        col_fig1, col_fig2 = st.columns(2)
        
        # 1. Price Distribution Chart
        with col_fig1:
            st.markdown("**Ames Property Price Distribution**")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.histplot(df["SalePrice"], kde=True, color="#2a9d8f", ax=ax1)
            # If a prediction was made, show it:
            if 'price' in locals():
                ax1.axvline(price, color='#e76f51', linestyle='--', linewidth=2, label='Your Prediction')
                ax1.legend()
            ax1.set_xlabel("Sale Price ($)")
            ax1.set_ylabel("Number of Houses")
            st.pyplot(fig1)
            
        # 2. Feature Importances Chart
        with col_fig2:
            st.markdown("**Top Key Price Drivers (XGBoost)**")
            try:
                # If the model is an sklearn Pipeline or raw XGBoost
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'feature_importances_'):
                    importances = model.named_steps['model'].feature_importances_
                else:
                    importances = None
                    
                if importances is not None:
                    # Map to feature names
                    features = meta.get("features", [f"Feature {i}" for i in range(len(importances))])
                    imp_series = pd.Series(importances, index=features)
                    top_features = imp_series.sort_values(ascending=False).head(10)
                    
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    sns.barplot(x=top_features.values, y=top_features.index, palette="mako", ax=ax2)
                    ax2.set_xlabel("Relative Importance Score")
                    st.pyplot(fig2)
                else:
                    st.info("Feature importances aren't supported by the currently loaded algorithm type.")
            except Exception as e:
                st.error(f"Could not load feature importances: {e}")
                
    else:
        st.warning("Data not available. Make sure you have fully ran the setup and preprocessing scripts.")

# ==========================================
# TAB 3: MODEL INFO
# ==========================================
with tab_model:
    st.subheader("🤖 Under the Hood")
    st.markdown("""
    This House Price Predictor is built utilizing advanced machine learning techniques on the standard 
    **Ames Housing Dataset**, which contains nearly 80 separate property variables.
    
    ### Pipeline Architecture
    1. **Data Engineering**: We constructed new multi-variate features like `Total SF` and `House Age`. 
    2. **Log Transformation**: Housing prices are typically right-skewed; predicting `log1p(SalePrice)` drastically improved regression performance.
    3. **Ensemble Modeling**: During training, we evaluated **Ridge Regression**, **Random Forest**, and evaluated out the champion algorithm: **XGBoost Regressor**, an industry-gold standard for tabular data.
    """)
    
    st.divider()
    
    if meta:
        st.markdown(f"#### Technical Metrics (`{meta.get('model', 'XGBoost')}`)")
        # Cleanly laying out stats
        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score (Accuracy)", f"{meta.get('r2', 0):.4f}")
        m2.metric("Mean Absolute Error", f"${meta.get('mae', 0):,.0f}")
        m3.metric("Root Mean Squared Error", f"${meta.get('rmse', 0):,.0f}")
        
    st.caption("Developed natively in Python using Scikit-Learn, XGBoost, Pandas, and Streamlit.")
