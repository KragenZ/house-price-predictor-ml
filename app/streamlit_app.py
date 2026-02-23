"""
streamlit_app.py — Interactive House Price Predictor web app.
Run: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import numpy as np
from predict import predict_price
from utils import load_metadata

# Page Config 
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 House Price Predictor")
st.markdown("Enter the details of a house to get an estimated sale price.")

# Load model metadata 
try:
    meta = load_metadata()
    st.sidebar.success(f"✅ Model: **{meta['model']}**")
    st.sidebar.metric("R² Score", f"{meta['r2']:.3f}")
    st.sidebar.metric("RMSE", f"${meta['rmse']:,.0f}")
    st.sidebar.caption(f"Trained on: {meta['trained_on']}")
except Exception:
    st.sidebar.warning("Model not loaded. Run train.py first.")

# Input Form 
st.subheader("🏗️ House Details")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 5)
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 500, 6000, 1500)
    total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 3000, 800)
    first_flr_sf = st.number_input("1st Floor Area (sq ft)", 500, 4000, 1000)
    second_flr_sf = st.number_input("2nd Floor Area (sq ft)", 0, 2000, 0)

with col2:
    garage_cars = st.selectbox("Garage Capacity (cars)", [0, 1, 2, 3, 4], index=2)
    full_bath = st.selectbox("Full Bathrooms", [1, 2, 3, 4], index=1)
    half_bath = st.selectbox("Half Bathrooms", [0, 1, 2], index=0)
    year_built = st.number_input("Year Built", 1872, 2023, 1990)
    year_remod = st.number_input("Year Remodeled", 1872, 2023, 1990)

yr_sold = 2010  

# Derived features
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

# Predict 
st.divider()
if st.button("💰 Predict Price", type="primary", use_container_width=True):
    try:
        price = predict_price(input_data)
        st.success(f"### Estimated Price: **${price:,.0f}**")
        st.balloons()

        # Show price range (±10%)
        low, high = price * 0.90, price * 1.10
        st.caption(f"Likely range: ${low:,.0f} – ${high:,.0f}")

    except FileNotFoundError:
        st.error("⚠️ Model not found. Please run `python src/train.py` first.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.divider()
st.caption("Built with Scikit-learn, XGBoost & Streamlit · Ames Housing Dataset")
