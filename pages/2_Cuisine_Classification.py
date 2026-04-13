# ============================================================
# Cuisine Classification Page
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import (
    load_cleaned_data,
    load_cuisine_model,
    get_price_label
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title = "🍜 Cuisine Classification",
    page_icon  = "🍜",
    layout     = "wide"
)

st.title("🍜 Cuisine Classification")
st.markdown(
    "Enter restaurant details to predict its "
    "**Cuisine Type**."
)
st.markdown("---")

# ============================================================
# LOAD RESOURCES
# ============================================================

df                  = load_cleaned_data()
model, scaler, le   = load_cuisine_model()

if df is None or model is None:
    st.error("❌ Could not load model or data.")
    st.stop()

# ============================================================
# INPUT FORM
# ============================================================

st.markdown("### 🔧 Enter Restaurant Details")

col1, col2, col3 = st.columns(3)

with col1:
    country_code = st.number_input(
        "🌍 Country Code",
        min_value = 1,
        max_value = 216,
        value     = 1
    )
    avg_cost = st.number_input(
        "💵 Average Cost for Two (₹)",
        min_value = 0,
        max_value = 100000,
        value     = 500,
        step      = 100
    )
    price_range = st.selectbox(
        "💰 Price Range",
        options     = [1, 2, 3, 4],
        format_func = get_price_label,
        index       = 1
    )

with col2:
    aggregate_rating = st.slider(
        "⭐ Aggregate Rating",
        min_value = 0.0,
        max_value = 5.0,
        value     = 3.5,
        step      = 0.1
    )
    votes = st.number_input(
        "🗳️ Number of Votes",
        min_value = 0,
        max_value = 15000,
        value     = 100
    )

with col3:
    has_table_booking = st.selectbox(
        "🪑 Has Table Booking?",
        options     = [1, 0],
        format_func = lambda x: "Yes" if x == 1 else "No"
    )
    has_online_delivery = st.selectbox(
        "🚴 Has Online Delivery?",
        options     = [1, 0],
        format_func = lambda x: "Yes" if x == 1 else "No"
    )
    is_delivering_now = st.selectbox(
        "📦 Is Delivering Now?",
        options     = [1, 0],
        format_func = lambda x: "Yes" if x == 1 else "No"
    )

st.markdown("---")

# ============================================================
# PREDICTION
# ============================================================

if st.button("🔮 Predict Cuisine", type="primary",
             use_container_width=True):

    input_data = pd.DataFrame([{
        'Country Code'        : country_code,
        'Average Cost for two': avg_cost,
        'Has Table booking'   : has_table_booking,
        'Has Online delivery' : has_online_delivery,
        'Is delivering now'   : is_delivering_now,
        'Price range'         : price_range,
        'Aggregate rating'    : aggregate_rating,
        'Votes'               : votes
    }])

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    cuisine_name = le.inverse_transform([prediction])[0]

    # Get prediction probabilities
    proba        = model.predict_proba(input_scaled)[0]
    top_3_idx    = proba.argsort()[::-1][:3]
    top_3_labels = le.inverse_transform(top_3_idx)
    top_3_proba  = proba[top_3_idx]

    # Display result
    st.markdown("---")
    st.markdown("### 🎯 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"### 🍽️ Predicted Cuisine: **{cuisine_name}**")
        st.markdown("#### Top 3 Predictions:")

        for label, prob in zip(top_3_labels, top_3_proba):
            st.markdown(f"**{label}**")
            st.progress(float(prob))
            st.caption(f"{prob*100:.1f}% confidence")

    with col2:
        # Show cuisine emoji mapping
        cuisine_emojis = {
            'North Indian' : '🍛',
            'Chinese'      : '🥢',
            'Fast Food'    : '🍔',
            'South Indian' : '🥘',
            'Mughlai'      : '🍖',
            'Bakery'       : '🍞',
            'Cafe'         : '☕',
            'Italian'      : '🍝',
            'Continental'  : '🥗',
            'Other'        : '🍽️'
        }
        emoji = cuisine_emojis.get(cuisine_name, '🍽️')

        st.markdown(
            f"<div style='text-align:center; "
            f"font-size:100px'>{emoji}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='text-align:center; "
            f"font-size:24px'><b>{cuisine_name}</b></div>",
            unsafe_allow_html=True
        )