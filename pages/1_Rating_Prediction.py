# ============================================================
# Rating Prediction Page
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import (
    load_rated_data,
    load_rating_model,
    get_top_cities,
    get_top_cuisines,
    format_rating,
    get_price_label
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title = "⭐ Rating Prediction",
    page_icon  = "⭐",
    layout     = "wide"
)

st.title("⭐ Restaurant Rating Prediction")
st.markdown(
    "Enter restaurant details below to predict "
    "its **Aggregate Rating**."
)
st.markdown("---")

# ============================================================
# LOAD RESOURCES
# ============================================================

df            = load_rated_data()
model, scaler = load_rating_model()

if df is None or model is None:
    st.error("❌ Could not load model or data.")
    st.stop()

top_cities   = get_top_cities(df)
top_cuisines = get_top_cuisines(df)

# ============================================================
# INPUT FORM
# ============================================================

st.markdown("### 🔧 Enter Restaurant Details")

col1, col2, col3 = st.columns(3)

with col1:
    city = st.selectbox(
        "🌆 City",
        options = top_cities,
        index   = 0
    )
    cuisine = st.selectbox(
        "🍜 Primary Cuisine",
        options = top_cuisines,
        index   = 0
    )
    country_code = st.number_input(
        "🌍 Country Code",
        min_value = 1,
        max_value = 216,
        value     = 1
    )

with col2:
    price_range = st.selectbox(
        "💰 Price Range",
        options = [1, 2, 3, 4],
        format_func = get_price_label,
        index   = 1
    )
    avg_cost = st.number_input(
        "💵 Average Cost for Two (₹)",
        min_value = 0,
        max_value = 100000,
        value     = 500,
        step      = 100
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
        options      = [1, 0],
        format_func  = lambda x: "Yes" if x == 1 else "No"
    )
    has_online_delivery = st.selectbox(
        "🚴 Has Online Delivery?",
        options      = [1, 0],
        format_func  = lambda x: "Yes" if x == 1 else "No"
    )
    is_delivering_now = st.selectbox(
        "📦 Is Delivering Now?",
        options      = [1, 0],
        format_func  = lambda x: "Yes" if x == 1 else "No"
    )

st.markdown("---")

# ============================================================
# PREDICTION
# ============================================================

# Encode cuisine and city
from sklearn.preprocessing import LabelEncoder

df_temp = df.copy()
df_temp['Primary Cuisine'] = df_temp['Cuisines'].apply(
    lambda x: x.split(',')[0].strip()
    if pd.notnull(x) else 'Unknown'
)
top_cuisine_list = (
    df_temp['Primary Cuisine']
    .value_counts().head(20).index.tolist()
)
cuisine_clean = cuisine if cuisine in top_cuisine_list else 'Other'

top_city_list = (
    df['City'].value_counts().head(20).index.tolist()
)
city_clean = city if city in top_city_list else 'Other'

# Build label encoders same way as training
df_temp['Primary Cuisine'] = df_temp['Primary Cuisine'].apply(
    lambda x: x if x in top_cuisine_list else 'Other'
)
df_temp['City Grouped'] = df['City'].apply(
    lambda x: x if x in top_city_list else 'Other'
)

le_cuisine = LabelEncoder()
le_city    = LabelEncoder()
le_cuisine.fit(df_temp['Primary Cuisine'])
le_city.fit(df_temp['City Grouped'])

try:
    cuisine_encoded = le_cuisine.transform([cuisine_clean])[0]
except:
    cuisine_encoded = 0

try:
    city_encoded = le_city.transform([city_clean])[0]
except:
    city_encoded = 0

# Predict button
if st.button("🔮 Predict Rating", type="primary",
             use_container_width=True):

    input_data = pd.DataFrame([{
        'Country Code'        : country_code,
        'Average Cost for two': avg_cost,
        'Has Table booking'   : has_table_booking,
        'Has Online delivery' : has_online_delivery,
        'Is delivering now'   : is_delivering_now,
        'Price range'         : price_range,
        'Votes'               : votes,
        'Primary Cuisine'     : cuisine_encoded,
        'City Grouped'        : city_encoded
    }])

    # Scale and predict
    input_scaled     = scaler.transform(input_data)
    predicted_rating = model.predict(input_scaled)[0]
    predicted_rating = round(np.clip(predicted_rating, 0, 5), 2)

    # Display result
    st.markdown("---")
    st.markdown("### 🎯 Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label = "Predicted Rating",
            value = f"{predicted_rating} / 5.0"
        )
    with col2:
        st.metric(
            label = "Rating Stars",
            value = format_rating(predicted_rating)
        )
    with col3:
        if predicted_rating >= 4.0:
            st.success("🌟 Excellent Restaurant!")
        elif predicted_rating >= 3.5:
            st.info("👍 Good Restaurant!")
        elif predicted_rating >= 3.0:
            st.warning("😐 Average Restaurant")
        else:
            st.error("👎 Below Average")

    # Progress bar
    st.markdown(f"**Rating: {predicted_rating} / 5.0**")
    st.progress(predicted_rating / 5.0)