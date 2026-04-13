# ============================================================
# Restaurant Recommendation Page
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import (
    load_cleaned_data,
    load_recommender_models,
    get_top_cities,
    get_top_cuisines,
    get_price_label
)

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')
)
from src.recommendation_system import (
    build_feature_matrix,
    recommend_restaurants
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title = "🔍 Recommendations",
    page_icon  = "🔍",
    layout     = "wide"
)

st.title("🔍 Restaurant Recommendation System")
st.markdown(
    "Tell us your preferences and we'll find the "
    "**best restaurants** for you!"
)
st.markdown("---")

# ============================================================
# LOAD RESOURCES
# ============================================================

df             = load_cleaned_data()
tfidf, scaler  = load_recommender_models()

if df is None or tfidf is None:
    st.error("❌ Could not load data or models.")
    st.stop()

# Build feature matrix (cached)
@st.cache_resource
def get_feature_matrix(_df):
    return build_feature_matrix(_df)

with st.spinner("🔄 Building recommendation engine..."):
    feature_matrix, _, _, city_cols, df_updated = (
        get_feature_matrix(df)
    )

top_cities   = get_top_cities(df, top_n=50)
top_cuisines = get_top_cuisines(df, top_n=30)

# ============================================================
# INPUT FORM
# ============================================================

st.markdown("### 🎯 Your Preferences")

col1, col2, col3 = st.columns(3)

with col1:
    cuisine_pref = st.selectbox(
        "🍜 Preferred Cuisine",
        options = top_cuisines,
        index   = 0
    )
    city_pref = st.selectbox(
        "🌆 Preferred City",
        options = top_cities,
        index   = 0
    )

with col2:
    price_pref = st.selectbox(
        "💰 Price Range",
        options     = [1, 2, 3, 4],
        format_func = get_price_label,
        index       = 1
    )
    min_rating = st.slider(
        "⭐ Minimum Rating",
        min_value = 0.0,
        max_value = 5.0,
        value     = 3.0,
        step      = 0.5
    )

with col3:
    online_delivery = st.selectbox(
        "🚴 Online Delivery",
        options     = [None, 1, 0],
        format_func = lambda x: (
            "No Preference" if x is None
            else "Yes" if x == 1
            else "No"
        )
    )
    table_booking = st.selectbox(
        "🪑 Table Booking",
        options     = [None, 1, 0],
        format_func = lambda x: (
            "No Preference" if x is None
            else "Yes" if x == 1
            else "No"
        )
    )
    top_n = st.slider(
        "📋 Number of Recommendations",
        min_value = 3,
        max_value = 10,
        value     = 5
    )

st.markdown("---")

# ============================================================
# RECOMMENDATION
# ============================================================

if st.button("🔍 Find Restaurants", type="primary",
             use_container_width=True):

    with st.spinner("🔄 Finding best restaurants..."):
        recs = recommend_restaurants(
            df              = df_updated,
            feature_matrix  = feature_matrix,
            tfidf           = tfidf,
            scaler          = scaler,
            city_cols       = city_cols,
            cuisine_preference = cuisine_pref,
            price_range     = price_pref,
            city            = city_pref,
            min_rating      = min_rating,
            online_delivery = online_delivery,
            table_booking   = table_booking,
            top_n           = top_n
        )

    if len(recs) == 0:
        st.warning("⚠️ No restaurants found. Try relaxing filters.")
    else:
        st.markdown(
            f"### 🎉 Top {len(recs)} Recommendations "
            f"for {cuisine_pref} in {city_pref}"
        )

        # Display each recommendation as a card
        for idx, row in recs.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([3,2,2,2])

                with col1:
                    st.markdown(
                        f"**{idx}. {row['Restaurant Name']}**"
                    )
                    st.caption(f"🍽️ {row['Primary Cuisine']}")

                with col2:
                    st.metric(
                        "⭐ Rating",
                        f"{row['Aggregate rating']}"
                    )

                with col3:
                    st.metric(
                        "💰 Price",
                        get_price_label(int(row['Price range']))
                    )

                with col4:
                    st.metric(
                        "🎯 Match Score",
                        f"{row['Similarity Score']:.2%}"
                    )

                st.markdown("---")