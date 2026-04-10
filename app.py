# ============================================================
# app.py — Main Home Page
# Restaurant ML Project — Cognifyz Technologies
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add utils to path
sys.path.append(os.path.dirname(__file__))
from utils import (
    load_cleaned_data,
    get_dataset_stats,
    get_top_cuisines,
    get_top_cities
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title = "🍽️ Restaurant ML Project",
    page_icon  = "🍽️",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ============================================================
# HEADER
# ============================================================

st.title("🍽️ Restaurant ML Project")
st.markdown("### Cognifyz Technologies — ML Internship")
st.markdown("---")

st.markdown("""
This app demonstrates **4 Machine Learning tasks** built on a 
dataset of **9,551 restaurants** across multiple countries.

Navigate using the **sidebar** to explore each task.
""")

# ============================================================
# LOAD DATA
# ============================================================

df = load_cleaned_data()

if df is None:
    st.error("❌ Could not load dataset. Please check data folder.")
    st.stop()

# ============================================================
# DATASET STATISTICS — KEY METRICS
# ============================================================

st.markdown("## 📊 Dataset Overview")

stats = get_dataset_stats(df)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label = "🏪 Total Restaurants",
        value = f"{stats['total_restaurants']:,}"
    )
with col2:
    st.metric(
        label = "🌆 Total Cities",
        value = f"{stats['total_cities']:,}"
    )
with col3:
    st.metric(
        label = "🌍 Countries",
        value = f"{stats['total_countries']:,}"
    )
with col4:
    st.metric(
        label = "⭐ Avg Rating",
        value = f"{stats['avg_rating']}"
    )
with col5:
    st.metric(
        label = "🍜 Cuisine Types",
        value = f"{stats['total_cuisines']:,}"
    )

st.markdown("---")

# ============================================================
# QUICK VISUALIZATIONS
# ============================================================

st.markdown("## 📈 Quick Insights")

col1, col2 = st.columns(2)

# Plot 1 — Rating Distribution
with col1:
    st.markdown("### ⭐ Rating Distribution")
    df_rated = df[df['Aggregate rating'] > 0]
    fig, ax  = plt.subplots(figsize=(7, 4))
    ax.hist(
        df_rated['Aggregate rating'],
        bins      = 20,
        color     = '#FF6B6B',
        edgecolor = 'black',
        alpha     = 0.8
    )
    ax.set_xlabel('Aggregate Rating')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Restaurant Ratings')
    st.pyplot(fig)
    plt.close()

# Plot 2 — Top 10 Cuisines
with col2:
    st.markdown("### 🍜 Top 10 Cuisines")
    top_cuisines = (
        df['Cuisines']
        .dropna()
        .apply(lambda x: x.split(',')[0].strip())
        .value_counts()
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    top_cuisines.plot(
        kind      = 'barh',
        ax        = ax,
        color     = '#4ECDC4',
        edgecolor = 'black'
    )
    ax.set_xlabel('Number of Restaurants')
    ax.set_title('Top 10 Most Common Cuisines')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

col3, col4 = st.columns(2)

# Plot 3 — Top 10 Cities
with col3:
    st.markdown("### 🌆 Top 10 Cities")
    top_cities = df['City'].value_counts().head(10)
    fig, ax    = plt.subplots(figsize=(7, 4))
    top_cities.plot(
        kind      = 'barh',
        ax        = ax,
        color     = '#45B7D1',
        edgecolor = 'black'
    )
    ax.set_xlabel('Number of Restaurants')
    ax.set_title('Top 10 Cities by Restaurant Count')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Plot 4 — Price Range Distribution
with col4:
    st.markdown("### 💰 Price Range Distribution")
    price_counts = df['Price range'].value_counts().sort_index()
    labels = ['Cheap(1)', 'Moderate(2)',
              'Expensive(3)', 'Very Expensive(4)']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.pie(
        price_counts.values,
        labels    = labels,
        autopct   = '%1.1f%%',
        colors    = ['#2ECC71', '#F39C12',
                     '#E74C3C', '#9B59B6'],
        startangle= 90
    )
    ax.set_title('Price Range Distribution')
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ============================================================
# TASK OVERVIEW CARDS
# ============================================================

st.markdown("## 🎯 ML Tasks")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    ### ⭐ Task 1 — Rating Prediction
    Predict restaurant aggregate rating using
    Random Forest regression model.

    **Features:** Price range, Votes, Cuisine,
    City, Online Delivery and more.
    """)

    st.success("""
    ### 🍜 Task 3 — Cuisine Classification
    Classify restaurant cuisine type using
    XGBoost classification model.

    **Models:** Logistic Regression,
    Random Forest, XGBoost
    """)

with col2:
    st.warning("""
    ### 🔍 Task 2 — Recommendation System
    Get personalized restaurant recommendations
    using content-based filtering.

    **Based on:** Cuisine preference,
    Price range, City, Rating
    """)

    st.error("""
    ### 🗺️ Task 4 — Location Analysis
    Explore restaurant distribution on
    interactive maps and city statistics.

    **Includes:** Heatmap, Cluster map,
    Rating map, City statistics
    """)

st.markdown("---")
st.markdown(
    "Built by **Rahul Shewatkar** | "
    "Cognifyz Technologies ML Internship | "
    "[GitHub](https://github.com/rshewatkar)"
)