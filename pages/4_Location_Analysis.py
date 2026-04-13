# ============================================================
# Location Analysis Page
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import load_cleaned_data

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')
)
from location_analysis import (
    validate_coordinates,
    create_cluster_map,
    create_heatmap,
    create_rating_map,
    get_city_statistics,
    plot_top_cities,
    plot_avg_rating_by_city,
    plot_price_by_city
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title = "🗺️ Location Analysis",
    page_icon  = "🗺️",
    layout     = "wide"
)

st.title("🗺️ Location-Based Analysis")
st.markdown(
    "Explore the **geographical distribution** "
    "of restaurants on interactive maps."
)
st.markdown("---")

# ============================================================
# LOAD DATA
# ============================================================

df = load_cleaned_data()

if df is None:
    st.error("❌ Could not load data.")
    st.stop()

df_valid = validate_coordinates(df)

# ============================================================
# MAP SELECTION
# ============================================================

st.markdown("### 🗺️ Interactive Maps")

map_type = st.radio(
    "Select Map Type:",
    options     = [
        "📍 Clustered Restaurant Map",
        "🔥 Density Heatmap",
        "⭐ Rating Color Map"
    ],
    horizontal  = True
)

st.markdown("---")

# Render selected map
with st.spinner("🔄 Loading map..."):
    if map_type == "📍 Clustered Restaurant Map":
        st.markdown(
            "**Click on clusters to zoom in. "
            "Click markers for restaurant details.**"
        )
        m = create_cluster_map(df_valid)
        st_folium(m, width=1200, height=500)

    elif map_type == "🔥 Density Heatmap":
        st.markdown(
            "**Brighter areas = higher restaurant density "
            "weighted by popularity.**"
        )
        m = create_heatmap(df_valid)
        st_folium(m, width=1200, height=500)

    else:
        st.markdown("""
        **Color Guide:**
        🟢 4.5+ Excellent | 🟩 4.0+ Very Good |
        🟡 3.5+ Good | 🟠 3.0+ Average | 🔴 Below 3.0
        """)
        m = create_rating_map(df_valid)
        st_folium(m, width=1200, height=500)

st.markdown("---")

# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

st.markdown("### 📊 City-wise Statistics")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏙️ Restaurant Count",
    "⭐ Avg Ratings",
    "💰 Price Ranges",
    "📋 Full Statistics"
])

with tab1:
    fig, ax = plt.subplots(figsize=(12, 5))
    top_cities = df['City'].value_counts().head(15)
    top_cities.plot(kind='bar', ax=ax,
                    color='steelblue', edgecolor='black')
    ax.set_title('Top 15 Cities by Restaurant Count',
                 fontweight='bold')
    ax.set_xlabel('City')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab2:
    df_rated   = df[df['Aggregate rating'] > 0]
    top_cities = df['City'].value_counts().head(15).index
    city_avg   = (
        df_rated[df_rated['City'].isin(top_cities)]
        .groupby('City')['Aggregate rating']
        .mean().round(2)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    city_avg.plot(kind='bar', ax=ax,
                  color='coral', edgecolor='black')
    ax.axhline(
        y         = df_rated['Aggregate rating'].mean(),
        color     = 'navy',
        linestyle = '--',
        linewidth = 2,
        label     = 'Overall Average'
    )
    ax.set_title('Average Rating by City', fontweight='bold')
    ax.set_ylabel('Average Rating')
    ax.set_ylim(0, 5)
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab3:
    top_10 = df['City'].value_counts().head(10).index
    price_city = (
        df[df['City'].isin(top_10)]
        .groupby(['City', 'Price range'])
        .size().unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    price_city.plot(
        kind='bar', stacked=True, ax=ax,
        colormap='RdYlGn', edgecolor='black'
    )
    ax.set_title('Price Range Distribution by City',
                 fontweight='bold')
    ax.set_xlabel('City')
    ax.set_ylabel('Number of Restaurants')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab4:
    st.markdown("#### 📋 Complete City Statistics Table")
    city_stats = get_city_statistics(df, top_n=15)
    st.dataframe(
        city_stats,
        use_container_width=True
    )
    st.download_button(
        label     = "📥 Download City Statistics",
        data      = city_stats.to_csv(),
        file_name = "city_statistics.csv",
        mime      = "text/csv"
    )