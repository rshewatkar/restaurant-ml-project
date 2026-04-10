# ============================================================
# utils.py
# Shared helper functions for Streamlit app
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
import sys

# Add src/ to path so we can import our modules
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'src')
)


# ============================================================
# SECTION 1 — DATA LOADING (Cached for performance)
# ============================================================

@st.cache_data
def load_cleaned_data():
    """
    Load cleaned restaurant dataset.
    Cached so it only loads once per session.
    """
    try:
        df = pd.read_csv('data/restaurant_cleaned.csv')
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None


@st.cache_data
def load_rated_data():
    """
    Load rated restaurants dataset.
    Cached so it only loads once per session.
    """
    try:
        df = pd.read_csv('data/restaurant_rated.csv')
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None


# ============================================================
# SECTION 2 — MODEL LOADING (Cached for performance)
# ============================================================

@st.cache_resource
def load_rating_model():
    """Load saved rating prediction model and scaler."""
    try:
        model  = joblib.load(
            'outputs/models/rating_model.pkl'
        )
        scaler = joblib.load(
            'outputs/models/rating_scaler.pkl'
        )
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading rating model: {e}")
        return None, None


@st.cache_resource
def load_cuisine_model():
    """Load saved cuisine classification model."""
    try:
        model  = joblib.load(
            'outputs/models/cuisine_classifier.pkl'
        )
        scaler = joblib.load(
            'outputs/models/cuisine_scaler.pkl'
        )
        le     = joblib.load(
            'outputs/models/cuisine_label_encoder.pkl'
        )
        return model, scaler, le
    except Exception as e:
        st.error(f"❌ Error loading cuisine model: {e}")
        return None, None, None


@st.cache_resource
def load_recommender_models():
    """Load saved TF-IDF vectorizer and scaler."""
    try:
        tfidf  = joblib.load(
            'outputs/models/tfidf_vectorizer.pkl'
        )
        scaler = joblib.load(
            'outputs/models/recommendation_scaler.pkl'
        )
        return tfidf, scaler
    except Exception as e:
        st.error(f"❌ Error loading recommender: {e}")
        return None, None


# ============================================================
# SECTION 3 — DATASET STATISTICS
# ============================================================

@st.cache_data
def get_dataset_stats(df):
    """
    Calculate key statistics about the dataset
    for display on home page.
    """
    stats = {
        'total_restaurants': len(df),
        'total_cities'     : df['City'].nunique(),
        'total_countries'  : df['Country Code'].nunique(),
        'avg_rating'       : round(
            df[df['Aggregate rating'] > 0]
            ['Aggregate rating'].mean(), 2
        ),
        'total_cuisines'   : df['Cuisines'].dropna().apply(
            lambda x: x.split(',')[0].strip()
        ).nunique()
    }
    return stats


# ============================================================
# SECTION 4 — HELPER FUNCTIONS
# ============================================================

def get_top_cities(df, top_n=30):
    """Return list of top N cities by restaurant count."""
    return df['City'].value_counts().head(top_n).index.tolist()


def get_top_cuisines(df, top_n=20):
    """Return list of top N primary cuisines."""
    return (
        df['Cuisines']
        .dropna()
        .apply(lambda x: x.split(',')[0].strip())
        .value_counts()
        .head(top_n)
        .index.tolist()
    )


def format_rating(rating):
    """Convert rating to star display string."""
    if rating >= 4.5: return f"⭐⭐⭐⭐⭐ {rating}"
    elif rating >= 4.0: return f"⭐⭐⭐⭐ {rating}"
    elif rating >= 3.0: return f"⭐⭐⭐ {rating}"
    elif rating >= 2.0: return f"⭐⭐ {rating}"
    else: return f"⭐ {rating}"


def get_price_label(price_range):
    """Convert price range number to label."""
    labels = {
        1: "💰 Cheap",
        2: "💰💰 Moderate",
        3: "💰💰💰 Expensive",
        4: "💰💰💰💰 Very Expensive"
    }
    return labels.get(price_range, "Unknown")