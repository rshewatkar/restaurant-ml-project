import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ML
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# SECTION 1 — BUILD FEATURE MATRIX
# ============================================================

def build_feature_matrix(df):
    """
    Build combined feature matrix for cosine similarity.

    Components:
    1. TF-IDF on Cuisines text (weighted 2x)
    2. Normalized numerical features
    3. One-hot encoded city

    Parameters:
    -----------
    df : pd.DataFrame → cleaned restaurant dataset

    Returns:
    --------
    feature_matrix : np.array  → combined feature matrix
    tfidf          : TfidfVectorizer → fitted vectorizer
    scaler         : MinMaxScaler   → fitted scaler
    city_cols      : list           → city column names
    df             : pd.DataFrame   → updated dataframe

    Example:
    --------
    matrix, tfidf, scaler, city_cols, df = build_feature_matrix(df)
    """
    df = df.copy()

    # Fill missing cuisines
    df['Cuisines'] = df['Cuisines'].fillna('Unknown')

    # Extract primary cuisine
    df['Primary Cuisine'] = df['Cuisines'].apply(
        lambda x: x.split(',')[0].strip()
    )

    # Step 1: TF-IDF on cuisine text
    tfidf = TfidfVectorizer(
        max_features = 50,
        stop_words   = 'english'
    )
    cuisine_matrix = tfidf.fit_transform(
        df['Cuisines']
    ).toarray() * 2  # Weight cuisine 2x

    # Step 2: Normalize numerical features
    scaler = MinMaxScaler()
    numerical_matrix = scaler.fit_transform(
        df[[
            'Price range',
            'Aggregate rating',
            'Votes',
            'Has Table booking',
            'Has Online delivery'
        ]]
    )

    # Step 3: One-hot encode city
    city_dummies = pd.get_dummies(df['City'], prefix='city')
    city_cols    = city_dummies.columns.tolist()
    city_matrix  = city_dummies.values

    # Step 4: Combine all features
    feature_matrix = np.hstack([
        cuisine_matrix,
        numerical_matrix,
        city_matrix
    ])

    print(f"Feature matrix built!")
    print(f"Cuisine features    : {cuisine_matrix.shape[1]}")
    print(f"Numerical features  : {numerical_matrix.shape[1]}")
    print(f"City features       : {city_matrix.shape[1]}")
    print(f"Final matrix shape  : {feature_matrix.shape}")

    return feature_matrix, tfidf, scaler, city_cols, df


# ============================================================
# SECTION 2 — RECOMMENDATION FUNCTION
# ============================================================

def recommend_restaurants(
    df,
    feature_matrix,
    tfidf,
    scaler,
    city_cols,
    cuisine_preference,
    price_range,
    city,
    min_rating     = 3.0,
    online_delivery= None,
    table_booking  = None,
    top_n          = 5
):
    """
    Recommend restaurants based on user preferences
    using content-based filtering with cosine similarity.

    Parameters:
    -----------
    df                 : pd.DataFrame → restaurant dataset
    feature_matrix     : np.array → combined feature matrix
    tfidf              : TfidfVectorizer → fitted vectorizer
    scaler             : MinMaxScaler → fitted scaler
    city_cols          : list → city column names
    cuisine_preference : str  → e.g., "North Indian"
    price_range        : int  → 1 (cheap) to 4 (expensive)
    city               : str  → e.g., "New Delhi"
    min_rating         : float→ minimum rating (default 3.0)
    online_delivery    : int  → 1=Yes, 0=No, None=No preference
    table_booking      : int  → 1=Yes, 0=No, None=No preference
    top_n              : int  → recommendations count (default 5)

    Returns:
    --------
    recommendations : pd.DataFrame → top N restaurants

    Example:
    --------
    recs = recommend_restaurants(
        df, feature_matrix, tfidf, scaler, city_cols,
        cuisine_preference = "North Indian",
        price_range        = 2,
        city               = "New Delhi",
        min_rating         = 3.5,
        top_n              = 5
    )
    """
    # Step 1: Filter by city and rating
    filtered_df = df[
        (df['City'].str.lower() == city.lower()) &
        (df['Aggregate rating'] >= min_rating)
    ].copy()

    # Apply optional filters
    if online_delivery is not None:
        filtered_df = filtered_df[
            filtered_df['Has Online delivery'] == online_delivery
        ]
    if table_booking is not None:
        filtered_df = filtered_df[
            filtered_df['Has Table booking'] == table_booking
        ]

    # Relax city filter if no results
    if len(filtered_df) == 0:
        print(f"No restaurants in {city}. Showing top matches...")
        filtered_df = df[
            df['Aggregate rating'] >= min_rating
        ].copy()

    # Step 2: Build user preference vector
    user_cuisine = tfidf.transform(
        [cuisine_preference]
    ).toarray() * 2

    user_numerical = scaler.transform([[
        price_range,
        min_rating,
        filtered_df['Votes'].median(),
        1 if table_booking == 1 else 0,
        1 if online_delivery == 1 else 0
    ]])

    # One-hot encode user city
    user_city = np.zeros(len(city_cols))
    city_col  = f'city_{city}'
    if city_col in city_cols:
        user_city[city_cols.index(city_col)] = 1

    user_vector = np.hstack([
        user_cuisine,
        user_numerical,
        user_city.reshape(1, -1)
    ])

    # Step 3: Calculate cosine similarity
    filtered_indices = filtered_df.index.tolist()
    filtered_matrix  = feature_matrix[filtered_indices]
    similarities     = cosine_similarity(
        user_vector, filtered_matrix
    )[0]

    # Step 4: Get top N restaurants
    top_indices      = similarities.argsort()[::-1][:top_n]
    recommendations  = filtered_df.iloc[top_indices].copy()
    recommendations['Similarity Score'] = (
        similarities[top_indices].round(4)
    )

    # Clean output
    result = recommendations[[
        'Restaurant Name',
        'Primary Cuisine',
        'City',
        'Price range',
        'Aggregate rating',
        'Has Online delivery',
        'Has Table booking',
        'Votes',
        'Similarity Score'
    ]].reset_index(drop=True)

    result.index = result.index + 1
    return result


# ============================================================
# SECTION 3 — EVALUATION
# ============================================================

def evaluate_recommendations(recs, cuisine_pref,
                              price_pref, min_rating):
    """
    Evaluate quality of recommendations against preferences.

    Checks:
    - How many recommendations match cuisine preference
    - How many match price range (within ±1)
    - Average rating of recommendations
    - Average similarity score

    Parameters:
    -----------
    recs         : pd.DataFrame → recommendations dataframe
    cuisine_pref : str   → user's cuisine preference
    price_pref   : int   → user's price range preference
    min_rating   : float → user's minimum rating preference

    Returns:
    --------
    evaluation : dict → quality metrics

    Example:
    --------
    quality = evaluate_recommendations(
        recs, "North Indian", 2, 3.0
    )
    """
    if len(recs) == 0:
        return {"error": "No recommendations found"}

    cuisine_match = recs['Primary Cuisine'].str.contains(
        cuisine_pref, case=False, na=False
    ).sum()

    price_match = (
        abs(recs['Price range'] - price_pref) <= 1
    ).sum()

    return {
        'Total Recommendations': len(recs),
        'Cuisine Match'        : f"{cuisine_match}/{len(recs)}",
        'Price Range Match'    : f"{price_match}/{len(recs)}",
        'Avg Rating'           : round(
            recs['Aggregate rating'].mean(), 2),
        'Avg Similarity Score' : round(
            recs['Similarity Score'].mean(), 4)
    }


# ============================================================
# SECTION 4 — VISUALIZATION
# ============================================================

def plot_recommendations(all_user_recs, save_path=None):
    """
    Plot ratings of recommended restaurants for
    multiple user profiles.

    Parameters:
    -----------
    all_user_recs : list of tuples → [(recs_df, title), ...]
    save_path     : str → path to save plot (optional)

    Example:
    --------
    plot_recommendations([
        (user1_recs, "User 1: North Indian, Delhi"),
        (user2_recs, "User 2: Chinese, Mumbai")
    ], save_path='../plots/task2_recommendations.png')
    """
    n_users = len(all_user_recs)
    cols    = 2
    rows    = (n_users + 1) // 2

    fig, axes = plt.subplots(rows, cols,
                             figsize=(14, 5 * rows))
    axes = axes.flatten()

    for i, (recs, title) in enumerate(all_user_recs):
        if len(recs) > 0:
            bars = axes[i].barh(
                recs['Restaurant Name'].str[:25],
                recs['Aggregate rating'],
                color     = 'steelblue',
                edgecolor = 'black'
            )
            axes[i].set_title(title,
                              fontsize=10,
                              fontweight='bold')
            axes[i].set_xlabel('Rating')
            axes[i].set_xlim(0, 5)

            for bar in bars:
                axes[i].text(
                    bar.get_width() + 0.05,
                    bar.get_y() + bar.get_height()/2,
                    f'{bar.get_width():.1f}',
                    va='center', fontsize=9
                )

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        'Recommended Restaurants — User Profiles',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Plot saved: {save_path}")
    plt.show()
# ============================================================
# SECTION 5 — SAVE & LOAD
# ============================================================

def save_recommender(tfidf, scaler,
                     save_dir='../outputs/models/'):
    """
    Save TF-IDF vectorizer and scaler.

    Parameters:
    -----------
    tfidf    : TfidfVectorizer → fitted vectorizer
    scaler   : MinMaxScaler   → fitted scaler
    save_dir : str → directory to save files

    Example:
    --------
    save_recommender(tfidf, scaler)
    """
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(
        tfidf,
        os.path.join(save_dir, 'tfidf_vectorizer.pkl')
    )
    joblib.dump(
        scaler,
        os.path.join(save_dir, 'recommendation_scaler.pkl')
    )

    print(f" TF-IDF saved : {save_dir}tfidf_vectorizer.pkl")
    print(f"aler saved : {save_dir}recommendation_scaler.pkl")
    
def load_recommender(tfidf_path, scaler_path):
    """
    Load saved TF-IDF vectorizer and scaler.

    Parameters:
    -----------
    tfidf_path  : str → path to saved vectorizer
    scaler_path : str → path to saved scaler

    Returns:
    --------
    tfidf  : loaded TfidfVectorizer
    scaler : loaded MinMaxScaler

    Example:
    --------
    tfidf, scaler = load_recommender(
        '../outputs/models/tfidf_vectorizer.pkl',
        '../outputs/models/recommendation_scaler.pkl'
    )
    """
    tfidf  = joblib.load(tfidf_path)
    scaler = joblib.load(scaler_path)
    print("TF-IDF and scaler loaded!")
    return tfidf, scaler

# ============================================================
# MAIN — Run directly if needed
# ============================================================

if __name__ == "__main__":

    # Load data
    df = pd.read_csv('../data/restaurant_cleaned.csv')
    print(f"Loaded: {df.shape}")

    # Build feature matrix
    (feature_matrix,
     tfidf, scaler,
     city_cols, df) = build_feature_matrix(df)

    # Test recommendations
    print("\n User 1: North Indian, New Delhi, Budget")
    recs1 = recommend_restaurants(
        df, feature_matrix, tfidf, scaler, city_cols,
        cuisine_preference = "North Indian",
        price_range        = 1,
        city               = "New Delhi",
        min_rating         = 3.0,
        online_delivery    = 1,
        top_n              = 5
    )
    print(recs1.to_string())

    # Evaluate
    quality = evaluate_recommendations(
        recs1, "North Indian", 1, 3.0
    )
    print("\nQuality Metrics:")
    for k, v in quality.items():
        print(f"  {k:30s}: {v}")

    # Plot
    plot_recommendations(
        [(recs1, "User 1: North Indian, Delhi")],
        save_path='../plots/task2_recommendations.png'
    )

    # Save
    save_recommender(tfidf, scaler)
    print("All done!")