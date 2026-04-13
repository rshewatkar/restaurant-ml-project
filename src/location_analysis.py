import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster

# ✅ IMPORTANT FIX
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")


# ============================================================
# SECTION 1 — DATA VALIDATION
# ============================================================

def validate_coordinates(df):
    df = df.copy()
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    df_valid = df[
        df['Latitude'].notna() &
        df['Longitude'].notna() &
        (df['Latitude'] != 0) &
        (df['Longitude'] != 0) &
        (df['Latitude'].between(-90, 90)) &
        (df['Longitude'].between(-180, 180))
    ].copy()

    print(f"Valid coordinates: {len(df_valid)} / {len(df)}")
    return df_valid

# ============================================================
# SECTION 2 — MAP CREATION
# ============================================================

def create_cluster_map(df, save_path=None):
    if df.empty:
        print("Skipping cluster map: no valid coordinates found.")
        return None

    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()

    map_cluster = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='CartoDB positron'
    )

    marker_cluster = MarkerCluster().add_to(map_cluster)

    for _, row in df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"{row['Restaurant Name']} ({row['City']})",
        ).add_to(marker_cluster)

    if save_path:
        map_cluster.save(save_path)

    return map_cluster


def create_heatmap(df, save_path=None):
    if df.empty:
        print("Skipping heatmap: no valid coordinates found.")
        return None

    df = df.copy()
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce').fillna(0)

    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()

    map_heat = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='CartoDB dark_matter'
    )

    heat_data = df[['Latitude', 'Longitude', 'Votes']].values.tolist()
    HeatMap(heat_data).add_to(map_heat)

    if save_path:
        map_heat.save(save_path)

    return map_heat


def create_rating_map(df, save_path=None):
    if df.empty:
        print("Skipping rating map: no valid coordinates found.")
        return None

    df = df.copy()
    df['Aggregate rating'] = pd.to_numeric(df['Aggregate rating'], errors='coerce').fillna(0)

    map_rating = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5
    )

    sample_size = min(2000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42) if sample_size > 0 else df.iloc[0:0]

    for _, row in df_sample.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color='green' if row['Aggregate rating'] >= 4 else 'red',
            fill=True,
            fill_opacity=0.7
        ).add_to(map_rating)

    if save_path:
        map_rating.save(save_path)

    return map_rating


# ============================================================
# SECTION 3 — STATS
# ============================================================

def get_city_statistics(df, top_n=15):
    df = df.copy()
    df['Aggregate rating'] = pd.to_numeric(df['Aggregate rating'], errors='coerce').fillna(0)

    df_rated = df[df['Aggregate rating'] > 0]

    stats = df_rated.groupby('City').agg(
        Total=('Restaurant Name', 'count'),
        Avg_Rating=('Aggregate rating', 'mean')
    ).sort_values('Total', ascending=False).head(top_n)

    return stats


# ============================================================
# SECTION 4 — MAIN
# ============================================================

if __name__ == "__main__":

    # ✅ Load data safely
    data_path = os.path.join(BASE_DIR, 'data', 'restaurant_cleaned.csv')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ File not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded: {df.shape}")

    # ✅ Create directories
    os.makedirs(os.path.join(BASE_DIR, 'outputs', 'maps'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'outputs', 'reports'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'plots'), exist_ok=True)

    # ✅ Validate
    df_valid = validate_coordinates(df)

    # ✅ Paths
    cluster_map_path = os.path.join(BASE_DIR, 'outputs', 'maps', 'map_clustered.html')
    heatmap_path     = os.path.join(BASE_DIR, 'outputs', 'maps', 'map_heatmap.html')
    rating_map_path  = os.path.join(BASE_DIR, 'outputs', 'maps', 'map_ratings.html')

    # ✅ Create maps
    create_cluster_map(df_valid, cluster_map_path)
    create_heatmap(df_valid, heatmap_path)
    create_rating_map(df_valid, rating_map_path)

    # ✅ Stats
    stats = get_city_statistics(df_valid)
    stats.to_csv(os.path.join(BASE_DIR, 'outputs', 'reports', 'city_stats.csv'))

    print("\n✅ Location analysis complete!")