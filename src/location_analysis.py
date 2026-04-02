import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")


# ============================================================
# SECTION 1 — DATA VALIDATION
# ============================================================

def validate_coordinates(df):
    """
    Validate and filter restaurants with valid coordinates.

    Rules:
    - Remove (0, 0) coordinates
    - Keep latitude between -90 and 90
    - Keep longitude between -180 and 180

    Parameters:
    -----------
    df : pd.DataFrame → restaurant dataset

    Returns:
    --------
    df_valid : pd.DataFrame → restaurants with valid coordinates

    Example:
    --------
    df_valid = validate_coordinates(df)
    """
    original = len(df)

    df_valid = df[
        (df['Latitude']  != 0) &
        (df['Longitude'] != 0) &
        (df['Latitude'].between(-90, 90)) &
        (df['Longitude'].between(-180, 180))
    ].copy()

    removed = original - len(df_valid)

    print(f"Coordinates validated!")
    print(f"   Valid       : {len(df_valid)}")
    print(f"   Removed     : {removed}")
    print(f"   Lat range   : {df_valid['Latitude'].min():.2f}"
          f" to {df_valid['Latitude'].max():.2f}")
    print(f"   Lon range   : {df_valid['Longitude'].min():.2f}"
          f" to {df_valid['Longitude'].max():.2f}")

    return df_valid


# ============================================================
# SECTION 2 — MAP CREATION
# ============================================================

def create_cluster_map(df, save_path=None):
    """
    Create interactive clustered restaurant map using Folium.
    Clicking a marker shows restaurant details.

    Parameters:
    -----------
    df        : pd.DataFrame → dataset with valid coordinates
    save_path : str → path to save HTML map (optional)

    Returns:
    --------
    map_cluster : folium.Map → interactive map object

    Example:
    --------
    map1 = create_cluster_map(df,
        save_path='../outputs/maps/map_clustered.html')
    """
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()

    map_cluster = folium.Map(
        location  = [center_lat, center_lon],
        zoom_start= 5,
        tiles     = 'CartoDB positron'
    )

    marker_cluster = MarkerCluster().add_to(map_cluster)

    for _, row in df.iterrows():
        folium.Marker(
            location = [row['Latitude'], row['Longitude']],
            popup    = folium.Popup(
                f"""
                <b>{row['Restaurant Name']}</b><br>
                {row['City']}<br>
                {row['Cuisines']}<br>
                Rating: {row['Aggregate rating']}<br>
                Price Range: {row['Price range']}
                """,
                max_width=200
            ),
            tooltip=row['Restaurant Name']
        ).add_to(marker_cluster)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        map_cluster.save(save_path)
        print(f"Cluster map saved: {save_path}")

    return map_cluster


def create_heatmap(df, save_path=None):
    """
    Create restaurant density heatmap using Folium.
    Weighted by votes — popular restaurants appear brighter.

    Parameters:
    -----------
    df        : pd.DataFrame → dataset with valid coordinates
    save_path : str → path to save HTML map (optional)

    Returns:
    --------
    map_heat : folium.Map → interactive heatmap object

    Example:
    --------
    map2 = create_heatmap(df,
        save_path='../outputs/maps/map_heatmap.html')
    """
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()

    map_heat = folium.Map(
        location  = [center_lat, center_lon],
        zoom_start= 5,
        tiles     = 'CartoDB dark_matter'
    )

    heat_data = df[['Latitude', 'Longitude', 'Votes']
                   ].values.tolist()

    HeatMap(
        heat_data,
        min_opacity = 0.3,
        radius      = 15,
        blur        = 10,
        gradient    = {
            0.2: 'blue',
            0.4: 'lime',
            0.6: 'yellow',
            0.8: 'orange',
            1.0: 'red'
        }
    ).add_to(map_heat)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        map_heat.save(save_path)
        print(f"Heatmap saved: {save_path}")

    return map_heat


def get_rating_color(rating):
    """
    Return color string based on restaurant rating.

    Parameters:
    -----------
    rating : float → aggregate rating value

    Returns:
    --------
    color : str → color name for folium marker

    Example:
    --------
    color = get_rating_color(4.2)  # Returns 'green'
    """
    if rating >= 4.5:   return 'darkgreen'
    elif rating >= 4.0: return 'green'
    elif rating >= 3.5: return 'lightgreen'
    elif rating >= 3.0: return 'orange'
    elif rating > 0:    return 'red'
    else:               return 'gray'


def create_rating_map(df, sample_size=2000, save_path=None):
    """
    Create color-coded rating map using Folium.
    Each restaurant is a circle colored by its rating.

    Parameters:
    -----------
    df          : pd.DataFrame → dataset with valid coordinates
    sample_size : int → number of restaurants to plot
                        (default 2000 for performance)
    save_path   : str → path to save HTML map (optional)

    Returns:
    --------
    map_rating : folium.Map → interactive rating map

    Example:
    --------
    map3 = create_rating_map(df,
        save_path='../outputs/maps/map_ratings.html')
    """
    map_rating = folium.Map(
        location  = [20.5937, 78.9629],  # Center of India
        zoom_start= 5,
        tiles     = 'CartoDB positron'
    )

    # Sample for performance
    df_sample = df.sample(
        min(sample_size, len(df)),
        random_state=42
    )

    for _, row in df_sample.iterrows():
        folium.CircleMarker(
            location     = [row['Latitude'], row['Longitude']],
            radius       = 5,
            color        = get_rating_color(row['Aggregate rating']),
            fill         = True,
            fill_opacity = 0.7,
            popup        = folium.Popup(
                f"""
                <b>{row['Restaurant Name']}</b><br>
                Rating: {row['Aggregate rating']}<br>
                {row['Cuisines']}<br>
                {row['City']}
                """,
                max_width=200
            )
        ).add_to(map_rating)

    # Add legend
    legend_html = '''
    <div style="position:fixed; bottom:30px; left:30px;
         background:white; padding:10px; border-radius:5px;
         border:2px solid grey; font-size:12px;">
        <b>Rating Legend</b><br>
        <span style="color:darkgreen">●</span> 4.5+ Excellent<br>
        <span style="color:green">●</span> 4.0-4.5 Very Good<br>
        <span style="color:lightgreen">●</span> 3.5-4.0 Good<br>
        <span style="color:orange">●</span> 3.0-3.5 Average<br>
        <span style="color:red">●</span> Below 3.0 Poor<br>
        <span style="color:gray">●</span> Not Rated
    </div>
    '''
    map_rating.get_root().html.add_child(
        folium.Element(legend_html)
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        map_rating.save(save_path)
        print(f"Rating map saved: {save_path}")

    return map_rating


# ============================================================
# SECTION 3 — STATISTICAL ANALYSIS
# ============================================================

def get_city_statistics(df, top_n=15):
    """
    Calculate comprehensive statistics per city.

    Statistics:
    - Total restaurants
    - Average rating
    - Average cost for two
    - Average price range
    - Online delivery percentage
    - Table booking percentage

    Parameters:
    -----------
    df    : pd.DataFrame → restaurant dataset
    top_n : int → number of top cities (default 15)

    Returns:
    --------
    city_stats : pd.DataFrame → city statistics table

    Example:
    --------
    stats = get_city_statistics(df, top_n=15)
    """
    df_rated   = df[df['Aggregate rating'] > 0]
    top_cities = df['City'].value_counts().head(top_n).index

    city_stats = df_rated[
        df_rated['City'].isin(top_cities)
    ].groupby('City').agg(
        Total_Restaurants   = ('Restaurant Name', 'count'),
        Avg_Rating          = ('Aggregate rating', 'mean'),
        Avg_Cost_for_Two    = ('Average Cost for two', 'mean'),
        Avg_Price_Range     = ('Price range', 'mean'),
        Online_Delivery_Pct = ('Has Online delivery', 'mean'),
        Table_Booking_Pct   = ('Has Table booking', 'mean')
    ).round(2).sort_values(
        'Total_Restaurants', ascending=False
    )

    city_stats['Online_Delivery_Pct'] = (
        city_stats['Online_Delivery_Pct'] * 100
    ).round(1).astype(str) + '%'

    city_stats['Table_Booking_Pct'] = (
        city_stats['Table_Booking_Pct'] * 100
    ).round(1).astype(str) + '%'

    print("City statistics calculated!")
    print(city_stats.to_string())

    return city_stats


def get_top_cuisine_per_city(df, top_n_cities=10):
    """
    Find the most popular cuisine in each city.

    Parameters:
    -----------
    df           : pd.DataFrame → restaurant dataset
    top_n_cities : int → number of cities (default 10)

    Returns:
    --------
    result : pd.DataFrame → top cuisine per city

    Example:
    --------
    top_cuisines = get_top_cuisine_per_city(df)
    """
    df = df.copy()
    df['Primary Cuisine'] = df['Cuisines'].apply(
        lambda x: x.split(',')[0].strip()
        if pd.notnull(x) else 'Unknown'
    )

    top_cities = df['City'].value_counts().head(top_n_cities).index

    result = (
        df[df['City'].isin(top_cities)]
        .groupby(['City', 'Primary Cuisine'])
        .size()
        .reset_index(name='Count')
        .sort_values('Count', ascending=False)
        .groupby('City')
        .first()
        .reset_index()
        [['City', 'Primary Cuisine', 'Count']]
        .sort_values('Count', ascending=False)
    )

    print("Top cuisine per city:")
    print(result.to_string(index=False))

    return result


# ============================================================
# SECTION 4 — VISUALIZATION
# ============================================================

def plot_top_cities(df, top_n=15, save_path=None):
    """
    Bar chart of top cities by restaurant count.

    Parameters:
    -----------
    df        : pd.DataFrame → restaurant dataset
    top_n     : int → number of cities (default 15)
    save_path : str → path to save plot (optional)

    Example:
    --------
    plot_top_cities(df, top_n=15,
        save_path='../plots/task4_top_cities.png')
    """
    top_cities = df['City'].value_counts().head(top_n)

    plt.figure(figsize=(14, 6))
    bars = plt.bar(
        top_cities.index,
        top_cities.values,
        color     = 'steelblue',
        edgecolor = 'black'
    )

    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 10,
            str(int(bar.get_height())),
            ha='center', fontsize=9, fontweight='bold'
        )

    plt.title(f'Top {top_n} Cities by Number of Restaurants',
              fontsize=14, fontweight='bold')
    plt.xlabel('City')
    plt.ylabel('Number of Restaurants')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.show()


def plot_avg_rating_by_city(df, top_n=15, save_path=None):
    """
    Bar chart of average ratings by city with overall average line.

    Parameters:
    -----------
    df        : pd.DataFrame → restaurant dataset
    top_n     : int → number of cities (default 15)
    save_path : str → path to save plot (optional)

    Example:
    --------
    plot_avg_rating_by_city(df,
        save_path='../plots/task4_avg_rating_by_city.png')
    """
    df_rated   = df[df['Aggregate rating'] > 0]
    top_cities = df['City'].value_counts().head(top_n).index

    city_ratings = (
        df_rated[df_rated['City'].isin(top_cities)]
        .groupby('City')['Aggregate rating']
        .mean()
        .round(2)
        .sort_values(ascending=False)
    )

    overall_avg = df_rated['Aggregate rating'].mean()

    plt.figure(figsize=(14, 6))
    bars = plt.bar(
        city_ratings.index,
        city_ratings.values,
        color     = 'coral',
        edgecolor = 'black'
    )
    plt.axhline(
        y         = overall_avg,
        color     = 'navy',
        linestyle = '--',
        linewidth = 2,
        label     = f'Overall Avg: {overall_avg:.2f}'
    )

    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{bar.get_height():.2f}',
            ha='center', fontsize=9, fontweight='bold'
        )

    plt.title('Average Restaurant Rating by City',
              fontsize=14, fontweight='bold')
    plt.xlabel('City')
    plt.ylabel('Average Rating')
    plt.ylim(0, 5)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.show()


def plot_price_by_city(df, top_n=10, save_path=None):
    """
    Stacked bar chart of price range distribution by city.

    Parameters:
    -----------
    df        : pd.DataFrame → restaurant dataset
    top_n     : int → number of cities (default 10)
    save_path : str → path to save plot (optional)

    Example:
    --------
    plot_price_by_city(df,
        save_path='../plots/task4_price_by_city.png')
    """
    top_cities = df['City'].value_counts().head(top_n).index
    df_top     = df[df['City'].isin(top_cities)]

    price_city = df_top.groupby(
        ['City', 'Price range']
    ).size().unstack(fill_value=0)

    price_city.plot(
        kind      = 'bar',
        stacked   = True,
        figsize   = (14, 6),
        colormap  = 'RdYlGn',
        edgecolor = 'black'
    )
    plt.title('Price Range Distribution by City',
              fontsize=14, fontweight='bold')
    plt.xlabel('City')
    plt.ylabel('Number of Restaurants')
    plt.legend(
        title  = 'Price Range',
        labels = ['Cheap(1)', 'Moderate(2)',
                  'Expensive(3)', 'Very Expensive(4)'],
        bbox_to_anchor=(1.05, 1)
    )
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.show()


def plot_cuisine_by_city(df, top_n=10, save_path=None):
    """
    Bar chart of most popular cuisine per city.

    Parameters:
    -----------
    df        : pd.DataFrame → restaurant dataset
    top_n     : int → number of cities (default 10)
    save_path : str → path to save plot (optional)

    Example:
    --------
    plot_cuisine_by_city(df,
        save_path='../plots/task4_cuisine_by_city.png')
    """
    top_cuisine_by_city = get_top_cuisine_per_city(df, top_n)

    plt.figure(figsize=(14, 6))
    bars = plt.bar(
        top_cuisine_by_city['City'],
        top_cuisine_by_city['Count'],
        color     = 'mediumpurple',
        edgecolor = 'black'
    )

    for bar, cuisine in zip(
        bars, top_cuisine_by_city['Primary Cuisine']
    ):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5,
            cuisine,
            ha='center', fontsize=8,
            fontweight='bold', rotation=15
        )

    plt.title('Most Popular Cuisine in Top Cities',
              fontsize=14, fontweight='bold')
    plt.xlabel('City')
    plt.ylabel('Number of Restaurants')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.show()


# ============================================================
# MAIN — Run directly if needed
# ============================================================

if __name__ == "__main__":

    # Load data
    df = pd.read_csv('../data/restaurant_cleaned.csv')
    print(f"Loaded: {df.shape}")

    # Validate coordinates
    df_valid = validate_coordinates(df)

    # Create all maps
    create_cluster_map(
        df_valid,
        save_path='../outputs/maps/map_clustered.html'
    )
    create_heatmap(
        df_valid,
        save_path='../outputs/maps/map_heatmap.html'
    )
    create_rating_map(
        df_valid,
        save_path='../outputs/maps/map_ratings.html'
    )

    # Statistical analysis
    city_stats = get_city_statistics(df, top_n=15)
    city_stats.to_csv(
        '../outputs/reports/task4_city_statistics.csv'
    )

    # Plots
    plot_top_cities(
        df, save_path='../plots/task4_top_cities.png'
    )
    plot_avg_rating_by_city(
        df, save_path='../plots/task4_avg_rating_by_city.png'
    )
    plot_price_by_city(
        df, save_path='../plots/task4_price_by_city.png'
    )
    plot_cuisine_by_city(
        df, save_path='../plots/task4_cuisine_by_city.png'
    )

    print("\nLocation analysis complete!")