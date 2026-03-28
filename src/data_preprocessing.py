import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# SECTION 1 — DATA LOADING
# ============================================================

def load_data(filepath):
    """
    Load raw restaurant dataset from CSV file.

    Parameters:
    -----------
    filepath : str → path to the CSV file

    Returns:
    --------
    df : pd.DataFrame → raw dataframe

    Example:
    --------
    df = load_data('../data/restaurant_dataset.csv')
    """
    try:
        df = pd.read_csv(filepath, encoding='latin-1')
        print(f"✅ Data loaded successfully!")
        print(f"   Shape    : {df.shape}")
        print(f"   Rows     : {df.shape[0]}")
        print(f"   Columns  : {df.shape[1]}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


# ============================================================
# SECTION 2 — MISSING VALUE HANDLING
# ============================================================

def check_missing_values(df):
    """
    Check and display missing values in the dataset.

    Parameters:
    -----------
    df : pd.DataFrame → input dataframe

    Returns:
    --------
    missing_df : pd.DataFrame → summary of missing values

    Example:
    --------
    missing_summary = check_missing_values(df)
    """
    missing       = df.isnull().sum()
    missing_pct   = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        'Missing Count'  : missing,
        'Missing Percent': missing_pct.round(2)
    }).sort_values('Missing Count', ascending=False)

    # Only show columns with missing values
    missing_df = missing_df[missing_df['Missing Count'] > 0]

    if len(missing_df) == 0:
        print("✅ No missing values found!")
    else:
        print("⚠️ Missing Values Found:")
        print(missing_df.to_string())

    return missing_df


def handle_missing_values(df):
    """
    Handle missing values in the dataset.

    Strategy:
    - Cuisines : drop rows (only 9 missing — safe to drop)
    - Others   : no missing values in this dataset

    Parameters:
    -----------
    df : pd.DataFrame → input dataframe

    Returns:
    --------
    df : pd.DataFrame → cleaned dataframe

    Example:
    --------
    df_clean = handle_missing_values(df)
    """
    original_shape = df.shape[0]

    # Drop rows where Cuisines is missing
    df = df.dropna(subset=['Cuisines'])

    dropped = original_shape - df.shape[0]
    print(f"✅ Missing values handled!")
    print(f"   Rows dropped : {dropped}")
    print(f"   Rows remaining : {df.shape[0]}")

    return df


# ============================================================
# SECTION 3 — COLUMN CLEANING
# ============================================================

def drop_irrelevant_columns(df):
    """
    Drop columns that are not useful for ML tasks.

    Columns dropped and reasons:
    - Restaurant ID     : unique identifier, not useful
    - Address           : too specific, free text
    - Locality Verbose  : duplicate of Locality
    - Rating color      : derived from Aggregate rating (leakage)
    - Rating text       : derived from Aggregate rating (leakage)
    - Switch to order menu : almost all same value
    - Currency          : redundant with Country Code

    Parameters:
    -----------
    df : pd.DataFrame → input dataframe

    Returns:
    --------
    df : pd.DataFrame → dataframe with irrelevant columns dropped

    Example:
    --------
    df_clean = drop_irrelevant_columns(df)
    """
    cols_to_drop = [
        'Restaurant ID',        # ID column
        'Address',              # Free text
        'Locality Verbose',     # Duplicate
        'Rating color',         # Data leakage
        'Rating text',          # Data leakage
        'Switch to order menu', # No variance
        'Currency'              # Redundant
    ]

    # Only drop columns that exist in dataframe
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    print(f"✅ Irrelevant columns dropped!")
    print(f"   Dropped  : {cols_to_drop}")
    print(f"   Remaining columns : {df.shape[1]}")

    return df


# ============================================================
# SECTION 4 — ENCODING
# ============================================================

def encode_binary_columns(df):
    """
    Encode Yes/No columns to 1/0.

    Columns encoded:
    - Has Table booking    : Yes→1, No→0
    - Has Online delivery  : Yes→1, No→0
    - Is delivering now    : Yes→1, No→0

    Parameters:
    -----------
    df : pd.DataFrame → input dataframe

    Returns:
    --------
    df : pd.DataFrame → dataframe with encoded binary columns

    Example:
    --------
    df_encoded = encode_binary_columns(df)
    """
    binary_cols = [
        'Has Table booking',
        'Has Online delivery',
        'Is delivering now'
    ]

    # Only encode columns that exist
    binary_cols = [c for c in binary_cols if c in df.columns]

    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    print(f"✅ Binary columns encoded: {binary_cols}")
    return df


def extract_primary_cuisine(df):
    """
    Extract primary cuisine from the Cuisines column.
    
    Example:
    "North Indian, Chinese, Italian" → "North Indian"

    Parameters:
    -----------
    df : pd.DataFrame → input dataframe

    Returns:
    --------
    df : pd.DataFrame → dataframe with new Primary Cuisine column

    Example:
    --------
    df = extract_primary_cuisine(df)
    """
    df['Primary Cuisine'] = df['Cuisines'].apply(
        lambda x: x.split(',')[0].strip() 
        if pd.notnull(x) else 'Unknown'
    )

    print(f"✅ Primary Cuisine extracted!")
    print(f"   Unique cuisines: {df['Primary Cuisine'].nunique()}")
    return df


def group_top_categories(df, column, top_n=20, other_label='Other'):
    """
    Keep top N categories in a column, group rest as 'Other'.
    Prevents too many dummy variables in ML models.

    Parameters:
    -----------
    df          : pd.DataFrame → input dataframe
    column      : str → column name to group
    top_n       : int → number of top categories to keep (default 20)
    other_label : str → label for grouped categories (default 'Other')

    Returns:
    --------
    df : pd.DataFrame → dataframe with grouped column

    Example:
    --------
    df = group_top_categories(df, 'Primary Cuisine', top_n=10)
    df = group_top_categories(df, 'City', top_n=20)
    """
    top_categories = df[column].value_counts().head(top_n).index
    df[column] = df[column].apply(
        lambda x: x if x in top_categories else other_label
    )

    print(f"✅ {column} grouped!")
    print(f"   Categories kept : {top_n}")
    print(f"   Unique values   : {df[column].nunique()}")
    return df


def encode_categorical_columns(df, columns):
    """
    Label encode categorical columns for ML models.

    Parameters:
    -----------
    df      : pd.DataFrame → input dataframe
    columns : list → list of column names to encode

    Returns:
    --------
    df       : pd.DataFrame → dataframe with encoded columns
    encoders : dict → dictionary of fitted LabelEncoders
                      (save these to decode predictions later)

    Example:
    --------
    df, encoders = encode_categorical_columns(
        df, ['Primary Cuisine', 'City']
    )
    """
    encoders = {}

    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"✅ Encoded: {col} → {df[col].nunique()} categories")
        else:
            print(f"⚠️ Column not found: {col}")

    return df, encoders


# ============================================================
# SECTION 5 — OUTLIER HANDLING
# ============================================================

def handle_outliers(df, column, method='iqr'):
    """
    Handle outliers in a numerical column.

    Methods:
    - 'iqr'  : Remove values beyond 1.5*IQR (default)
    - 'clip' : Clip values to 1st and 99th percentile

    Parameters:
    -----------
    df     : pd.DataFrame → input dataframe
    column : str → column name to handle outliers
    method : str → 'iqr' or 'clip' (default 'iqr')

    Returns:
    --------
    df : pd.DataFrame → dataframe with outliers handled

    Example:
    --------
    df = handle_outliers(df, 'Average Cost for two', method='clip')
    """
    original_count = len(df)

    if method == 'iqr':
        Q1  = df[column].quantile(0.25)
        Q3  = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower) & (df[column] <= upper)]
        removed = original_count - len(df)
        print(f"✅ Outliers removed from {column}: {removed} rows")

    elif method == 'clip':
        lower = df[column].quantile(0.01)
        upper = df[column].quantile(0.99)
        df[column] = df[column].clip(lower, upper)
        print(f"✅ {column} clipped to [{lower:.2f}, {upper:.2f}]")

    return df


# ============================================================
# SECTION 6 — SPLITTING
# ============================================================

def filter_rated_restaurants(df):
    """
    Filter out unrated restaurants (rating == 0).
    Unrated restaurants are not useful for ML tasks.

    Parameters:
    -----------
    df : pd.DataFrame → input dataframe

    Returns:
    --------
    df_rated : pd.DataFrame → only rated restaurants

    Example:
    --------
    df_rated = filter_rated_restaurants(df)
    """
    df_rated  = df[df['Aggregate rating'] > 0].copy()
    unrated   = len(df) - len(df_rated)

    print(f"✅ Filtered rated restaurants!")
    print(f"   Rated     : {len(df_rated)}")
    print(f"   Unrated   : {unrated} (excluded)")

    return df_rated


# ============================================================
# SECTION 7 — FULL PIPELINE
# ============================================================

def full_preprocessing_pipeline(filepath):
    """
    Run the complete preprocessing pipeline in one function.

    Steps:
    1. Load data
    2. Handle missing values
    3. Drop irrelevant columns
    4. Encode binary columns
    5. Extract primary cuisine
    6. Handle outliers
    7. Filter rated restaurants
    8. Save cleaned datasets

    Parameters:
    -----------
    filepath : str → path to raw dataset

    Returns:
    --------
    df_clean : pd.DataFrame → full cleaned dataset
    df_rated : pd.DataFrame → only rated restaurants

    Example:
    --------
    df_clean, df_rated = full_preprocessing_pipeline(
        '../data/restaurant_dataset.csv'
    )
    """
    print("=" * 55)
    print("STARTING FULL PREPROCESSING PIPELINE")
    print("=" * 55)

    # Step 1: Load
    print("\n📌 Step 1: Loading data...")
    df = load_data(filepath)
    if df is None:
        return None, None

    # Step 2: Missing values
    print("\n📌 Step 2: Handling missing values...")
    df = handle_missing_values(df)

    # Step 3: Drop columns
    print("\n📌 Step 3: Dropping irrelevant columns...")
    df = drop_irrelevant_columns(df)

    # Step 4: Encode binary
    print("\n📌 Step 4: Encoding binary columns...")
    df = encode_binary_columns(df)

    # Step 5: Extract primary cuisine
    print("\n📌 Step 5: Extracting primary cuisine...")
    df = extract_primary_cuisine(df)

    # Step 6: Handle outliers in cost
    print("\n📌 Step 6: Handling outliers...")
    df = handle_outliers(df, 'Average Cost for two', method='clip')

    # Step 7: Filter rated restaurants
    print("\n📌 Step 7: Filtering rated restaurants...")
    df_rated = filter_rated_restaurants(df)
    df_clean = df.copy()

    # Step 8: Save
    print("\n📌 Step 8: Saving cleaned datasets...")
    df_clean.to_csv('../data/restaurant_cleaned.csv', index=False)
    df_rated.to_csv('../data/restaurant_rated.csv',   index=False)
    print("✅ Saved: restaurant_cleaned.csv")
    print("✅ Saved: restaurant_rated.csv")

    print("\n" + "=" * 55)
    print("✅ PIPELINE COMPLETE!")
    print(f"   Clean dataset : {df_clean.shape}")
    print(f"   Rated dataset : {df_rated.shape}")
    print("=" * 55)

    return df_clean, df_rated


# ============================================================
# MAIN — Run pipeline directly if needed
# ============================================================

if __name__ == "__main__":
    df_clean, df_rated = full_preprocessing_pipeline(
        '../data/restaurant_dataset.csv'
    )