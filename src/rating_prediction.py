import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# SECTION 1 — FEATURE PREPARATION
# ============================================================

def prepare_features(df):
    """
    Prepare feature matrix and target variable
    for rating prediction.

    Parameters:
    -----------
    df : pd.DataFrame → cleaned rated restaurant dataset

    Returns:
    --------
    X        : pd.DataFrame → feature matrix
    y        : pd.Series    → target variable
    encoders : dict         → fitted label encoders

    Example:
    --------
    X, y, encoders = prepare_features(df)
    """
    df_model = df.copy()

    # Extract primary cuisine if not already done
    if 'Primary Cuisine' not in df_model.columns:
        df_model['Primary Cuisine'] = df_model['Cuisines'].apply(
            lambda x: x.split(',')[0].strip()
            if pd.notnull(x) else 'Unknown'
        )

    # Group rare cuisines
    top_cuisines = (
        df_model['Primary Cuisine']
        .value_counts().head(20).index
    )
    df_model['Primary Cuisine'] = df_model['Primary Cuisine'].apply(
        lambda x: x if x in top_cuisines else 'Other'
    )

    # Group rare cities
    top_cities = df_model['City'].value_counts().head(20).index
    df_model['City Grouped'] = df_model['City'].apply(
        lambda x: x if x in top_cities else 'Other'
    )

    # Label encode categorical columns
    encoders = {}
    for col in ['Primary Cuisine', 'City Grouped']:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        encoders[col] = le

    # Define feature columns
    feature_cols = [
        'Country Code',
        'Average Cost for two',
        'Has Table booking',
        'Has Online delivery',
        'Is delivering now',
        'Price range',
        'Votes',
        'Primary Cuisine',
        'City Grouped'
    ]

    # Target variable
    target = 'Aggregate rating'

    X = df_model[feature_cols].copy()
    y = df_model[target].copy()

    print(f"Features prepared!")
    print(f"Feature shape : {X.shape}")
    print(f"Target range  : {y.min():.1f} to {y.max():.1f}")
    print(f"Features used : {feature_cols}")

    return X, y, encoders


# ============================================================
# SECTION 2 — DATA SPLITTING & SCALING
# ============================================================

def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and scale features.

    Parameters:
    -----------
    X            : pd.DataFrame → feature matrix
    y            : pd.Series    → target variable
    test_size    : float → proportion for testing (default 0.2)
    random_state : int   → reproducibility seed (default 42)

    Returns:
    --------
    X_train        : np.array → scaled training features
    X_test         : np.array → scaled testing features
    y_train        : pd.Series → training targets
    y_test         : pd.Series → testing targets
    X_train_raw    : pd.DataFrame → unscaled training features
    X_test_raw     : pd.DataFrame → unscaled testing features
    scaler         : StandardScaler → fitted scaler object

    Example:
    --------
    X_train, X_test, y_train, y_test, 
    X_train_raw, X_test_raw, scaler = split_and_scale(X, y)
    """
    # Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = random_state
    )

    # Scale
    scaler      = StandardScaler()
    X_train     = scaler.fit_transform(X_train_raw)
    X_test      = scaler.transform(X_test_raw)

    print(f"Data split and scaled!")
    print(f"Training set : {X_train_raw.shape[0]} restaurants")
    print(f"Testing set  : {X_test_raw.shape[0]} restaurants")

    return (X_train, X_test, y_train, y_test,
            X_train_raw, X_test_raw, scaler)


# ============================================================
# SECTION 3 — MODEL TRAINING
# ============================================================

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.

    Parameters:
    -----------
    X_train : array → scaled training features
    y_train : array → training targets

    Returns:
    --------
    model : trained LinearRegression model

    Example:
    --------
    lr_model = train_linear_regression(X_train, y_train)
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression trained!")
    return model


def train_decision_tree(X_train, y_train,
                        max_depth=10, random_state=42):
    """
    Train a Decision Tree Regressor.

    Parameters:
    -----------
    X_train      : array → training features (unscaled)
    y_train      : array → training targets
    max_depth    : int   → max tree depth (default 10)
    random_state : int   → reproducibility seed (default 42)

    Returns:
    --------
    model : trained DecisionTreeRegressor model

    Example:
    --------
    dt_model = train_decision_tree(X_train_raw, y_train)
    """
    model = DecisionTreeRegressor(
        max_depth    = max_depth,
        random_state = random_state
    )
    model.fit(X_train, y_train)
    print("Decision Tree trained!")
    return model


def train_random_forest(X_train, y_train,
                        n_estimators=100,
                        max_depth=10,
                        random_state=42):
    """
    Train a Random Forest Regressor.

    Parameters:
    -----------
    X_train      : array → training features (unscaled)
    y_train      : array → training targets
    n_estimators : int   → number of trees (default 100)
    max_depth    : int   → max tree depth (default 10)
    random_state : int   → reproducibility seed (default 42)

    Returns:
    --------
    model : trained RandomForestRegressor model

    Example:
    --------
    rf_model = train_random_forest(X_train_raw, y_train)
    """
    model = RandomForestRegressor(
        n_estimators = n_estimators,
        max_depth    = max_depth,
        random_state = random_state,
        n_jobs       = -1
    )
    model.fit(X_train, y_train)
    print("Random Forest trained!")
    return model


def train_all_models(X_train_scaled, X_train_raw, y_train):
    """
    Train all 3 regression models at once.

    Parameters:
    -----------
    X_train_scaled : array → scaled training features
    X_train_raw    : array → unscaled training features
    y_train        : array → training targets

    Returns:
    --------
    models : dict → dictionary of trained models

    Example:
    --------
    models = train_all_models(X_train_scaled, X_train_raw, y_train)
    """
    print("🚀 Training all models...")
    print("-" * 40)

    models = {
        'Linear Regression': train_linear_regression(
            X_train_scaled, y_train
        ),
        'Decision Tree'    : train_decision_tree(
            X_train_raw, y_train
        ),
        'Random Forest'    : train_random_forest(
            X_train_raw, y_train
        )
    }

    print("-" * 40)
    print("✅ All models trained successfully!")
    return models


# ============================================================
# SECTION 4 — MODEL EVALUATION
# ============================================================

def evaluate_model(name, y_true, y_pred):
    """
    Calculate regression evaluation metrics for one model.

    Metrics:
    - MSE  : Mean Squared Error (lower is better)
    - RMSE : Root Mean Squared Error (lower is better)
    - MAE  : Mean Absolute Error (lower is better)
    - R²   : R-squared score (higher is better, max=1.0)

    Parameters:
    -----------
    name   : str   → model name for display
    y_true : array → actual ratings
    y_pred : array → predicted ratings

    Returns:
    --------
    dict : evaluation metrics

    Example:
    --------
    metrics = evaluate_model("Random Forest", y_test, rf_pred)
    """
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    return {
        'Model': name,
        'MSE'  : round(mse,  4),
        'RMSE' : round(rmse, 4),
        'MAE'  : round(mae,  4),
        'R²'   : round(r2,   4)
    }


def evaluate_all_models(models, X_test_scaled,
                        X_test_raw, y_test):
    """
    Evaluate all trained models and return comparison table.

    Parameters:
    -----------
    models        : dict  → trained models dictionary
    X_test_scaled : array → scaled test features
    X_test_raw    : array → unscaled test features
    y_test        : array → actual ratings

    Returns:
    --------
    results_df : pd.DataFrame → model comparison table

    Example:
    --------
    results = evaluate_all_models(models, X_test_scaled,
                                  X_test_raw, y_test)
    """
    results = []

    for name, model in models.items():
        # Linear Regression needs scaled data
        if name == 'Linear Regression':
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test_raw)

        metrics = evaluate_model(name, y_test, y_pred)
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values(
        'R²', ascending=False
    )

    print("=" * 55)
    print("MODEL COMPARISON RESULTS")
    print("=" * 55)
    print(results_df.to_string(index=False))
    print("\n✅ Higher R² = Better | Lower RMSE = Better")

    return results_df


# ============================================================
# SECTION 5 — VISUALIZATION
# ============================================================

def plot_model_comparison(results_df, save_path=None):
    """
    Plot model comparison bar chart for R² and RMSE.

    Parameters:
    -----------
    results_df : pd.DataFrame → model evaluation results
    save_path  : str → path to save plot (optional)

    Example:
    --------
    plot_model_comparison(results_df,
        save_path='../plots/task1_model_comparison.png')
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors    = ['#2196F3', '#4CAF50', '#FF5722']
    models    = results_df['Model']

    # R² Score
    axes[0].bar(models, results_df['R²'], color=colors)
    axes[0].set_title('R² Score\n(Higher is Better)',
                      fontweight='bold')
    axes[0].set_ylabel('R² Score')
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(results_df['R²']):
        axes[0].text(i, v + 0.01, str(v),
                     ha='center', fontweight='bold')

    # RMSE
    axes[1].bar(models, results_df['RMSE'], color=colors)
    axes[1].set_title('RMSE\n(Lower is Better)',
                      fontweight='bold')
    axes[1].set_ylabel('RMSE')
    for i, v in enumerate(results_df['RMSE']):
        axes[1].text(i, v + 0.005, str(v),
                     ha='center', fontweight='bold')

    plt.suptitle('Model Comparison — Rating Prediction',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved: {save_path}")

    plt.show()


def plot_feature_importance(model, feature_cols, save_path=None):
    """
    Plot feature importance from Random Forest model.

    Parameters:
    -----------
    model       : trained RandomForestRegressor
    feature_cols: list → list of feature column names
    save_path   : str  → path to save plot (optional)

    Example:
    --------
    plot_feature_importance(rf_model, feature_cols,
        save_path='../plots/task1_feature_importance.png')
    """
    importance_df = pd.DataFrame({
        'Feature'   : feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data    = importance_df,
        x       = 'Importance',
        y       = 'Feature',
        palette = 'viridis'
    )
    plt.title('Feature Importance — What Drives Ratings?',
              fontsize=13, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved: {save_path}")

    plt.show()
    return importance_df


# ============================================================
# SECTION 6 — SAVE & LOAD MODELS
# ============================================================

def save_models(models, scaler, save_dir='../outputs/models/'):
    """
    Save trained models and scaler to disk.

    Parameters:
    -----------
    models   : dict → trained models dictionary
    scaler   : StandardScaler → fitted scaler
    save_dir : str → directory to save models

    Example:
    --------
    save_models(models, scaler)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save best model (Random Forest)
    joblib.dump(
        models['Random Forest'],
        os.path.join(save_dir, 'rating_model.pkl')
    )
    # Save scaler
    joblib.dump(
        scaler,
        os.path.join(save_dir, 'rating_scaler.pkl')
    )

    print(f"✅ Model saved : {save_dir}rating_model.pkl")
    print(f"✅ Scaler saved: {save_dir}rating_scaler.pkl")


def load_model(model_path, scaler_path):
    """
    Load saved model and scaler from disk.

    Parameters:
    -----------
    model_path  : str → path to saved model .pkl
    scaler_path : str → path to saved scaler .pkl

    Returns:
    --------
    model  : loaded model
    scaler : loaded scaler

    Example:
    --------
    model, scaler = load_model(
        '../outputs/models/rating_model.pkl',
        '../outputs/models/rating_scaler.pkl'
    )
    """
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model and scaler loaded successfully!")
    return model, scaler


# ============================================================
# SECTION 7 — PREDICTION
# ============================================================

def predict_rating(model, scaler, input_data):
    """
    Predict rating for a new restaurant.

    Parameters:
    -----------
    model      : trained model
    scaler     : fitted StandardScaler
    input_data : dict → restaurant features

    Returns:
    --------
    predicted_rating : float

    Example:
    --------
    restaurant = {
        'Country Code'        : 1,
        'Average Cost for two': 500,
        'Has Table booking'   : 1,
        'Has Online delivery' : 0,
        'Is delivering now'   : 0,
        'Price range'         : 2,
        'Votes'               : 100,
        'Primary Cuisine'     : 5,
        'City Grouped'        : 3
    }
    rating = predict_rating(model, scaler, restaurant)
    """
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    predicted = model.predict(input_scaled)[0]

    # Clip to valid range 0-5
    predicted = np.clip(predicted, 0, 5)

    print(f"✅ Predicted Rating: {predicted:.2f} / 5.0")
    return round(predicted, 2)


# ============================================================
# MAIN — Run full pipeline directly if needed
# ============================================================

if __name__ == "__main__":

    # Load data
    df = pd.read_csv('../data/restaurant_rated.csv')
    print(f"Loaded: {df.shape}")

    # Prepare features
    X, y, encoders = prepare_features(df)

    # Split and scale
    (X_train, X_test, y_train, y_test,
     X_train_raw, X_test_raw, scaler) = split_and_scale(X, y)

    # Train all models
    models = train_all_models(X_train, X_train_raw, y_train)

    # Evaluate all models
    results_df = evaluate_all_models(
        models, X_test, X_test_raw, y_test
    )

    # Plot results
    feature_cols = X.columns.tolist()
    plot_model_comparison(
        results_df,
        save_path='../plots/task1_model_comparison.png'
    )
    plot_feature_importance(
        models['Random Forest'],
        feature_cols,
        save_path='../plots/task1_feature_importance.png'
    )

    # Save models
    save_models(models, scaler)

    # Save results
    results_df.to_csv(
        '../outputs/reports/task1_model_results.csv',
        index=False
    )
    print("Results saved!")