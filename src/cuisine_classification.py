import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SECTION 1 — FEATURE PREPARATION
# ============================================================

def prepare_features(df, top_n_cuisines=10):
    """
    Prepare feature matrix and target variable
    for cuisine classification.

    Parameters:
    -----------
    df             : pd.DataFrame → cleaned dataset
    top_n_cuisines : int → number of top cuisines to keep
                          (default 10)

    Returns:
    --------
    X          : pd.DataFrame → feature matrix
    y          : np.array     → encoded target labels
    le         : LabelEncoder → fitted label encoder
    cuisine_list: list        → list of cuisine class names

    Example:
    --------
    X, y, le, cuisines = prepare_features(df, top_n_cuisines=10)
    """
    df_model = df.copy()

    # Extract primary cuisine
    df_model['Primary Cuisine'] = df_model['Cuisines'].apply(
        lambda x: x.split(',')[0].strip()
        if pd.notnull(x) else 'Unknown'
    )

    # Keep only top N cuisines
    top_cuisines = (
        df_model['Primary Cuisine']
        .value_counts()
        .head(top_n_cuisines)
        .index
    )
    df_model = df_model[
        df_model['Primary Cuisine'].isin(top_cuisines)
    ].copy()

    print(f"Top {top_n_cuisines} cuisines selected!")
    print(f"Dataset size after filtering : {df_model.shape[0]}")
    print(f"Cuisines : {top_cuisines.tolist()}")

    # Define features
    feature_cols = [
        'Country Code',
        'Average Cost for two',
        'Has Table booking',
        'Has Online delivery',
        'Is delivering now',
        'Price range',
        'Aggregate rating',
        'Votes'
    ]

    X = df_model[feature_cols].copy()

    # Encode target variable
    le = LabelEncoder()
    y  = le.fit_transform(df_model['Primary Cuisine'])

    print(f"\nFeatures prepared!")
    print(f"Feature shape : {X.shape}")
    print(f"Classes       : {le.classes_.tolist()}")

    return X, y, le, le.classes_.tolist()


# ============================================================
# SECTION 2 — DATA SPLITTING & SCALING
# ============================================================

def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and scale features.
    Uses stratify to maintain class distribution in both sets.

    Parameters:
    -----------
    X            : pd.DataFrame → feature matrix
    y            : array        → encoded target labels
    test_size    : float → test proportion (default 0.2)
    random_state : int   → reproducibility seed (default 42)

    Returns:
    --------
    X_train        : array          → scaled training features
    X_test         : array          → scaled testing features
    y_train        : array          → training labels
    y_test         : array          → testing labels
    X_train_raw    : pd.DataFrame   → unscaled training features
    X_test_raw     : pd.DataFrame   → unscaled testing features
    scaler         : StandardScaler → fitted scaler

    Example:
    --------
    X_train, X_test, y_train, y_test,
    X_train_raw, X_test_raw, scaler = split_and_scale(X, y)
    """
    # stratify ensures equal class distribution in splits
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = random_state,
        stratify     = y
    )

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

def train_logistic_regression(X_train, y_train,
                               max_iter=1000,
                               random_state=42):
    """
    Train Logistic Regression classifier.

    Parameters:
    -----------
    X_train      : array → scaled training features
    y_train      : array → training labels
    max_iter     : int   → max iterations (default 1000)
    random_state : int   → reproducibility seed (default 42)

    Returns:
    --------
    model : trained LogisticRegression model

    Example:
    --------
    lr_model = train_logistic_regression(X_train, y_train)
    """
    model = LogisticRegression(
        max_iter     = max_iter,
        random_state = random_state
    )
    model.fit(X_train, y_train)
    print("Logistic Regression trained!")
    return model


def train_random_forest(X_train, y_train,
                        n_estimators=100,
                        max_depth=10,
                        random_state=42):
    """
    Train Random Forest classifier.

    Parameters:
    -----------
    X_train      : array → training features (unscaled)
    y_train      : array → training labels
    n_estimators : int   → number of trees (default 100)
    max_depth    : int   → max tree depth (default 10)
    random_state : int   → reproducibility seed (default 42)

    Returns:
    --------
    model : trained RandomForestClassifier model

    Example:
    --------
    rf_model = train_random_forest(X_train_raw, y_train)
    """
    model = RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth    = max_depth,
        random_state = random_state,
        n_jobs       = -1
    )
    model.fit(X_train, y_train)
    print("Random Forest trained!")
    return model


def train_xgboost(X_train, y_train,
                  n_estimators=100,
                  max_depth=6,
                  learning_rate=0.1,
                  random_state=42):
    """
    Train XGBoost classifier.

    Parameters:
    -----------
    X_train       : array → training features (unscaled)
    y_train       : array → training labels
    n_estimators  : int   → boosting rounds (default 100)
    max_depth     : int   → tree depth (default 6)
    learning_rate : float → step size (default 0.1)
    random_state  : int   → reproducibility seed (default 42)

    Returns:
    --------
    model : trained XGBClassifier model

    Example:
    --------
    xgb_model = train_xgboost(X_train_raw, y_train)
    """
    model = XGBClassifier(
        n_estimators  = n_estimators,
        max_depth     = max_depth,
        learning_rate = learning_rate,
        random_state  = random_state,
        eval_metric   = 'mlogloss',
        verbosity     = 0
    )
    model.fit(X_train, y_train)
    print("XGBoost trained!")
    return model


def train_all_models(X_train_scaled, X_train_raw, y_train):
    """
    Train all 3 classification models at once.

    Parameters:
    -----------
    X_train_scaled : array → scaled training features
    X_train_raw    : array → unscaled training features
    y_train        : array → training labels

    Returns:
    --------
    models : dict → dictionary of all trained models

    Example:
    --------
    models = train_all_models(X_train_scaled, X_train_raw, y_train)
    """
    print("Training all models...")
    print("-" * 40)

    models = {
        'Logistic Regression': train_logistic_regression(
            X_train_scaled, y_train
        ),
        'Random Forest'      : train_random_forest(
            X_train_raw, y_train
        ),
        'XGBoost'            : train_xgboost(
            X_train_raw, y_train
        )
    }

    print("-" * 40)
    print("All models trained successfully!")
    return models


# ============================================================
# SECTION 4 — MODEL EVALUATION
# ============================================================

def evaluate_model(name, y_true, y_pred):
    """
    Calculate classification metrics for one model.

    Metrics:
    - Accuracy  : overall correct predictions
    - Precision : correctness of positive predictions
    - Recall    : coverage of actual positives
    - F1 Score  : harmonic mean of precision and recall

    Parameters:
    -----------
    name   : str   → model name for display
    y_true : array → actual labels
    y_pred : array → predicted labels

    Returns:
    --------
    dict : evaluation metrics

    Example:
    --------
    metrics = evaluate_model("XGBoost", y_test, xgb_pred)
    """
    return {
        'Model'    : name,
        'Accuracy' : round(accuracy_score(
            y_true, y_pred), 4),
        'Precision': round(precision_score(
            y_true, y_pred,
            average='weighted',
            zero_division=0), 4),
        'Recall'   : round(recall_score(
            y_true, y_pred,
            average='weighted',
            zero_division=0), 4),
        'F1 Score' : round(f1_score(
            y_true, y_pred,
            average='weighted',
            zero_division=0), 4)
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
    y_test        : array → actual labels

    Returns:
    --------
    results_df : pd.DataFrame → model comparison table
    predictions: dict         → predictions from each model

    Example:
    --------
    results_df, preds = evaluate_all_models(
        models, X_test_scaled, X_test_raw, y_test
    )
    """
    results     = []
    predictions = {}

    for name, model in models.items():
        # Logistic Regression needs scaled data
        if name == 'Logistic Regression':
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test_raw)

        predictions[name] = y_pred
        results.append(evaluate_model(name, y_test, y_pred))

    results_df = pd.DataFrame(results).sort_values(
        'F1 Score', ascending=False
    )

    print("=" * 65)
    print("MODEL COMPARISON RESULTS")
    print("=" * 65)
    print(results_df.to_string(index=False))
    print("\n Higher scores = Better performance")

    return results_df, predictions


def print_classification_report(y_test, y_pred,
                                 class_names, model_name):
    """
    Print detailed classification report per cuisine.

    Parameters:
    -----------
    y_test      : array → actual labels
    y_pred      : array → predicted labels
    class_names : list  → cuisine class names
    model_name  : str   → model name for display

    Example:
    --------
    print_classification_report(
        y_test, xgb_pred, le.classes_, "XGBoost"
    )
    """
    print(f"\n{'='*65}")
    print(f"DETAILED CLASSIFICATION REPORT — {model_name}")
    print(f"{'='*65}")
    print(classification_report(
        y_test, y_pred,
        target_names = class_names,
        zero_division = 0
    ))


# ============================================================
# SECTION 5 — VISUALIZATION
# ============================================================

def plot_model_comparison(results_df, save_path=None):
    """
    Plot model comparison bar chart for all metrics.

    Parameters:
    -----------
    results_df : pd.DataFrame → model evaluation results
    save_path  : str → path to save plot (optional)

    Example:
    --------
    plot_model_comparison(results_df,
        plot_path = os.path.join(BASE_DIR, 'plots', 'task3_model_comparison.png')
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x       = np.arange(len(metrics))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (_, row) in enumerate(results_df.iterrows()):
        values = [row[m] for m in metrics]
        ax.bar(x + i * width, values, width,
               label=row['Model'], alpha=0.85)

    ax.set_title('Model Comparison — Cuisine Classification',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()

    if save_path:
      os.makedirs(os.path.dirname(save_path), exist_ok=True)
      plt.savefig(save_path, dpi=150, bbox_inches='tight')  
    

    plt.show()


def plot_confusion_matrix(y_test, y_pred,
                          class_names,
                          model_name,
                          save_path=None):
    """
    Plot confusion matrix heatmap.

    Parameters:
    -----------
    y_test      : array → actual labels
    y_pred      : array → predicted labels
    class_names : list  → cuisine class names
    model_name  : str   → model name for title
    save_path   : str   → path to save plot (optional)

    Example:
    --------
    plot_confusion_matrix(
        y_test, xgb_pred, le.classes_, "XGBoost",
        save_path='../plots/task3_confusion_matrix.png'
    )
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        cm,
        annot        = True,
        fmt          = 'd',
        cmap         = 'Blues',
        xticklabels  = class_names,
        yticklabels  = class_names
    )
    plt.title(f'Confusion Matrix — {model_name}',
              fontsize=13, fontweight='bold')
    plt.ylabel('Actual Cuisine')
    plt.xlabel('Predicted Cuisine')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  

    plt.show()


def plot_class_distribution(df, save_path=None):
    """
    Plot distribution of cuisine classes.

    Parameters:
    -----------
    df        : pd.DataFrame → dataset with Primary Cuisine column
    save_path : str → path to save plot (optional)

    Example:
    --------
    plot_class_distribution(df,
        save_path='../plots/task3_class_distribution.png')
    """
    cuisine_counts = df['Primary Cuisine'].value_counts()

    plt.figure(figsize=(12, 5))
    sns.barplot(
        x       = cuisine_counts.index,
        y       = cuisine_counts.values,
        palette = 'viridis'
    )
    plt.title('Class Distribution — Cuisine Types',
              fontsize=13, fontweight='bold')
    plt.xlabel('Cuisine')
    plt.ylabel('Number of Restaurants')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
       os.makedirs(os.path.dirname(save_path), exist_ok=True)
       plt.savefig(save_path, dpi=150, bbox_inches='tight')  
          
    plt.show()


# ============================================================
# SECTION 6 — SAVE & LOAD
# ============================================================

def save_models(models, scaler, le,
                save_dir='../outputs/models/'):
    """
    Save trained models, scaler and label encoder.

    Parameters:
    -----------
    models   : dict          → trained models dictionary
    scaler   : StandardScaler→ fitted scaler
    le       : LabelEncoder  → fitted label encoder
    save_dir : str           → directory to save files

    Example:
    --------
    save_models(models, scaler, le)
    """
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(
        models['XGBoost'],
        os.path.join(save_dir, 'cuisine_classifier.pkl')
    )
    joblib.dump(
        scaler,
        os.path.join(save_dir, 'cuisine_scaler.pkl')
    )
    joblib.dump(
        le,
        os.path.join(save_dir, 'cuisine_label_encoder.pkl')
    )

    print(f"Classifier saved : {save_dir}cuisine_classifier.pkl")
    print(f"Scaler saved     : {save_dir}cuisine_scaler.pkl")
    print(f"Encoder saved    : {save_dir}cuisine_label_encoder.pkl")


def load_model(model_path, scaler_path, encoder_path):
    """
    Load saved classifier, scaler and encoder.

    Parameters:
    -----------
    model_path   : str → path to saved model .pkl
    scaler_path  : str → path to saved scaler .pkl
    encoder_path : str → path to saved encoder .pkl

    Returns:
    --------
    model  : loaded classifier
    scaler : loaded scaler
    le     : loaded label encoder

    Example:
    --------
    model, scaler, le = load_model(
        '../outputs/models/cuisine_classifier.pkl',
        '../outputs/models/cuisine_scaler.pkl',
        '../outputs/models/cuisine_label_encoder.pkl'
    )
    """
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le     = joblib.load(encoder_path)
    print("Model, scaler and encoder loaded!")
    return model, scaler, le


# ============================================================
# SECTION 7 — PREDICTION
# ============================================================

def predict_cuisine(model, scaler, le, input_data):
    """
    Predict cuisine type for a new restaurant.

    Parameters:
    -----------
    model      : trained classifier
    scaler     : fitted StandardScaler
    le         : fitted LabelEncoder
    input_data : dict → restaurant features

    Returns:
    --------
    predicted_cuisine : str → predicted cuisine name

    Example:
    --------
    restaurant = {
        'Country Code'        : 1,
        'Average Cost for two': 500,
        'Has Table booking'   : 1,
        'Has Online delivery' : 0,
        'Is delivering now'   : 0,
        'Price range'         : 2,
        'Aggregate rating'    : 3.5,
        'Votes'               : 100
    }
    cuisine = predict_cuisine(model, scaler, le, restaurant)
    """
    input_df     = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    cuisine_name = le.inverse_transform([prediction])[0]

    print(f"Predicted Cuisine: {cuisine_name}")
    return cuisine_name


# ============================================================
# MAIN — Run full pipeline directly if needed
# ============================================================

if __name__ == "__main__":

    # Load data
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, 'data', 'restaurant_cleaned.csv')
    df = pd.read_csv(data_path)
    print(f"Loaded from: {data_path}")    

    # Prepare features
    X, y, le, cuisine_list = prepare_features(
        df, top_n_cuisines=10
    )

    # Plot class distribution
    df['Primary Cuisine'] = df['Cuisines'].apply(
        lambda x: x.split(',')[0].strip()
        if pd.notnull(x) else 'Unknown'
    )
    plot_class_distribution(
        df[df['Primary Cuisine'].isin(cuisine_list)],
        save_path='../plots/task3_class_distribution.png'
    )

    # Split and scale
    (X_train, X_test, y_train, y_test,
     X_train_raw, X_test_raw, scaler) = split_and_scale(X, y)

    # Train all models
    models = train_all_models(X_train, X_train_raw, y_train)

    # Evaluate
    results_df, predictions = evaluate_all_models(
        models, X_test, X_test_raw, y_test
    )

    # Detailed report for best model
    print_classification_report(
        y_test,
        predictions['XGBoost'],
        cuisine_list,
        "XGBoost"
    )

    # Plots
    plot_path = os.path.join(BASE_DIR, 'plots', 'task3_model_comparison.png')

    confusion_path = os.path.join(BASE_DIR, 'plots', 'task3_confusion_matrix.png')
    
    plot_model_comparison(
        results_df,
        save_path = plot_path
    )
    plot_confusion_matrix(
        y_test,
        predictions['XGBoost'],
        cuisine_list,
        "XGBoost",
        save_path=confusion_path
    )

    # Save
    save_models(models, scaler, le)

    # Save results
    save_path = os.path.join(BASE_DIR, 'outputs', 'reports', 'task3_model_results.csv')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df.to_csv(save_path, index=False)

    print(f"Saved: {save_path}")
    print("All done!")