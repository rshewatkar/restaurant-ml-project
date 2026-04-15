---
title: Restaurant ML Project
emoji: 🍽️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.45.1"
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🍽️ Restaurant ML Project — Cognifyz Technologies

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-red.svg)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-ML-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Deployed-yellow.svg)

---

## 🚀 Live Demo

👉 **[Click Here to Try the App](https://rshewatkar-restaurant-ml-project.hf.space)**

> Deployed on Hugging Face Spaces — No installation required!

---

## 📌 Project Overview

This project is part of the **Cognifyz Technologies Machine Learning 
Internship**. It analyzes a dataset of **9,551 restaurants** across 
multiple countries to build end-to-end machine learning solutions for:

- 🎯 Predicting restaurant ratings
- 🍜 Classifying cuisine types
- 🔍 Recommending restaurants
- 🗺️ Location-based geographical analysis

---
## 📁 Project Structure

```bash
restaurant-ml-project/
│
├── .gitignore
├── app.py                          # Main Streamlit app
├── Dockerfile                      # Containerization setup
├── README.md
├── requirements.txt
├── utils.py                        # Shared utility functions
│
├── .github/
│   └── workflows/
│       └── sync-to-hub.yml         # CI/CD pipeline (GitHub → Hugging Face)
│
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
│
├── data/
│   ├── restaurant_dataset.csv      # Raw dataset
│   ├── restaurant_cleaned.csv      # Cleaned dataset
│   └── restaurant_rated.csv        # Filtered dataset
│
├── notebooks/
│   ├── EDA_and_Preprocessing.ipynb
│   ├── Task1_Rating_Prediction.ipynb
│   ├── Task2_Recommendation_System.ipynb
│   ├── Task3_Cuisine_Classification.ipynb
│   └── Task4_Location_Analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── rating_prediction.py
│   ├── cuisine_classification.py
│   ├── recommendation_system.py
│   └── location_analysis.py
│
├── outputs/
│   ├── models/                     # Trained ML models
│   │   ├── rating_model.pkl
│   │   ├── cuisine_classifier.pkl
│   │   ├── tfidf_vectorizer.pkl
│   │   └── ...
│   │
│   ├── reports/                    # Evaluation results
│   │   ├── task1_model_results.csv
│   │   ├── task2_sample_recommendations.csv
│   │   └── ...
│   │
│   └── maps/                       # Folium visualizations
│       ├── map_clustered.html
│       ├── map_heatmap.html
│       └── map_ratings.html
│
├── pages/                          # Streamlit multi-page app
│   ├── 1_Rating_Prediction.py
│   ├── 2_Cuisine_Classification.py
│   ├── 3_Recommendation.py
│   └── 4_Location_Analysis.py
│
├── plots/                          # Visualizations
│   ├── correlation_heatmap.png
│   ├── rating_distribution.png
│   ├── task1_feature_importance.png
│   ├── task2_recommendations.png
│   ├── task3_class_distribution.png
│   ├── task3_confusion_matrix.png
│   ├── task3_model_comparison.png
│   ├── task4_avg_rating_by_city.png
│   ├── task4_cuisine_by_city.png
│   ├── task4_price_by_city.png
│   ├── task4_top_cities.png
│   ├── top_cities.png
│   └── top_cuisines.png

```
---

## 📊 Dataset Overview

| Property | Details |
|---|---|
| Total Restaurants | 9,551 |
| Total Features | 21 |
| Countries Covered | 15+ (primarily India) |
| Missing Values | 9 (only in Cuisines) |
| Target Variable | Aggregate Rating (0–4.9) |

---

## ✅ ML Tasks — All Completed

### 📓 Task 1 — Restaurant Rating Prediction ✅
- **Type:** Regression
- **Models:** Linear Regression, Decision Tree, Random Forest
- **Best Model:** Random Forest
- **Metrics:** MSE, RMSE, MAE, R²
- **Key Finding:** Votes and Price Range are strongest 
  predictors of restaurant ratings

---

### 📓 Task 2 — Restaurant Recommendation System ✅
- **Type:** Content-Based Filtering
- **Method:** Cosine Similarity on TF-IDF + numerical features
- **Input:** Cuisine preference, city, price range, min rating
- **Output:** Top N personalized restaurant recommendations
- **Key Finding:** Cuisine type weighted 2x for better matching

---

### 📓 Task 3 — Cuisine Classification ✅
- **Type:** Multi-class Classification
- **Models:** Logistic Regression, Random Forest, XGBoost
- **Best Model:** XGBoost
- **Metrics:** Accuracy, Precision, Recall, F1-Score
- **Key Finding:** Class imbalance identified — 
  North Indian cuisine heavily dominates dataset

---

### 📓 Task 4 — Location-Based Analysis ✅
- **Type:** Geographical Analysis
- **Tools:** Folium interactive maps, Matplotlib, Seaborn
- **Maps:** Clustered map, Density heatmap, Rating color map
- **Key Finding:** New Delhi has highest restaurant 
  concentration; smaller cities have higher avg ratings

---

## 🖥️ App Features

| Page | Feature |
|---|---|
| 🏠 Home | Dataset overview + key insights charts |
| ⭐ Rating Prediction | Predict rating from restaurant features |
| 🍜 Cuisine Classification | Predict cuisine type with confidence % |
| 🔍 Recommendation | Get top 5 personalized recommendations |
| 🗺️ Location Analysis | 3 interactive maps + city statistics |

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| ML Models | Scikit-Learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Maps | Folium, Streamlit-Folium |
| App Framework | Streamlit |
| Deployment | Hugging Face Spaces |

---

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/rshewatkar/restaurant-ml-project.git
cd restaurant-ml-project
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in browser**
http://localhost:8501

---

## 📈 Key Findings

1. 🏆 **Random Forest** achieved best performance 
   for rating prediction
2. 🍛 **North Indian** is the most common cuisine 
   across Indian cities
3. 🏙️ **New Delhi** has 4x more restaurants than 
   any other city in the dataset
4. ⭐ Restaurants with **more votes** tend to have 
   more stable and higher ratings
5. 💰 **80%+ restaurants** fall in price range 1-2 
   (budget to moderate)
6. 🗺️ **Smaller cities** tend to have higher average 
   ratings than metro cities

---

## 👤 Author

**Rahul Shewatkar**
Machine Learning Intern — Cognifyz Technologies

[![GitHub](https://img.shields.io/badge/GitHub-rshewatkar-black?logo=github)](https://github.com/rshewatkar)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/rshewatkar)

---

## 📄 License

This project is built for educational purposes as part of the
Cognifyz Technologies Machine Learning Internship Program.
