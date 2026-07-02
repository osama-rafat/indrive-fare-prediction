#  indrive Fare Prediction using Machine Learning

### End-to-End Regression Pipeline with Feature Engineering & Neural Network (MLPRegressor)


---

## 📌 Project Overview

This project presents a complete **end-to-end Machine Learning pipeline** for predicting taxi fares using a realistic, intentionally dirty InDrive-style dataset.

The project demonstrates the full Data Science workflow, including data preprocessing, feature engineering, outlier treatment, model training, evaluation, and visualization using **Scikit-learn's MLPRegressor (Neural Network Regression).**

---

## 🚀 Project Highlights

- Built a complete Machine Learning regression pipeline.
- Cleaned and transformed noisy real-world data.
- Handled missing values, duplicates, inconsistent categories, and mixed data types.
- Applied feature engineering using spatial and temporal features.
- Removed data leakage and multicollinearity.
- Trained a Neural Network Regression model using **MLPRegressor**.
- Achieved excellent predictive performance.

---

# 📊 Model Performance

| Metric | Score |
|---------|-------|
| **R² Score** | **0.9748** |

---

# 📂 Project Structure

```text
Taxi-Fare-Prediction/

│
├── dirty_taxi_fare_data.csv
├── taxi_fare_prediction_pro.ipynb
├── requirements-ml-basics.txt
├── README.md
└── github_plots/
```

---

# ⚙️ Complete Pipeline

```text
Raw Dirty CSV
      │
      ▼
1. Data Loading & EDA
      │
      ▼
2. Data Cleaning
      │
      ▼
3. Outlier Detection & Removal
      │
      ▼
4. Feature Engineering
      │
      ▼
5. Encoding & Feature Selection
      │
      ▼
6. Feature Scaling
      │
      ▼
7. Neural Network Regression (MLPRegressor)
      │
      ▼
8. Model Evaluation
      │
      ▼
9. Visualization
```

---

# 🧹 Dataset Problems Solved

| Problem | Solution |
|----------|----------|
| Missing Values | Median / Mode Imputation |
| Duplicate Records | Removed |
| Mixed Numeric Types | Converted using `pd.to_numeric(errors="coerce")` |
| Mixed Date Formats | `pd.to_datetime(format="mixed")` |
| Inconsistent Categories | Normalized using `.str.lower().str.strip()` |
| Outliers | Removed using IQR |
| Multicollinearity | Removed highly correlated features |
| Data Leakage | Removed leaked target columns |
| Zero Variance Columns | Removed |
| Irrelevant Features | Dropped |

---

# 🛠 Feature Engineering

| Feature | Description |
|----------|-------------|
| hour | Pickup hour |
| weekday | Day of week |
| month | Pickup month |
| weekend | Weekend indicator |
| rush_hour | Peak traffic hours |
| time_category | Morning / Afternoon / Evening / Night |
| coord_distance_km | Haversine distance |
| avg_speed_kmh | Estimated trip speed |
| distance_category | Short / Medium / Long |
| car_age | Vehicle age |

---

# 🧠 Neural Network Architecture

```text
Input Features
      │
      ▼
Dense (128) + ReLU
      │
      ▼
Dense (64) + ReLU
      │
      ▼
Dense (32) + ReLU
      │
      ▼
Output Layer (Regression)
```

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Model | MLPRegressor |
| Activation | ReLU |
| Solver | Adam |
| Loss Function | Mean Squared Error |
| Hidden Layers | (128, 64, 32) |
| Max Iterations | 500 |

---

# 📈 Visualizations

## Missing Values

```
github_plots/01_missing_values.png
```

## Outlier Treatment

```
github_plots/02_outliers_before_after.png
```

## Fare Distribution

```
github_plots/03_fare_distribution.png
```

## Correlation Heatmap

```
github_plots/04_correlation_heatmap.png
```

## Target Correlation

```
github_plots/05_target_correlation.png
```

## Categorical Features

```
github_plots/06_categorical_distributions.png
```

## Temporal Patterns

```
github_plots/07_temporal_patterns.png
```

## Distance vs Fare

```
github_plots/08_distance_vs_fare.png
```

## Actual vs Predicted

```
github_plots/09_actual_vs_predicted.png
```

## Model Evaluation

```
github_plots/10_model_evaluation.png
```

---

# ▶️ Quick Start

```bash
# Clone repository

git clone https://github.com/osama-rafat/indrive-fare-prediction.git

cd indrive-fare-prediction

# Create virtual environment

python -m venv venv

# Activate

venv\Scripts\activate

# Install dependencies

pip install -r requirements-ml-basics.txt

# Run Notebook

jupyter notebook taxi_fare_prediction_pro.ipynb
```

---

# 📦 Requirements

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
jupyter
```

---

# 📈 Evaluation Metrics

The model is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Actual vs Predicted Visualization
- Residual Analysis

---

# 📌 Future Improvements

- XGBoost Regression
- LightGBM
- CatBoost
- Hyperparameter Tuning
- FastAPI Deployment
- Docker Support
- Streamlit Web Application

---

# 👨‍💻 Author

**Osama Raafat**

Computer Science Student

Machine Learning & Artificial Intelligence Enthusiast

---

⭐ If you found this project useful, consider giving it a star on GitHub!
