#  Taxi Fare Prediction — End-to-End ML Pipeline

> Predicting taxi fares from an intentionally dirty, real-world-style InDrive dataset.  
> Covers the full pipeline: **data cleaning → feature engineering → outlier handling → model training → evaluation**.

---

##  Model Results

| Metric | Score |
|--------|-------|
| **R² Score** | **0.9748** |

---

##  Project Structure

```
taxi-fare-prediction/
│
├── dirty_taxi_fare_data.csv          # Raw dirty dataset
├── taxi_fare_prediction_pro.ipynb    # Main notebook (end-to-end)
├── requirements-ml-basics.txt        # Dependencies
└── README.md
```

---

##  Pipeline Overview

```
Raw Dirty CSV
     │
     ▼
1. Data Loading & EDA          ← shape, dtypes, missing %, distributions
     │
     ▼
2. Data Cleaning               ← nulls, types, duplicates, leakage removal
     │
     ▼
3. Outlier Treatment           ← IQR method on fare, distance, duration
     │
     ▼
4. Feature Engineering         ← datetime, haversine, speed, binning
     │
     ▼
5. Encoding & Selection        ← One-Hot, correlation filter
     │
     ▼
6. Scaling + Model             ← StandardScaler + MLPRegressor
     │
     ▼
7. Evaluation                  ← MAE / RMSE / R² / residuals
```

---

## 🔍 Dataset Problems Handled

| Problem | Description | Fix Applied |
|---------|-------------|-------------|
| **Nulls** | Up to 27% missing in some columns | Median / Mode imputation |
| **Outliers** | Fares ×80, impossible distances | IQR filtering |
| **Mixed dtypes** | Numerics containing "NaN", "?" strings | `pd.to_numeric(errors='coerce')` |
| **Inconsistent categories** | cash / Cash / CASH / N/A | `.str.strip().str.lower()` + replace |
| **Mixed datetime formats** | 4 different date formats in one column | `pd.to_datetime(format='mixed')` |
| **Duplicate rows** | ~2% exact duplicate trips | `drop_duplicates()` |
| **One-to-many** | Same trip_id logged multiple times | Dedup by trip_id |
| **Many-to-one** | Same driver with spelling variants | Normalised + dropped |
| **Multicollinearity** | distance_km ↔ distance_miles (r=1.0) | Dropped redundant column |
| **Data Leakage** | `fare_amount_plus_tip_leak` | Dropped before training |
| **Zero-variance** | `constant_col` = "v1" everywhere | Dropped |
| **Irrelevant features** | row_uuid, server_log_id, app_version | Dropped |

---

## 🛠️ Feature Engineering

| Feature | Description |
|---------|-------------|
| `hour` | Hour extracted from pickup_datetime |
| `weekday` | Day of week (0=Mon … 6=Sun) |
| `month` | Month of trip |
| `weekend` | 1 if Saturday or Sunday |
| `rush_hour` | 1 if 7–9 AM or 4–7 PM |
| `time_category` | morning / afternoon / night |
| `coord_distance_km` | Haversine distance from coordinates |
| `avg_speed_kmh` | distance_km ÷ (duration_min / 60) |
| `distance_category` | short (<5km) / medium / long (>15km) |
| `car_age` | 2025 − car_year |

---

##  Visualizations

###  Missing Values (Raw Data)
![Missing Values](github_plots/01_missing_values.png)

###  Outlier Treatment — Before vs After
![Outliers](github_plots/02_outliers_before_after.png)

###  Fare Amount Distribution
![Fare Distribution](github_plots/03_fare_distribution.png)

###  Correlation Matrix
![Correlation](github_plots/04_correlation_heatmap.png)

###  Feature Correlation with Target
![Target Correlation](github_plots/05_target_correlation.png)

###  Categorical Distributions
![Categorical](github_plots/06_categorical_distributions.png)

###  Temporal Fare Patterns
![Temporal](github_plots/07_temporal_patterns.png)

###  Distance vs Fare (by Surge)
![Distance vs Fare](github_plots/08_distance_vs_fare.png)

###  Actual vs Predicted Fare
![Predictions](github_plots/09_actual_vs_predicted.png)

###  Model Evaluation
![Evaluation](github_plots/10_model_evaluation.png)

---

##  Quick Start

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/taxi-fare-prediction.git
cd taxi-fare-prediction

# 2. Create virtual environment (Python 3.10 or 3.11)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements-ml-basics.txt

# 4. Run notebook
jupyter notebook taxi_fare_prediction_pro.ipynb
```

---

##  Requirements

```
numpy
pandas
scipy
matplotlib
seaborn
scikit-learn
jupyter
```

---

##  Model Architecture

```
Input Layer (n features)
        │
   Dense(128) + ReLU
        │
   Dense(64)  + ReLU
        │
   Dense(32)  + ReLU
        │
   Output (1) — Fare Amount
```
Optimizer: **Adam** | Early Stopping:  | Max Iterations: 500

---

## 👤 Author

Made with  for learning Data Science & ML end-to-end pipelines.
