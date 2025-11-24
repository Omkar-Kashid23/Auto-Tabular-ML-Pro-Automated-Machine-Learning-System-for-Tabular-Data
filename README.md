# 🚀 Auto Tabular ML Pro  
### _Fully Automated Machine Learning Pipeline for Any Tabular Dataset_

Auto Tabular ML Pro is a **complete end-to-end AutoML system** built in Python that automatically analyzes, preprocesses, trains, optimizes, and exports state-of-the-art machine learning models for any tabular dataset.

This project is designed for **ML engineers**, **students**, and **production teams** who want a high-quality, reproducible tabular ML pipeline without manual feature engineering or model selection.

---

## 🌟 Features

### ✅ **Stage 1 — EDA (Exploratory Data Analysis)**
- Automatic detection of numeric and categorical columns  
- Dataset summary: shape, dtypes, missing values, unique counts  
- Correlation heatmap (numeric-only)  
- Saves:  
  - `eda_summary.json`  
  - `correlation_heatmap.png`  

---

### ✅ **Stage 2 — Preprocessing Engine**
Handles all major preprocessing steps required for tabular ML:

| Step | Description |
|------|-------------|
| Missing Value Imputation | Median (numeric), Mode (categorical) |
| Outlier Detection | IQR clipping using custom transformer |
| Skewness Adjustment | Yeo–Johnson transformation |
| Encoding | OneHot for categorical features |
| Scaling | StandardScaler (numeric features) |

All transformations are wrapped inside a **Scikit-Learn Pipeline** to ensure reproducibility.

---

### ✅ **Stage 3 — Auto Model Training**
Trains multiple models automatically and ranks them:

#### Classification Models
- Logistic Regression  
- SVC (RBF)  
- KNN  
- GaussianNB  
- Decision Tree  
- Random Forest  
- AdaBoost  
- Gradient Boosting  
- XGBoost  

#### Regression Models
- Linear Regression  
- Ridge  
- Lasso  
- RandomForestRegressor  
- GradientBoostingRegressor  
- XGBRegressor  

Outputs:
- Cross-validation scores  
- Accuracy/F1 scores (classification)  
- RMSE/MAE/R² (regression)  
- `model_comparison_stage3.csv`  
- `best_model_stage3_<Model>.joblib`  

---

### ✅ **Stage 4 — Hyperparameter Tuning**
Uses **RandomizedSearchCV** to tune top models:

- XGBoost (default)
- Support for both classification & regression
- Auto-selects best hyperparameters
- Output:  
  - `best_tuned_model.joblib`

---

### ✅ **Stage 5 — Final Model Export**
Retrains the best tuned model on the full dataset and exports:

- `final_model.joblib`

This is the file used in production (API, Streamlit, deployment, etc.)

---

### ✅ **Bonus: Auto-Generated HTML Report**
A beautiful, structured HTML report that includes:

- Dataset summary  
- Correlation heatmap  
- Model comparison table  
- Metrics summary  
- Download links for all `.joblib` artifacts  
- Recommendations  

Saved as:

