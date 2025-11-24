#######################################################################
# AUTO TABULAR ML PRO – FULL PIPELINE (Stage 1 → Stage 5)
# Author: Omkar (ML Engineer)
# Description: Fully automated ML system: EDA, preprocessing, training,
#              evaluation, tuning, and final model export.
#######################################################################

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sys
sys.stdout.reconfigure(encoding='utf-8')


from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold,
    cross_val_score, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, PowerTransformer, LabelEncoder
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.base import BaseEstimator, TransformerMixin

from xgboost import XGBClassifier, XGBRegressor


#######################################################################
# CONFIGURATION
#######################################################################

RANDOM_STATE = 42
DATA_PATH = Path(r"C:\Users\okash\Desktop\data_science\abc\auto_ml_sample_dataset.csv")   #give here path
OUTPUT_DIR = Path("full_automl_output")
OUTPUT_DIR.mkdir(exist_ok=True)


#######################################################################
# STAGE 1 – EDA
#######################################################################

def run_eda(df, output_dir):

    print("\n====== STAGE 1: EDA ======\n")

    print("Data Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDescribe:\n", df.describe().T)

    target_col = df.columns[-1]

    # Save EDA summary
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "target": target_col,
        "missing_total": int(df.isnull().sum().sum())
    }
    with open(output_dir / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("EDA summary saved.")

    # Correlation Heatmap
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) >= 2:
        import seaborn as sns
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
        plt.close()

    return target_col


#######################################################################
# STAGE 2 – PREPROCESSING MODULE
#######################################################################

class SkewOutlierTransformer(BaseEstimator, TransformerMixin):
    """Fixes outliers + applies Yeo-Johnson transform on skewed columns."""

    def __init__(self, skew_threshold=0.75):
        self.skew_threshold = skew_threshold
        self.bounds_ = {}
        self.skewed_cols_ = []

    def fit(self, X, y=None):
        X = X.copy()
        num_cols = X.select_dtypes(include=[np.number]).columns

        for col in num_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.bounds_[col] = (lower, upper)

        self.skewed_cols_ = [
            col for col in num_cols if abs(X[col].skew()) > self.skew_threshold
        ]
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.select_dtypes(include=[np.number]).columns:
            lower, upper = self.bounds_[col]
            X[col] = np.clip(X[col], lower, upper)

        if self.skewed_cols_:
            pt = PowerTransformer(method="yeo-johnson")
            X[self.skewed_cols_] = pt.fit_transform(X[self.skewed_cols_])

        return X


#######################################################################
# STAGE 3 – AUTO TRAINING
#######################################################################

def auto_train(X_train, y_train, X_test, y_test, task, preprocessor):

    print("\n====== STAGE 3: MODEL TRAINING ======\n")

    classification_models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "SVC": SVC(kernel="rbf", probability=True),
        "KNN": KNeighborsClassifier(),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=150),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=150),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.1,
            eval_metric="logloss", n_jobs=-1
        )
    }

    regression_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=6),
        "Lasso": Lasso(alpha=0.1),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, n_jobs=-1),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=150),
        "XGBRegressor": XGBRegressor(n_estimators=200, learning_rate=0.1, n_jobs=-1)
    }

    models = classification_models if task == "classification" else regression_models

    results = []
    best_metric = None
    best_model_name = None
    best_pipeline = None

    for name, model in models.items():
        print(f"\n▶ Training {name}")

        pipe = Pipeline([
            ("skew_fix", SkewOutlierTransformer()),
            ("preprocess", preprocessor),
            ("model", model)
        ])

        # Cross-validation setup
        cv = StratifiedKFold(5) if task == "classification" else KFold(5)

        # CV score
        metric_type = "accuracy" if task == "classification" else "neg_root_mean_squared_error"
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=metric_type, n_jobs=-1)

        # Fit
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        record = {
            "model": name,
            "cv_mean": float(np.mean(scores)),
            "cv_std": float(np.std(scores))
        }

        # Classification metrics
        if task == "classification":
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            record.update({"accuracy": acc, "f1": f1})
            metric = acc

        else:  # Regression metrics
            rmse = mean_squared_error(y_test, preds, squared=False)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            record.update({"rmse": rmse, "mae": mae, "r2": r2})
            metric = -rmse

        print("Metrics:", record)
        results.append(record)

        # Select best
        if best_metric is None or metric > best_metric:
            best_metric = metric
            best_model_name = name
            best_pipeline = pipe

    # Save results
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "model_comparison_stage3.csv", index=False)
    joblib.dump(best_pipeline, OUTPUT_DIR / f"best_model_stage3_{best_model_name}.joblib")

    print("\nBest Model from Stage 3:", best_model_name)
    return best_model_name, best_pipeline


#######################################################################
# STAGE 4 – HYPERPARAMETER TUNING
#######################################################################

def hyper_tune(best_name, preprocessor, X_train, y_train, task):

    print("\n====== STAGE 4: HYPERPARAMETER TUNING ======\n")
    
    if task == "classification":
        base_model = XGBClassifier(eval_metric="logloss", n_jobs=-1)
        param_dist = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__subsample": [0.7, 0.9, 1.0]
        }
    else:
        base_model = XGBRegressor(n_jobs=-1)
        param_dist = {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [3, 5, 8],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.7, 1.0]
        }

    pipe = Pipeline([
        ("skew_fix", SkewOutlierTransformer()),
        ("preprocess", preprocessor),
        ("model", base_model)
    ])

    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=20,
        cv=5, n_jobs=-1, scoring="accuracy" if task=="classification" else "neg_root_mean_squared_error",
        random_state=42
    )

    search.fit(X_train, y_train)

    tuned_model = search.best_estimator_
    joblib.dump(tuned_model, OUTPUT_DIR / "best_tuned_model.joblib")

    print("Best Params:", search.best_params_)
    return tuned_model


#######################################################################
# STAGE 5 – FINAL MODEL BUILD & EXPORT
#######################################################################

def final_training(tuned_model, X, y):
    tuned_model.fit(X, y)
    joblib.dump(tuned_model, OUTPUT_DIR / "final_model.joblib")
    print("\n====== FINAL MODEL TRAINED AND SAVED ======\n")


#######################################################################
# MAIN SCRIPT
#######################################################################

if __name__ == "__main__":

    df = pd.read_csv(DATA_PATH)

    # Stage 1
    target_col = run_eda(df, OUTPUT_DIR)

    # Setup X,y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Label encode if needed
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    # Detect task
    task = "classification" if y.nunique() == 2 else "regression"
    print("\nDetected Task:", task)

    # Column detection
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Preprocessor
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task=="classification" else None
    )

    # Stage 3: Auto-train
    best_name, best_pipe = auto_train(X_train, y_train, X_test, y_test, task, preprocessor)

    # Stage 4: Hyperparameter tuning
    tuned_model = hyper_tune(best_name, preprocessor, X_train, y_train, task)

    # Stage 5: Final training on full dataset
    final_training(tuned_model, X, y)

    print("ALL STAGES EXECUTED SUCCESSFULLY!")
