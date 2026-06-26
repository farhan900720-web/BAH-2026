"""
train_models.py
================
Trains three RandomForestRegressor models to predict daily climate variables
(tmax, tmin, rainfall) from engineered features, evaluates each model, and
persists them as compressed .joblib files for web deployment.

Optimised for size: fewer trees (15), shallower depth (10), joblib compress=3.

Usage:
    python models/train_models.py
"""

import os
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

DATA_PATH = os.path.join("data", "features_master.csv")
MODEL_DIR = os.path.join("models")

# Feature columns used for prediction (8 features).
FEATURE_COLS = [
    "lat",
    "lon",
    "day_of_year",
    "tmax_lag1",
    "tmin_lag1",
    "rainfall_lag1",
    "rainfall_7d",
    "dtr",
]

# Target variables — one model will be trained per target.
TARGET_COLS = ["tmax", "tmin", "rainfall"]

# Model output filenames (one per target, same order as TARGET_COLS).
MODEL_FILENAMES = {
    "tmax": "model_tmax.joblib",
    "tmin": "model_tmin.joblib",
    "rainfall": "model_rain.joblib",
}

# Reproducibility seed
RANDOM_STATE = 42


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────


def load_and_clean(path: str) -> pd.DataFrame:
    """Load the CSV and drop rows with any NaN values."""
    print(f"[INFO] Loading dataset from {path} ...")
    df = pd.read_csv(path)
    initial_rows = len(df)
    df = df.dropna()
    dropped = initial_rows - len(df)
    print(f"[INFO] Loaded {initial_rows:,} rows — dropped {dropped:,} NaNs — "
          f"{len(df):,} rows remaining.")
    return df


def split_data(df: pd.DataFrame):
    """
    Separate features (X) and targets (Y), then perform an 80/20
    train-test split.

    Returns
    -------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.DataFrame   (multi-target)
    """
    X = df[FEATURE_COLS]
    Y = df[TARGET_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.20, random_state=RANDOM_STATE
    )

    print(f"[INFO] Train set: {len(X_train):,} samples | "
          f"Test set: {len(X_test):,} samples")
    return X_train, X_test, y_train, y_test


def train_and_evaluate(
    target: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor for a single target variable,
    print MAE and R2 on the test set, and return the fitted model.
    """
    print(f"\n{'-' * 50}")
    print(f"  Training model for: {target}")
    print(f"{'-' * 50}")

    model = RandomForestRegressor(
        n_estimators=15,     # fewer trees for smaller model size
        max_depth=10,        # shallower depth for web deployment
        n_jobs=-1,           # use all CPU cores for faster training
        random_state=RANDOM_STATE,
    )

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    # Predict on the held-out test set
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"  * Trained in {elapsed:.1f}s")
    print(f"  * MAE  = {mae:.4f}")
    print(f"  * R2   = {r2:.4f}")

    return model


def save_model(model: RandomForestRegressor, filename: str) -> None:
    """Persist a trained model to disk using joblib."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, path, compress=3)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  -> Saved to {path}  ({size_mb:.1f} MB)")


# ----------------------------------------------
# Main pipeline
# ----------------------------------------------


def main():
    # 1. Load and clean
    df = load_and_clean(DATA_PATH)

    # 2. Split into train / test
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Train, evaluate, and save one model per target
    for target in TARGET_COLS:
        model = train_and_evaluate(
            target,
            X_train,
            X_test,
            y_train[target],
            y_test[target],
        )
        save_model(model, MODEL_FILENAMES[target])

    print(f"\n{'=' * 50}")
    print("  All models trained and saved successfully!")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
