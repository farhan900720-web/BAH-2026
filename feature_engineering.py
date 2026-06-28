"""
feature_engineering.py
=====================
Engineers predictive features from merged IMD + ISRO climate data.

Input:  A DataFrame with columns [lat, lon, day_of_year, tmax, tmin, rainfall,
        soil_moisture]
Output: A feature-enriched DataFrame ready for ML modelling.

Features created
----------------
1. rainfall_7d         : 7-day rolling sum of rainfall (soil saturation proxy)
2. dtr                 : Diurnal temperature range (tmax - tmin)
3. rainfall_lag1       : Previous day's rainfall
4. tmax_lag1           : Previous day's max temperature
5. tmin_lag1           : Previous day's min temperature
6. soil_moisture_lag1  : Previous day's root-zone soil moisture

All rolling/lag operations are grouped by (lat, lon) to prevent
cross-coordinate data leakage.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core feature-engineering function
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add climate-derived predictive features to *df* (per coordinate).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``lat``, ``lon``, ``day_of_year``,
        ``tmax``, ``tmin``, ``rainfall``, ``soil_moisture``.
        Rows should be sorted by ``(lat, lon, day_of_year)`` or by a
        time-like ordering within each coordinate group.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with additional columns and NaN rows dropped:
        - ``rainfall_7d``         : 7-day rolling sum of rainfall
        - ``dtr``                 : tmax - tmin
        - ``rainfall_lag1``       : rainfall shifted by +1 day
        - ``tmax_lag1``           : tmax shifted by +1 day
        - ``tmin_lag1``           : tmin shifted by +1 day
        - ``soil_moisture_lag1``  : soil_moisture shifted by +1 day
    """

    required_cols = {"lat", "lon", "day_of_year", "tmax", "tmin", "rainfall", "soil_moisture"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    # Work on a copy so the caller's data is untouched
    df = df.copy()

    # Ensure deterministic ordering within each coordinate
    df.sort_values(["lat", "lon", "day_of_year"], inplace=True)

    # ------------------------------------------------------------------
    # 1. Diurnal Temperature Range (no grouping needed — row-wise)
    # ------------------------------------------------------------------
    df["dtr"] = df["tmax"] - df["tmin"]

    # ------------------------------------------------------------------
    # 2. Group by coordinate for rolling / lag features
    # ------------------------------------------------------------------
    grouped = df.groupby(["lat", "lon"], sort=False)

    # 7-day rolling sum of rainfall  (min_periods=1 avoids leading NaNs)
    df["rainfall_7d"] = grouped["rainfall"].transform(
        lambda s: s.rolling(window=7, min_periods=1).sum()
    )

    # Lagged features (shift=1 → previous day)
    df["rainfall_lag1"]       = grouped["rainfall"].transform(lambda s: s.shift(1))
    df["tmax_lag1"]           = grouped["tmax"].transform(lambda s: s.shift(1))
    df["tmin_lag1"]           = grouped["tmin"].transform(lambda s: s.shift(1))
    df["soil_moisture_lag1"]  = grouped["soil_moisture"].transform(lambda s: s.shift(1))

    # ------------------------------------------------------------------
    # 3. Drop rows still containing NaNs (first day of each coordinate
    #    will have NaN lags; coordinates with missing source data, etc.)
    # ------------------------------------------------------------------
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    print(f"Dropped {before - after:,} NaN rows ({before:,} -> {after:,})")

    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Helper: convert a merged xarray Dataset into the flat DataFrame expected
# by  engineer_features()
# ---------------------------------------------------------------------------
def dataset_to_dataframe(ds) -> pd.DataFrame:
    """Flatten an xarray Dataset (time, lat, lon) into a Pandas DataFrame.

    Adds a ``day_of_year`` column derived from the time coordinate.
    Drops rows where *all* data variables are NaN (ocean pixels, etc.).
    """
    df = ds.to_dataframe().reset_index()

    # Add day_of_year from the time coordinate
    df["day_of_year"] = df["time"].dt.dayofyear

    # Drop pixels where every climate variable is missing (ocean / border)
    climate_vars = [c for c in ["tmax", "tmin", "rainfall", "soil_moisture"] if c in df.columns]
    df.dropna(subset=climate_vars, how="all", inplace=True)

    return df


# ---------------------------------------------------------------------------
# Execution block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import xarray as xr

    parser = argparse.ArgumentParser(description="Engineer features for a given year.")
    parser.add_argument("year", type=int, help="Year to process (e.g. 2022, 2023, 2024)")
    args = parser.parse_args()

    YEAR   = args.year
    INPUT  = f"data/merged_climate_{YEAR}.nc"
    OUTPUT = f"data/features_{YEAR}.csv"

    # ── Load merged dataset ───────────────────────────────────────────
    print(f"Loading {INPUT} ...")
    ds = xr.open_dataset(INPUT)
    print(f"  Dimensions: {dict(ds.sizes)}")
    print(f"  Variables:  {list(ds.data_vars)}\n")

    # ── Convert to DataFrame ──────────────────────────────────────────
    print("Flattening to DataFrame ...")
    df = dataset_to_dataframe(ds)
    print(f"  Rows after dropping all-NaN pixels: {len(df):,}\n")

    # ── Engineer features ─────────────────────────────────────────────
    print("Engineering features ...")
    df = engineer_features(df)

    print(f"\nFinal DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    print(df.head(10).to_string(index=False))

    # ── Summary statistics ────────────────────────────────────────────
    feature_cols = ["rainfall_7d", "dtr", "rainfall_lag1", "tmax_lag1", "tmin_lag1", "soil_moisture_lag1"]
    print("\n--- Feature Statistics ---")
    print(df[feature_cols].describe().to_string())

    # ── Save ──────────────────────────────────────────────────────────
    print(f"\nSaving to {OUTPUT} ...")
    df.to_csv(OUTPUT, index=False)
    print(f"[OK] Saved -> {OUTPUT}  ({len(df):,} rows)")
