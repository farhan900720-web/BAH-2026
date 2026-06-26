"""
inference_engine.py
===================
Loads the three optimised Random Forest models (tmax, tmin, rainfall) and
exposes a single function for running climate predictions with optional
'What-If' scenario adjustments.

Usage (as a module):
    from inference_engine import run_climate_simulation

    prediction = run_climate_simulation(
        current_features={
            "lat": 28.5, "lon": 77.0, "day_of_year": 150,
            "tmax_lag1": 42.0, "tmin_lag1": 25.0,
            "rainfall_lag1": 0.0, "rainfall_7d": 2.5, "dtr": 12.0,
        },
        temp_anomaly=2.0,      # +2 °C warming scenario
        rain_multiplier=1.3,   # 30 % more rainfall
    )
    print(prediction)
    # {'tmax': 43.1, 'tmin': 27.5, 'rainfall': 4.2}

Usage (standalone):
    python inference_engine.py
"""

import os
from typing import Dict

import joblib
import pandas as pd


# ──────────────────────────────────────────────
# Model paths
# ──────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

_MODEL_PATHS = {
    "tmax":     os.path.join(MODEL_DIR, "model_tmax.joblib"),
    "tmin":     os.path.join(MODEL_DIR, "model_tmin.joblib"),
    "rainfall": os.path.join(MODEL_DIR, "model_rain.joblib"),
}

# Lazy-loaded model cache (loaded once, reused across calls)
_models: Dict[str, object] = {}

# Ordered feature columns expected by the models
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


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _load_models() -> Dict[str, object]:
    """Load all three models from disk (once) and cache them."""
    if not _models:
        for name, path in _MODEL_PATHS.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Model file not found: {path}. "
                    "Run  python models/train_models.py  first."
                )
            _models[name] = joblib.load(path)
            print(f"[INFO] Loaded {name} model from {path}")
    return _models


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def run_climate_simulation(
    current_features: Dict[str, float],
    temp_anomaly: float = 0.0,
    rain_multiplier: float = 1.0,
) -> Dict[str, float]:
    """
    Run a climate prediction with optional 'What-If' adjustments.

    Parameters
    ----------
    current_features : dict
        Must contain keys: lat, lon, day_of_year, tmax_lag1, tmin_lag1,
        rainfall_lag1, rainfall_7d, dtr.
    temp_anomaly : float, optional
        Value (°C) added to tmax_lag1 and tmin_lag1 to simulate a warming
        or cooling scenario.  Default 0.0 (no change).
    rain_multiplier : float, optional
        Multiplicative factor applied to rainfall_lag1 and rainfall_7d
        to simulate wetter (>1) or drier (<1) scenarios.
        Default 1.0 (no change).

    Returns
    -------
    dict
        Predicted values: ``{"tmax": float, "tmin": float, "rainfall": float}``
    """

    # ── 1. Validate required keys ────────────────────────────────────
    missing = set(FEATURE_COLS) - set(current_features)
    if missing:
        raise ValueError(f"Missing feature keys: {missing}")

    # ── 2. Copy features so the caller's dict is untouched ───────────
    features = dict(current_features)

    # ── 3. Apply 'What-If' scenario adjustments ──────────────────────
    #   Temperature anomaly → shift lag temperatures
    features["tmax_lag1"] += temp_anomaly
    features["tmin_lag1"] += temp_anomaly

    #   Rainfall multiplier → scale lag and rolling rainfall
    features["rainfall_lag1"] *= rain_multiplier
    features["rainfall_7d"]  *= rain_multiplier

    #   Recalculate DTR from the adjusted lag temperatures
    features["dtr"] = features["tmax_lag1"] - features["tmin_lag1"]

    # ── 4. Build a single-row DataFrame in the correct column order ──
    df = pd.DataFrame([features])[FEATURE_COLS]

    # ── 5. Load models and predict ───────────────────────────────────
    models = _load_models()

    predictions = {
        "tmax":     round(float(models["tmax"].predict(df)[0]), 2),
        "tmin":     round(float(models["tmin"].predict(df)[0]), 2),
        "rainfall": round(float(models["rainfall"].predict(df)[0]), 2),
    }

    return predictions


# ──────────────────────────────────────────────
# Standalone demo
# ──────────────────────────────────────────────

if __name__ == "__main__":

    # Example: a grid point near Delhi on ~30 May
    sample_features = {
        "lat": 28.5,
        "lon": 77.0,
        "day_of_year": 150,
        "tmax_lag1": 42.0,
        "tmin_lag1": 27.0,
        "rainfall_lag1": 0.0,
        "rainfall_7d": 2.5,
        "dtr": 15.0,
    }

    # ── Baseline prediction (no scenario adjustments) ────────────────
    print("=" * 55)
    print("  BASELINE  (no adjustments)")
    print("=" * 55)
    baseline = run_climate_simulation(sample_features)
    print(f"  Predicted tmax     : {baseline['tmax']} °C")
    print(f"  Predicted tmin     : {baseline['tmin']} °C")
    print(f"  Predicted rainfall : {baseline['rainfall']} mm/day")

    # ── What-If: +2 °C warming ───────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("  SCENARIO  (+2 °C warming)")
    print("=" * 55)
    warming = run_climate_simulation(sample_features, temp_anomaly=2.0)
    print(f"  Predicted tmax     : {warming['tmax']} °C")
    print(f"  Predicted tmin     : {warming['tmin']} °C")
    print(f"  Predicted rainfall : {warming['rainfall']} mm/day")

    # ── What-If: 50 % more rainfall ──────────────────────────────────
    print(f"\n{'=' * 55}")
    print("  SCENARIO  (50% more rainfall)")
    print("=" * 55)
    wetter = run_climate_simulation(sample_features, rain_multiplier=1.5)
    print(f"  Predicted tmax     : {wetter['tmax']} °C")
    print(f"  Predicted tmin     : {wetter['tmin']} °C")
    print(f"  Predicted rainfall : {wetter['rainfall']} mm/day")

    # ── What-If: Combined (+2 °C AND 50 % more rain) ────────────────
    print(f"\n{'=' * 55}")
    print("  SCENARIO  (+2 °C warming + 50% more rainfall)")
    print("=" * 55)
    combined = run_climate_simulation(
        sample_features, temp_anomaly=2.0, rain_multiplier=1.5
    )
    print(f"  Predicted tmax     : {combined['tmax']} °C")
    print(f"  Predicted tmin     : {combined['tmin']} °C")
    print(f"  Predicted rainfall : {combined['rainfall']} mm/day")

    print(f"\n{'=' * 55}")
    print("  All scenarios completed successfully!")
    print(f"{'=' * 55}\n")
