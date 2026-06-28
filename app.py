"""
app.py
======
Streamlit frontend for the 'Digital Twin of India's Climate' Proof of Concept.
Connects to the inference_engine.py backend to run real-time climate
predictions with interactive 'What-If' scenario controls.

Usage:
    streamlit run app.py
"""

import copy
import datetime
from typing import Dict, List

import ee
import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
from inference_engine import run_climate_simulation


# ──────────────────────────────────────────────────────────────────────
# City database with (lat, lon) coordinates
# ──────────────────────────────────────────────────────────────────────
CITIES = {
    "Delhi":      (28.50, 77.25),
    "Mumbai":     (19.00, 72.75),
    "Chennai":    (13.00, 80.25),
    "Kolkata":    (22.50, 88.50),
    "Bengaluru":  (13.00, 77.50),
    "Hyderabad":  (17.50, 78.50),
    "Jaipur":     (27.00, 75.75),
    "Custom Location": (None, None),
}

CITY_NAMES = list(CITIES.keys())


# ──────────────────────────────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────────────────────────────
if "city_select" not in st.session_state:
    st.session_state["city_select"] = "Delhi"
if "lat_input" not in st.session_state:
    st.session_state["lat_input"] = 28.50
if "lon_input" not in st.session_state:
    st.session_state["lon_input"] = 77.25
if "selected_month" not in st.session_state:
    st.session_state["selected_month"] = "June"
if "selected_day" not in st.session_state:
    st.session_state["selected_day"] = 1


# ──────────────────────────────────────────────────────────────────────
# Callbacks — bidirectional sync between city dropdown & coordinate inputs
# ──────────────────────────────────────────────────────────────────────
def on_city_change():
    """When a city is selected, update lat/lon to match."""
    city = st.session_state["city_select"]
    if city != "Custom Location":
        lat, lon = CITIES[city]
        st.session_state["lat_input"] = lat
        st.session_state["lon_input"] = lon


def on_coord_change():
    """When lat or lon is typed manually, update city to match or 'Custom'."""
    lat = st.session_state["lat_input"]
    lon = st.session_state["lon_input"]
    for city_name, (clat, clon) in CITIES.items():
        if city_name == "Custom Location":
            continue
        if clat is not None and abs(lat - clat) < 0.01 and abs(lon - clon) < 0.01:
            st.session_state["city_select"] = city_name
            return
    st.session_state["city_select"] = "Custom Location"


# ──────────────────────────────────────────────────────────────────────
# Helper functions (forecast logic lives in app.py — backend untouched)
# ──────────────────────────────────────────────────────────────────────

def get_weather_icon(rainfall: float, dtr: float) -> str:
    """Return an emoji weather icon based on rainfall intensity."""
    if rainfall > 5.0:
        return "🌧️"   # Heavy rain
    elif rainfall >= 0.5:
        return "🌦️"   # Light rain
    else:
        return "☀️"    # Clear / sunny


def generate_7_day_forecast(
    base_features: Dict[str, float],
    temp_anomaly: float = 0.0,
    rain_multiplier: float = 1.0,
    moisture_multiplier: float = 1.0,
) -> List[Dict[str, float]]:
    """
    Run a 7-step autoregressive forecast by feeding each day's
    predictions back as the next day's lag features.

    Uses the locked run_climate_simulation() API without modification.
    """
    forecast: List[Dict[str, float]] = []
    features = copy.deepcopy(base_features)

    for day in range(7):
        # Run prediction for the current day
        pred = run_climate_simulation(
            features,
            temp_anomaly=temp_anomaly,
            rain_multiplier=rain_multiplier,
            moisture_multiplier=moisture_multiplier,
        )

        # Compute DTR from the prediction and attach to result
        pred["dtr"] = round(pred["tmax"] - pred["tmin"], 2)
        pred["icon"] = get_weather_icon(pred["rainfall"], pred["dtr"])
        pred["day_of_year"] = features["day_of_year"]
        
        # Attach soil moisture for the UI
        current_sm = (base_features["soil_moisture_lag1"] * moisture_multiplier) if day == 0 else features["soil_moisture_lag1"]
        pred["soil_moisture"] = round(current_sm, 4)
        
        forecast.append(pred)

        # Save states that need to carry forward before we reset features
        old_7d = base_features["rainfall_7d"] if day == 0 else features["rainfall_7d"]

        # ── Prepare features for the NEXT day ────────────────────────
        features = copy.deepcopy(base_features)  # start from base coords
        features["day_of_year"] = base_features["day_of_year"] + day + 1
        features["tmax_lag1"]   = pred["tmax"]
        features["tmin_lag1"]   = pred["tmin"]
        features["rainfall_lag1"] = pred["rainfall"]

        # Update rolling 7-day rainfall:
        # add today's predicted rainfall, subtract the oldest day's contribution
        features["rainfall_7d"] = round(
            old_7d + pred["rainfall"] - features.get("rainfall_lag1", 0.0), 2
        )
        features["dtr"] = round(pred["tmax"] - pred["tmin"], 2)

        # Update soil moisture for next day:
        # Heuristic: if rain predicted, moisture increases slightly;
        # otherwise decays by ~5% toward a dry baseline.
        if pred["rainfall"] > 1.0:
            # Rainfall adds moisture (capped at 0.50 — saturated soil)
            features["soil_moisture_lag1"] = round(
                min(0.50, current_sm + pred["rainfall"] * 0.002), 4
            )
        else:
            # Dry day: slight evaporative decay
            features["soil_moisture_lag1"] = round(current_sm * 0.95, 4)

    return forecast


# ──────────────────────────────────────────────────────────────────────
# Climatology lookup — realistic baseline lag features by city & season
# ──────────────────────────────────────────────────────────────────────
# Each city maps to a list of (season_name, day_start, day_end, tmax_lag1,
# tmin_lag1, rainfall_lag1, rainfall_7d, soil_moisture_lag1) tuples.
# Seasons are checked in order; the first matching range wins.
# Soil moisture values are realistic root-zone volumetric water content
# (m³/m³) based on ISRO RZSM climatology for each city/season.

CLIMATOLOGY = {
    "Delhi": [
        #           day_start  day_end  tmax   tmin   rain_lag  rain_7d  sm_lag1
        ("winter",   330,  59,  20.0,   7.0,   0.0,    0.5,    0.10),
        ("spring",    60, 119,  32.0,  17.0,   0.2,    1.5,    0.08),
        ("summer",   120, 180,  42.0,  28.0,   0.5,    3.0,    0.06),
        ("monsoon",  181, 273,  35.0,  26.0,   8.0,   55.0,    0.30),
        ("autumn",   274, 329,  32.0,  18.0,   0.5,    3.0,    0.15),
    ],
    "Mumbai": [
        ("winter",   330,  59,  33.0,  19.0,   0.0,    0.2,    0.12),
        ("spring",    60, 119,  34.0,  23.0,   0.0,    0.3,    0.10),
        ("summer",   120, 180,  34.0,  27.0,   1.0,    5.0,    0.14),
        ("monsoon",  181, 273,  31.0,  25.0,  20.0,  130.0,    0.40),
        ("autumn",   274, 329,  34.0,  23.0,   2.0,   10.0,    0.22),
    ],
    "Chennai": [
        ("winter",   330,  59,  29.0,  21.0,   3.0,   20.0,    0.28),
        ("spring",    60, 119,  33.0,  24.0,   0.5,    2.0,    0.15),
        ("summer",   120, 180,  38.0,  28.0,   0.3,    2.0,    0.12),
        ("monsoon",  181, 273,  35.0,  26.0,   3.0,   15.0,    0.25),
        ("autumn",   274, 329,  31.0,  23.0,   6.0,   40.0,    0.35),
    ],
    "Kolkata": [
        ("winter",   330,  59,  26.0,  13.0,   0.2,    1.0,    0.14),
        ("spring",    60, 119,  33.0,  22.0,   1.0,    5.0,    0.12),
        ("summer",   120, 180,  36.0,  27.0,   3.0,   15.0,    0.20),
        ("monsoon",  181, 273,  33.0,  26.0,  10.0,   70.0,    0.38),
        ("autumn",   274, 329,  31.0,  22.0,   1.5,    8.0,    0.20),
    ],
    "Bengaluru": [
        ("winter",   330,  59,  28.0,  16.0,   0.2,    1.0,    0.15),
        ("spring",    60, 119,  34.0,  20.0,   0.5,    3.0,    0.12),
        ("summer",   120, 180,  34.0,  21.0,   3.0,   18.0,    0.22),
        ("monsoon",  181, 273,  29.0,  20.0,   5.0,   30.0,    0.32),
        ("autumn",   274, 329,  28.0,  19.0,   2.0,   12.0,    0.20),
    ],
    "Hyderabad": [
        ("winter",   330,  59,  30.0,  15.0,   0.2,    1.0,    0.12),
        ("spring",    60, 119,  36.0,  22.0,   0.3,    1.5,    0.09),
        ("summer",   120, 180,  40.0,  27.0,   1.0,    5.0,    0.08),
        ("monsoon",  181, 273,  32.0,  23.0,   6.0,   40.0,    0.30),
        ("autumn",   274, 329,  31.0,  19.0,   1.5,    8.0,    0.16),
    ],
    "Jaipur": [
        ("winter",   330,  59,  22.0,   8.0,   0.0,    0.3,    0.08),
        ("spring",    60, 119,  34.0,  19.0,   0.1,    0.5,    0.06),
        ("summer",   120, 180,  42.0,  28.0,   0.5,    3.0,    0.05),
        ("monsoon",  181, 273,  34.0,  25.0,   6.0,   40.0,    0.25),
        ("autumn",   274, 329,  33.0,  19.0,   0.3,    2.0,    0.12),
    ],
}

# Fallback for Custom Location — all-India average by season
_DEFAULT_CLIM = [
    ("winter",   330,  59,  27.0,  14.0,   0.5,    3.0,    0.12),
    ("spring",    60, 119,  34.0,  21.0,   0.5,    3.0,    0.10),
    ("summer",   120, 180,  38.0,  26.0,   1.0,    6.0,    0.10),
    ("monsoon",  181, 273,  33.0,  24.0,   7.0,   45.0,    0.30),
    ("autumn",   274, 329,  31.0,  19.0,   1.5,    8.0,    0.16),
]


def _day_in_range(day: int, start: int, end: int) -> bool:
    """Check if *day* falls within [start, end], wrapping around 365→1."""
    if start <= end:
        return start <= day <= end
    # Wraps around year boundary (e.g. winter: 330-59)
    return day >= start or day <= end


def get_dynamic_baseline(
    city: str, day_of_year: int, lat: float, lon: float,
) -> Dict[str, float]:
    """Return a complete 9-feature dict with realistic lag values
    based on the selected city and time of year."""

    seasons = CLIMATOLOGY.get(city, _DEFAULT_CLIM)

    # Find the matching season
    tmax_lag = 33.0
    tmin_lag = 22.0
    rain_lag = 1.0
    rain_7d  = 5.0
    sm_lag   = 0.15

    for _name, day_start, day_end, tmax, tmin, rain_l, rain_7, sm in seasons:
        if _day_in_range(day_of_year, day_start, day_end):
            tmax_lag = tmax
            tmin_lag = tmin
            rain_lag = rain_l
            rain_7d  = rain_7
            sm_lag   = sm
            break

    return {
        "lat":                  lat,
        "lon":                  lon,
        "day_of_year":          day_of_year,
        "tmax_lag1":            tmax_lag,
        "tmin_lag1":            tmin_lag,
        "rainfall_lag1":        rain_lag,
        "rainfall_7d":          rain_7d,
        "dtr":                  round(tmax_lag - tmin_lag, 2),
        "soil_moisture_lag1":   sm_lag,
    }


# ──────────────────────────────────────────────────────────────────────
# 1. Page Configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISRO Climate Digital Twin",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Custom CSS — Premium Monochrome (Black / White / Grey)
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Font ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    *, html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main background ─────────────────────────────────────────── */
    .stApp {
        background: #050505;
    }

    /* ── Sidebar ─────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #0a0a0a;
        border-right: 1px solid #1a1a1a;
    }

    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #e0e0e0;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    section[data-testid="stSidebar"] .stMarkdown h4 {
        color: #aaa;
    }

    /* ── Metric cards ────────────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: #111;
        border: 1px solid #222;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: #444;
    }

    div[data-testid="stMetric"] label {
        color: #777 !important;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.75rem !important;
        letter-spacing: 0.08em;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #fff !important;
        font-weight: 700;
        font-size: 2rem !important;
    }

    div[data-testid="stMetricDelta"] svg {
        display: none;
    }

    /* ── Hero banner ─────────────────────────────────────────────── */
    .hero-banner {
        background: #0a0a0a;
        border: 1px solid #1a1a1a;
        border-radius: 16px;
        padding: 36px 44px;
        margin-bottom: 28px;
    }

    .hero-banner h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 6px;
        letter-spacing: -0.03em;
    }

    .hero-banner p {
        color: #777;
        font-size: 1.05rem;
        margin: 0;
        font-weight: 400;
    }

    /* ── Section headers ─────────────────────────────────────────── */
    .section-header {
        color: #e0e0e0;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 36px 0 16px 0;
        padding-bottom: 10px;
        border-bottom: 1px solid #1a1a1a;
        letter-spacing: -0.01em;
    }

    /* ── Hazard cards ────────────────────────────────────────────── */
    .hazard-card {
        background: #0a0a0a;
        border: 1px solid #1a1a1a;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 16px;
    }

    .hazard-card h4 {
        color: #e0e0e0;
        font-weight: 600;
        margin-bottom: 8px;
        font-size: 1rem;
    }

    .hazard-label {
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 10px;
    }

    .hazard-low    { color: #555; }
    .hazard-medium { color: #999; }
    .hazard-high   { color: #ccc; }
    .hazard-severe { color: #fff; }

    /* ── Progress bar overrides ───────────────────────────────────── */
    .stProgress > div > div {
        border-radius: 8px;
        height: 10px !important;
        background-color: #1a1a1a !important;
    }

    .stProgress > div > div > div {
        background-color: #fff !important;
        border-radius: 8px;
    }

    /* ── Slider thumb ────────────────────────────────────────────── */
    .stSlider > div > div > div > div {
        background-color: #fff !important;
    }

    /* ── Divider ─────────────────────────────────────────────────── */
    hr {
        border-color: #1a1a1a !important;
    }

    /* ── Scenario badge ──────────────────────────────────────────── */
    .scenario-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-baseline {
        background: #111;
        color: #777;
        border: 1px solid #333;
    }
    .badge-modified {
        background: #fff;
        color: #000;
        border: 1px solid #fff;
    }

    /* ── 7-Day Forecast — responsive flexbox ─────────────────────── */
    .forecast-strip-wrapper {
        background: #0a0a0a;
        border: 1px solid #1a1a1a;
        border-radius: 16px;
        padding: clamp(12px, 2vw, 24px) clamp(8px, 1.5vw, 16px);
    }

    .forecast-strip {
        display: flex;
        flex-wrap: wrap;
        gap: clamp(6px, 1.2vw, 14px);
        justify-content: center;
    }

    .forecast-card {
        flex: 1 1 clamp(85px, 12%, 150px);
        min-width: 85px;
        max-width: 160px;
        background: #111;
        border: 1px solid #222;
        border-radius: 12px;
        padding: clamp(10px, 1.5vw, 18px) clamp(6px, 1vw, 12px);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .forecast-card:hover {
        transform: translateY(-4px);
        border-color: #444;
    }

    .fc-label {
        color: #555;
        font-size: clamp(0.58rem, 0.9vw, 0.72rem);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 4px;
    }

    .fc-icon {
        font-size: clamp(1.4rem, 2.8vw, 2.4rem);
        line-height: 1.2;
        margin: 4px 0 6px 0;
    }

    .fc-temps {
        display: flex;
        justify-content: center;
        align-items: baseline;
        gap: 4px;
        margin-bottom: 4px;
        flex-wrap: nowrap;
    }

    .fc-tmax {
        color: #fff;
        font-weight: 700;
        font-size: clamp(0.78rem, 1.4vw, 1.05rem);
    }

    .fc-sep {
        color: #333;
        font-size: clamp(0.7rem, 1.1vw, 0.85rem);
    }

    .fc-tmin {
        color: #666;
        font-weight: 600;
        font-size: clamp(0.78rem, 1.4vw, 1.05rem);
    }

    .fc-rain {
        color: #555;
        font-size: clamp(0.55rem, 0.9vw, 0.72rem);
        font-weight: 500;
        margin-top: 4px;
    }

    .fc-sm {
        color: #777;
        font-size: clamp(0.55rem, 0.9vw, 0.72rem);
        font-weight: 600;
        margin-top: 2px;
    }

    /* ── Controls area ───────────────────────────────────────────── */
    .controls-wrapper {
        background: #0a0a0a;
        border: 1px solid #1a1a1a;
        border-radius: 12px;
        padding: 20px 28px 12px 28px;
        margin-bottom: 24px;
    }

    .controls-wrapper h3 {
        color: #888;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 12px;
    }

    /* ── Summary bar ─────────────────────────────────────────────── */
    .summary-bar {
        background: #0a0a0a;
        border: 1px solid #1a1a1a;
        border-radius: 10px;
        padding: 10px 20px;
        margin-bottom: 8px;
        color: #666;
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.01em;
    }

    .summary-bar strong {
        color: #ccc;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# 2. Sidebar — Scenario Controls
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 What-If Scenario Engine")
    st.caption("Adjust parameters to simulate climate scenarios and observe "
               "their downstream impact in real time.")

    st.markdown("---")

    # Temperature anomaly slider
    st.markdown("#### 🌡️ Temperature Anomaly")
    temp_anomaly = st.slider(
        "Shift applied to lag temperatures (°C)",
        min_value=-3.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
        format="%+.1f °C",
        help="Positive = warming scenario, Negative = cooling scenario. "
             "Applied to tmax_lag1 and tmin_lag1 before prediction.",
    )

    st.markdown("")

    # Rainfall multiplier slider
    st.markdown("#### 🌧️ Rainfall Multiplier")
    rain_multiplier = st.slider(
        "Scaling factor for lag & 7-day rainfall",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
        format="%.1fx",
        help="1.0 = baseline.  1.5 = 50 %% more rain.  0.5 = 50 %% less rain.",
    )
    pct = int(rain_multiplier * 100)
    st.caption(f"→ Rainfall scaled to **{pct}%** of baseline")

    st.markdown("")

    # Soil moisture multiplier slider
    st.markdown("#### 🌱 Soil Moisture Multiplier")
    moisture_multiplier = st.slider(
        "Scaling factor for soil moisture",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
        format="%.1fx",
        help="1.0 = baseline. 0.5 = drought (50%% less moisture). "
             "2.0 = saturated (2× moisture).",
    )
    sm_pct = int(moisture_multiplier * 100)
    st.caption(f"→ Soil moisture scaled to **{sm_pct}%** of baseline")

    st.markdown("---")

    # Scenario status indicator
    is_modified = (
        temp_anomaly != 0.0
        or rain_multiplier != 1.0
        or moisture_multiplier != 1.0
    )
    if is_modified:
        st.markdown(
            '<span class="scenario-badge badge-modified">⚡ MODIFIED SCENARIO</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="scenario-badge badge-baseline">● BASELINE</span>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.caption("🛰️ Models: RandomForest (15 trees, depth 10)")
    st.caption("📍 Grid: IMD 0.25° × 0.25°")
    st.caption("📅 Training data: IMD 2022–2024")


# ──────────────────────────────────────────────────────────────────────
# 3. Main Content — Hero Banner
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🛰️ ISRO Climate Digital Twin</h1>
    <p>Real-time climate prediction engine powered by IMD gridded data &amp;
    Random Forest models — explore 'What-If' scenarios for India's climate.</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# 3a. Location & Time Controls — Bidirectional City / Coordinate Picker
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="controls-wrapper"><h3>📍 Location &amp; Time</h3></div>',
            unsafe_allow_html=True)

ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])

with ctrl_col1:
    st.selectbox(
        "City",
        CITY_NAMES,
        key="city_select",
        on_change=on_city_change,
    )

with ctrl_col2:
    st.number_input(
        "Latitude (°N)",
        min_value=6.5,
        max_value=38.5,
        step=0.25,
        format="%.2f",
        key="lat_input",
        on_change=on_coord_change,
    )

with ctrl_col3:
    st.number_input(
        "Longitude (°E)",
        min_value=66.5,
        max_value=100.0,
        step=0.25,
        format="%.2f",
        key="lon_input",
        on_change=on_coord_change,
    )

# ── Date selection via Month + Day dropdowns ────────────────────────
_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_DAYS_IN_MONTH = {
    "January": 31, "February": 28, "March": 31, "April": 30,
    "May": 31, "June": 30, "July": 31, "August": 31,
    "September": 30, "October": 31, "November": 30, "December": 31,
}

date_col1, date_col2 = st.columns(2)
with date_col1:
    selected_month = st.selectbox(
        "📅 Month",
        _MONTH_NAMES,
        key="selected_month",
    )
with date_col2:
    max_day = _DAYS_IN_MONTH[st.session_state["selected_month"]]
    selected_day = st.selectbox(
        "📅 Day",
        list(range(1, max_day + 1)),
        key="selected_day",
    )

# Convert Month + Day → day_of_year integer for the model
_month_num = _MONTH_NAMES.index(st.session_state["selected_month"]) + 1
_day_num = st.session_state["selected_day"]
# Clamp day to valid range (handles month change leaving stale day)
_day_num = min(_day_num, _DAYS_IN_MONTH[st.session_state["selected_month"]])
day_val = datetime.datetime(2023, _month_num, _day_num).timetuple().tm_yday
date_label = f"{st.session_state['selected_month']} {_day_num}"
st.caption(f"Day {day_val} ≈ **{date_label}**")


# ──────────────────────────────────────────────────────────────────────
# 3b. Build the 8-feature dictionary from user inputs
# ──────────────────────────────────────────────────────────────────────
baseline_features = get_dynamic_baseline(
    city=st.session_state["city_select"],
    day_of_year=day_val,
    lat=st.session_state["lat_input"],
    lon=st.session_state["lon_input"],
)


# ──────────────────────────────────────────────────────────────────────
# 3c. Run predictions (baseline + scenario)
# ──────────────────────────────────────────────────────────────────────
baseline_pred = run_climate_simulation(baseline_features)
scenario_pred = run_climate_simulation(
    baseline_features,
    temp_anomaly=temp_anomaly,
    rain_multiplier=rain_multiplier,
    moisture_multiplier=moisture_multiplier,
)

# Calculate deltas
delta_tmax = round(scenario_pred["tmax"] - baseline_pred["tmax"], 2)
delta_tmin = round(scenario_pred["tmin"] - baseline_pred["tmin"], 2)
delta_rain = round(scenario_pred["rainfall"] - baseline_pred["rainfall"], 2)


# ──────────────────────────────────────────────────────────────────────
# 3d. Summary bar
# ──────────────────────────────────────────────────────────────────────
city_display = st.session_state["city_select"]
lat_val = st.session_state["lat_input"]
lon_val = st.session_state["lon_input"]
if city_display == "Custom Location":
    loc_str = f"{lat_val:.2f}°N, {lon_val:.2f}°E"
else:
    loc_str = f"{city_display} ({lat_val:.2f}°N, {lon_val:.2f}°E)"

st.markdown(
    f'<div class="summary-bar">'
    f'📍 <strong>{loc_str}</strong> &nbsp;·&nbsp; '
    f'📅 <strong>Day {day_val}</strong> (~{date_label}) &nbsp;·&nbsp; '
    f'🔧 <strong>{temp_anomaly:+.1f} °C</strong> | <strong>{rain_multiplier:.1f}×</strong> rain'
    f' | <strong>{moisture_multiplier:.1f}×</strong> moisture'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown("---")


# ──────────────────────────────────────────────────────────────────────
# 3e. Prediction metric cards
# ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📊 Climate Predictions</p>',
            unsafe_allow_html=True)

# ── Compute coupled soil moisture (mirrors inference_engine.py) ──────
# The inference engine applies: direct slider → rainfall coupling →
# temperature evaporation → floor at 0.  We replicate that here so the
# displayed card matches what the model actually received.
baseline_sm = baseline_features["soil_moisture_lag1"]
scenario_sm = baseline_sm * moisture_multiplier       # Step A: direct slider
scenario_sm *= rain_multiplier                        # Coupling: rainfall → soil
scenario_sm -= temp_anomaly * 0.05                    # Coupling: temp → soil evap
scenario_sm = round(max(0.0, scenario_sm), 4)         # Floor at 0
delta_sm = round(scenario_sm - baseline_sm, 4)

# ── Row 1: Temperature & Rainfall (3 equal columns) ─────────────────
pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    st.metric(
        label="🌡️ Max Temperature (Tmax)",
        value=f"{scenario_pred['tmax']} °C",
        delta=f"{delta_tmax:+.2f} °C" if delta_tmax != 0 else None,
        delta_color="inverse",  # red for increasing temp = bad
    )

with pred_col2:
    st.metric(
        label="❄️ Min Temperature (Tmin)",
        value=f"{scenario_pred['tmin']} °C",
        delta=f"{delta_tmin:+.2f} °C" if delta_tmin != 0 else None,
        delta_color="inverse",
    )

with pred_col3:
    st.metric(
        label="🌧️ Rainfall",
        value=f"{scenario_pred['rainfall']} mm/day",
        delta=f"{delta_rain:+.2f} mm" if delta_rain != 0 else None,
        delta_color="normal",
    )

# ── Row 2: Topsoil Moisture (full-width dedicated card) ─────────────
sm_pad_l, sm_center, sm_pad_r = st.columns([1, 2, 1])
with sm_center:
    st.metric(
        label="🌱 Topsoil Moisture Content",
        value=f"{scenario_sm} m³/m³",
        delta=f"{delta_sm:+.4f} m³/m³" if delta_sm != 0 else None,
        delta_color="normal",
    )

# Baseline reference note
if is_modified:
    st.caption(
        f"_Baseline predictions — "
        f"Tmax: {baseline_pred['tmax']} °C  |  "
        f"Tmin: {baseline_pred['tmin']} °C  |  "
        f"Rainfall: {baseline_pred['rainfall']} mm/day  |  "
        f"Topsoil Moisture: {baseline_sm} m³/m³_"
    )


# ──────────────────────────────────────────────────────────────────────
# 4. Downstream Risk Indicators
# ──────────────────────────────────────────────────────────────────────
st.markdown("")
st.markdown('<p class="section-header">⚠️ Downstream Hazard Projections</p>',
            unsafe_allow_html=True)

# ── Row 1: Heat Stress & Flood Hazard (2 columns) ───────────────────
haz_col1, haz_col2 = st.columns(2)


# ── 4a. Agriculture Heat Stress Index ────────────────────────────────
with haz_col1:
    tmax_val = scenario_pred["tmax"]

    # Heat stress ramps from 0 at 35°C to 100 at 48°C
    HEAT_FLOOR = 35.0
    HEAT_CEIL = 48.0
    if tmax_val <= HEAT_FLOOR:
        heat_index = 0.0
    elif tmax_val >= HEAT_CEIL:
        heat_index = 1.0
    else:
        heat_index = (tmax_val - HEAT_FLOOR) / (HEAT_CEIL - HEAT_FLOOR)

    heat_pct = int(heat_index * 100)

    # Severity label
    if heat_pct < 25:
        severity, css_class = "LOW", "hazard-low"
    elif heat_pct < 50:
        severity, css_class = "MODERATE", "hazard-medium"
    elif heat_pct < 75:
        severity, css_class = "HIGH", "hazard-high"
    else:
        severity, css_class = "SEVERE", "hazard-severe"

    card_html = f"""
    <div class="hazard-card">
        <h4>🌾 Agriculture Heat Stress Index</h4>
        <p class="hazard-label {css_class}">Risk Level: <strong>{severity}</strong> ({heat_pct}%)</p>
        <div style="width: 100%; background-color: #1a1a1a; border-radius: 8px; height: 10px; margin: 12px 0;">
            <div style="background-color: #fff; width: {heat_pct}%; height: 10px; border-radius: 8px;"></div>
        </div>
        <p style="color: #777; font-size: 0.85rem; margin-top: 10px; margin-bottom: 0; padding-right: 8px;">
            Predicted Tmax: <strong>{tmax_val} °C</strong> — Crop stress threshold: 35 °C.
            Severe yield loss expected above 45 °C.
        </p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# ── 4b. Flood Hazard Rating ─────────────────────────────────────────
with haz_col2:
    rain_val = scenario_pred["rainfall"]

    # Flood hazard ramps from 0 at 5 mm/day to 100 at 60 mm/day
    FLOOD_FLOOR = 5.0
    FLOOD_CEIL = 60.0
    if rain_val <= FLOOD_FLOOR:
        flood_index = 0.0
    elif rain_val >= FLOOD_CEIL:
        flood_index = 1.0
    else:
        flood_index = (rain_val - FLOOD_FLOOR) / (FLOOD_CEIL - FLOOD_FLOOR)

    flood_pct = int(flood_index * 100)

    # Severity label
    if flood_pct < 25:
        severity_f, css_class_f = "LOW", "hazard-low"
    elif flood_pct < 50:
        severity_f, css_class_f = "MODERATE", "hazard-medium"
    elif flood_pct < 75:
        severity_f, css_class_f = "HIGH", "hazard-high"
    else:
        severity_f, css_class_f = "SEVERE", "hazard-severe"

    card_html_f = f"""
    <div class="hazard-card">
        <h4>🌊 Flood Hazard Rating</h4>
        <p class="hazard-label {css_class_f}">Risk Level: <strong>{severity_f}</strong> ({flood_pct}%)</p>
        <div style="width: 100%; background-color: #1a1a1a; border-radius: 8px; height: 10px; margin: 12px 0;">
            <div style="background-color: #fff; width: {flood_pct}%; height: 10px; border-radius: 8px;"></div>
        </div>
        <p style="color: #777; font-size: 0.85rem; margin-top: 10px; margin-bottom: 0; padding-right: 8px;">
            Predicted Rainfall: <strong>{rain_val} mm/day</strong> — Flash flood threshold: 60 mm/day.
            Urban waterlogging begins above 30 mm/day.
        </p>
    </div>
    """
    st.markdown(card_html_f, unsafe_allow_html=True)


# ── Row 2: Agricultural Drought Risk (full-width dedicated card) ────
drought_pad_l, drought_center, drought_pad_r = st.columns([1, 3, 1])

with drought_center:
    # Gather values from model predictions and coupled features
    drought_tmax = scenario_pred["tmax"]
    drought_rain_7d = baseline_features["rainfall_7d"] * rain_multiplier
    drought_sm = scenario_sm                     # coupled value computed above
    drought_sm_baseline = baseline_sm            # original baseline soil moisture

    # Programmatic threshold logic evaluating model predictions:
    #   SEVERE (76-100%): soil critically low, no 7d rain, predicted tmax > 40 °C
    #   HIGH   (51-75%):  soil severely depleted, no 7d rain, predicted tmax > 35 °C
    #   MODERATE (26-50%): soil below baseline OR 7d rain < 5 mm
    #   LOW    (0-25%):   soil near baseline AND 7d rain > 10 mm
    if (drought_sm < drought_sm_baseline * 0.3
            and drought_rain_7d == 0
            and drought_tmax > 40.0):
        drought_pct = min(100, 76 + int(
            24 * min(1.0, (drought_tmax - 40.0) / 5.0
                     + (1.0 - drought_sm / max(drought_sm_baseline * 0.3, 0.001)))
        ))
        severity_d, css_class_d = "SEVERE", "hazard-severe"

    elif (drought_sm < drought_sm_baseline * 0.5
              and drought_rain_7d == 0
              and drought_tmax > 35.0):
        drought_pct = min(75, 51 + int(
            24 * min(1.0, (drought_tmax - 35.0) / 10.0)
        ))
        severity_d, css_class_d = "HIGH", "hazard-high"

    elif (drought_sm < drought_sm_baseline * 0.8
              or drought_rain_7d < 5.0):
        # Moderate: compute a sub-score within 26-50 range
        sm_score = max(0.0, 1.0 - drought_sm / max(drought_sm_baseline * 0.8, 0.001))
        rain_score = max(0.0, 1.0 - drought_rain_7d / 5.0) if drought_rain_7d < 5.0 else 0.0
        combined = min(1.0, (sm_score + rain_score) / 2.0)
        drought_pct = 26 + int(24 * combined)
        severity_d, css_class_d = "MODERATE", "hazard-medium"

    else:
        # LOW: soil near baseline AND rainfall_7d > 10 mm
        if drought_rain_7d > 10.0 and drought_sm >= drought_sm_baseline * 0.9:
            drought_pct = max(0, 5)     # very low risk
        else:
            drought_pct = max(0, min(25, 25 - int(
                25 * min(1.0, drought_rain_7d / 10.0)
            )))
        severity_d, css_class_d = "LOW", "hazard-low"

    card_html_d = f"""
    <div class="hazard-card">
        <h4>🏜️ Agricultural Drought Risk</h4>
        <p class="hazard-label {css_class_d}">Risk Level: <strong>{severity_d}</strong> ({drought_pct}%)</p>
        <div style="width: 100%; background-color: #1a1a1a; border-radius: 8px; height: 10px; margin: 12px 0;">
            <div style="background-color: #fff; width: {drought_pct}%; height: 10px; border-radius: 8px;"></div>
        </div>
        <p style="color: #777; font-size: 0.85rem; margin-top: 10px; margin-bottom: 0; padding-right: 8px;">
            Soil Moisture: <strong>{drought_sm:.4f} m³/m³</strong> (baseline: {drought_sm_baseline:.4f})
            &nbsp;·&nbsp; 7-Day Rain: <strong>{drought_rain_7d:.1f} mm</strong>
            &nbsp;·&nbsp; Predicted Tmax: <strong>{drought_tmax} °C</strong>
        </p>
    </div>
    """
    st.markdown(card_html_d, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# 5. 7-Day Autonomous Forecast — Responsive Flexbox Cards
# ──────────────────────────────────────────────────────────────────────
st.markdown("")
st.markdown('<p class="section-header">📅 7-Day Autonomous Forecast</p>',
            unsafe_allow_html=True)
st.caption(
    "Autoregressive simulation — each day's prediction feeds into the next "
    "day's input features.  Scenario adjustments are applied at every step."
)

# Generate the forecast
forecast = generate_7_day_forecast(
    base_features=baseline_features,
    temp_anomaly=temp_anomaly,
    rain_multiplier=rain_multiplier,
    moisture_multiplier=moisture_multiplier,
)

# Build a single responsive HTML block (no st.columns — pure CSS flexbox)
forecast_cards = []
for i, day_pred in enumerate(forecast):
    forecast_day = day_val + i
    try:
        fc_date = datetime.datetime(2023, 1, 1) + datetime.timedelta(days=forecast_day - 1)
        fc_date_str = fc_date.strftime("%b %d")
    except Exception:
        fc_date_str = f"Day {forecast_day}"
    card = (
        '<div class="forecast-card">'
        f'<div class="fc-label">{fc_date_str}</div>'
        f'<div class="fc-icon">{day_pred["icon"]}</div>'
        '<div class="fc-temps">'
        f'<span class="fc-tmax">{day_pred["tmax"]}°</span>'
        '<span class="fc-sep">/</span>'
        f'<span class="fc-tmin">{day_pred["tmin"]}°</span>'
        '</div>'
        f'<div class="fc-rain">💧 {day_pred["rainfall"]} mm</div>'
        f'<div class="fc-sm">🌱 {day_pred["soil_moisture"]:.3f} m³/m³</div>'
        '</div>'
    )
    forecast_cards.append(card)

forecast_html = (
    '<div class="forecast-strip-wrapper">'
    '<div class="forecast-strip">'
    + "".join(forecast_cards)
    + '</div></div>'
)
st.markdown(forecast_html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# 6. Geospatial Climate Dashboard — Interactive PyDeck Map
# ──────────────────────────────────────────────────────────────────────
st.markdown("")
st.markdown('<p class="section-header">🗺️ Geospatial Climate Dashboard</p>',
            unsafe_allow_html=True)
st.caption(
    "Interactive spatial heatmap of climate predictions across India. "
    "Select a region and climate variable to explore predicted spatial patterns."
)

# ── Map control selectors ────────────────────────────────────────────
GEO_REGIONS = {
    "All India":  {"center": [22.0, 80.0], "zoom": 4},
    "Delhi":      {"center": [28.50, 77.25], "zoom": 10},
    "Mumbai":     {"center": [19.00, 72.75], "zoom": 10},
    "Chennai":    {"center": [13.00, 80.25], "zoom": 10},
    "Kolkata":    {"center": [22.50, 88.50], "zoom": 10},
    "Bengaluru":  {"center": [13.00, 77.50], "zoom": 10},
}

CLIMATE_FACTORS = ["Max Temperature", "Rainfall", "Topsoil Moisture"]

geo_ctrl1, geo_ctrl2 = st.columns(2)
with geo_ctrl1:
    geo_region = st.selectbox(
        "🗺️ Region",
        list(GEO_REGIONS.keys()),
        key="geo_region_select",
    )
with geo_ctrl2:
    climate_factor = st.selectbox(
        "📊 Climate Factor",
        CLIMATE_FACTORS,
        key="climate_factor_select",
    )

import datetime
map_time_ctrl1, map_time_ctrl2 = st.columns(2)
with map_time_ctrl1:
    _month_num = _MONTH_NAMES.index(st.session_state.get("selected_month", "January")) + 1
    _day_num = st.session_state.get("selected_day", 1)
    # Default to the most recent year in the dataset for safety
    default_map_date = datetime.date(2024, _month_num, _day_num)
    
    map_date = st.date_input(
        "📅 Map Date (One-Way Sync)",
        value=default_map_date,
        key="map_date"
    )
with map_time_ctrl2:
    time_scale = st.selectbox(
        "⏳ Time Scale",
        ["Daily", "Monthly", "Annual"],
        key="time_scale"
    )


# ── Generate grid of predictions (cached for performance) ───────────
import pandas as pd
from inference_engine import _load_models, FEATURE_COLS

@st.cache_data(show_spinner="Loading full master dataset (cached)…")
def load_full_spatial_data():
    df = pd.read_csv(
        "data/features_master.csv", 
        usecols=['time', 'lat', 'lon', 'day_of_year', 'dtr', 'rainfall_7d', 'rainfall_lag1', 'tmax_lag1', 'tmin_lag1', 'soil_moisture_lag1']
    )
    df['time'] = pd.to_datetime(df['time'])
    return df

@st.cache_data(show_spinner="Computing spatial predictions…")
def generate_geo_predictions(
    region, map_date_val, time_scale_val, temp_anom, rain_mult, moist_mult,
):
    """
    Extracts a geographic slice of baseline features, aggregates temporally,
    and runs vectorized Random Forest inference across the grid.
    """
    df = load_full_spatial_data()
    
    # ── Temporal Aggregation ─────────────────────────────────────────
    if time_scale_val == "Daily":
        df = df[df['time'].dt.date == map_date_val].copy()
    elif time_scale_val == "Monthly":
        df = df[(df['time'].dt.year == map_date_val.year) & (df['time'].dt.month == map_date_val.month)].copy()
        df = df.groupby(['lat', 'lon']).agg({
            'tmax_lag1': 'mean',
            'tmin_lag1': 'mean',
            'soil_moisture_lag1': 'mean',
            'rainfall_lag1': 'sum',
            'rainfall_7d': 'sum',
            'day_of_year': 'first',
            'dtr': 'mean'
        }).reset_index()
    elif time_scale_val == "Annual":
        df = df[df['time'].dt.year == map_date_val.year].copy()
        df = df.groupby(['lat', 'lon']).agg({
            'tmax_lag1': 'mean',
            'tmin_lag1': 'mean',
            'soil_moisture_lag1': 'mean',
            'rainfall_lag1': 'sum',
            'rainfall_7d': 'sum',
            'day_of_year': 'first',
            'dtr': 'mean'
        }).reset_index()
    
    if region == "All India":
        # Downsample: every 4th IMD 0.25° point -> 1.0° step
        # Ensures DataFrame stays under ~1500 rows to prevent Streamlit memory crashes
        df = df[
            (df['lat'] % 1.0 == 0.25) & 
            (df['lon'] % 1.0 == 0.25)
        ].copy()
    else:
        # City bounding box: ±1.0° at full 0.25° resolution
        c_lat, c_lon = GEO_REGIONS[region]["center"]
        df = df[
            (df['lat'] >= c_lat - 1.0) & (df['lat'] <= c_lat + 1.25) &
            (df['lon'] >= c_lon - 1.0) & (df['lon'] <= c_lon + 1.25)
        ].copy()

    # ── 1. Apply 'What-If' scenario adjustments (Vectorized) ────────
    df["tmax_lag1"] += temp_anom
    df["tmin_lag1"] += temp_anom
    df["rainfall_lag1"] *= rain_mult
    df["rainfall_7d"] *= rain_mult
    df["soil_moisture_lag1"] *= moist_mult

    # ── 2. Physics Coupling Layer (Vectorized) ───────────────────────
    rain_delta = rain_mult - 1.0
    temp_coupling = -2.0 * rain_delta
    df["tmax_lag1"] += temp_coupling
    df["tmin_lag1"] += temp_coupling

    df["soil_moisture_lag1"] *= rain_mult
    df["soil_moisture_lag1"] -= temp_anom * 0.05
    df["soil_moisture_lag1"] = df["soil_moisture_lag1"].clip(lower=0.0)
    
    df["dtr"] = df["tmax_lag1"] - df["tmin_lag1"]

    # ── 3. Run Vectorized Inference ──────────────────────────────────
    df["day_of_year"] = map_date_val.timetuple().tm_yday
    
    models = _load_models()
    X = df[FEATURE_COLS]
    
    df["tmax"] = models["tmax"].predict(X)
    df["tmin"] = models["tmin"].predict(X)
    df["rainfall"] = models["rainfall"].predict(X).clip(min=0).round(2)
    df["soil_moisture"] = df["soil_moisture_lag1"].round(4)
    
    # Return as list of dicts to perfectly match the existing rendering loop
    return df[["lat", "lon", "tmax", "tmin", "rainfall", "soil_moisture"]].to_dict(orient="records")


geo_data = generate_geo_predictions(
    geo_region, map_date, time_scale, temp_anomaly, rain_multiplier, moisture_multiplier,
)


# ── Map value column + colour mapping ────────────────────────────────
FACTOR_COL = {
    "Max Temperature": "tmax",
    "Rainfall":        "rainfall",
    "Topsoil Moisture": "soil_moisture",
}
FACTOR_UNIT = {
    "Max Temperature": "°C",
    "Rainfall":        "mm/day",
    "Topsoil Moisture": "m³/m³",
}

val_key = FACTOR_COL[climate_factor]
values  = [pt[val_key] for pt in geo_data]
v_min   = min(values) if values else 0.0
v_max   = max(values) if values else 1.0
v_range = max(v_max - v_min, 0.001)          # avoid division by zero


# ── Build Earth Engine FeatureCollection ──────────────────────────────
try:
    ee.Initialize(project='isro-digital-twin')
except Exception:
    pass  # Already initialized

# Filter out any points with null/zero lat or lon that would render at
# (0,0) in the Gulf of Guinea, producing a misplaced blue blob.
clean_data = [
    pt for pt in geo_data
    if pt['lat'] and pt['lon']
    and abs(pt['lat']) > 0.01 and abs(pt['lon']) > 0.01
]

# Recalculate dynamic min/max from the CLEANED data
values  = [pt[val_key] for pt in clean_data]
v_min   = float(min(values)) if values else 0.0
v_max   = float(max(values)) if values else 1.0

# 3. Fallback for Min/Max
if v_min == v_max:
    v_max += 0.1

# 1. Explicit Column Mapping
column_map = {
    'Max Temperature': 'tmax',
    'Rainfall': 'rainfall',
    'Topsoil Moisture': 'soil_moisture'
}
feature_col = column_map[climate_factor]

# 2. Force Float Data Type (Crucial)
df_geo = pd.DataFrame(clean_data)
df_geo[feature_col] = df_geo[feature_col].astype(float)

features = []
for _, row in df_geo.iterrows():
    geom = ee.Geometry.Point([row['lon'], row['lat']])
    features.append(ee.Feature(geom, {
        feature_col: row[feature_col]
    }))
fc = ee.FeatureCollection(features)

# 3. Fix reduceToImage
# We intentionally omit .unmask() here because unmasking with 0 before 
# focal_mean dilutes the data with millions of zero-pixels, clamping the map to v_min.
base_img = fc.reduceToImage(
    properties=[feature_col], 
    reducer=ee.Reducer.first()
)

# Blend and smooth the grid points (adjust radius based on grid size)
# Increased radius to perfectly blend the sparse grid points across India
smoothed_img = base_img.focal_mean(radius=100000, units='meters')

# Re-mask to the exact India shapefile bounds so the ocean remains dark
india_shapefile = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'India'))
final_img = smoothed_img.clip(india_shapefile)

# ── Map Configuration and Rendering ──────────────────────────────────
if climate_factor == "Max Temperature":
    palette = ['#313695', '#74add1', '#e0f3f8', '#fee090', '#f46d43', '#a50026']
elif climate_factor == "Rainfall":
    palette = ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b']
else:
    palette = ['#feedde', '#fdbe85', '#fd8d3c', '#e6550d', '#a63603']

vis_params = {
    'min': v_min,
    'max': v_max,
    'palette': palette,
}

print(f"Data range: {v_min} to {v_max}")

view_cfg = GEO_REGIONS[geo_region]

# Helper to mimic geemap's addLayer for native Folium
def folium_add_ee_layer(self, ee_object, vis_params, name):
    try:
        if isinstance(ee_object, ee.FeatureCollection):
            ee_dict = ee_object.getMapId(vis_params)
        else:
            ee_dict = ee.Image(ee_object).getMapId(vis_params)
        
        folium.TileLayer(
            tiles=ee_dict['tile_fetcher'].url_format,
            attr='Google Earth Engine',
            name=name,
            overlay=True,
            control=True,
            opacity=0.7
        ).add_to(self)
    except Exception as e:
        print(f"EE rendering error: {e}")

folium.Map.addLayer = folium_add_ee_layer

# Initialize native Folium map
m = folium.Map(
    location=[view_cfg["center"][0], view_cfg["center"][1]],
    zoom_start=view_cfg["zoom"],
    tiles='CartoDB dark_matter',
)

# 3. Enforce Strict Layer Ordering
# Add the styled heatmap raster FIRST
m.addLayer(final_img, vis_params, f"{climate_factor} Heatmap")

# 2. Fix the Shapefile Outline (Remove the Gray Block)
# Draw an empty outline using ee.Image().paint()
empty_outline = ee.Image().byte().paint(featureCollection=india_shapefile, color=1, width=2)
# Add the transparent border outline SECOND (so it sits on top)
m.addLayer(empty_outline, {'palette': ['white']}, 'India Border')

# Interactivity Workaround: Since we are using st_folium (Folium) instead 
# of geemap's ipyleaflet, add_inspector() does not exist because EE tiles 
# are flat PNGs. We overlay invisible tooltips on the data points instead!
unit = FACTOR_UNIT[climate_factor]
for pt in clean_data:
    folium.CircleMarker(
        location=[pt['lat'], pt['lon']],
        radius=5,
        color=None,
        fill=True,
        fill_color="white",
        fill_opacity=0.0,  # Invisible hit-box
        tooltip=f"<b>{climate_factor}:</b> {pt[val_key]:.2f} {unit}"
    ).add_to(m)

# Render using st_folium
st_folium(m, width=800, height=500, returned_objects=[])


# ── Colour-scale legend (CSS gradient, updates with climate factor) ─
GRADIENT_CSS = {
    "Max Temperature":  "linear-gradient(to right, #313695, #74add1, #e0f3f8, #fee090, #f46d43, #a50026)",
    "Rainfall":         "linear-gradient(to right, #f7fbff, #c6dbef, #6baed6, #2171b5, #08306b)",
    "Topsoil Moisture": "linear-gradient(to right, #feedde, #fdbe85, #fd8d3c, #e6550d, #a63603)",
}

unit = FACTOR_UNIT[climate_factor]
legend_html = f"""
<div style="
    background: #0a0a0a;
    border: 1px solid #1a1a1a;
    border-radius: 12px;
    padding: 18px 28px;
    margin-top: 12px;
">
    <p style="color: #888; font-size: 0.78rem; font-weight: 600;
              text-transform: uppercase; letter-spacing: 0.1em;
              margin-bottom: 10px;">
        {climate_factor} — Colour Scale
    </p>
    <div style="
        height: 14px;
        border-radius: 7px;
        background: {GRADIENT_CSS[climate_factor]};
        margin-bottom: 8px;
    "></div>
    <div style="display: flex; justify-content: space-between;
                color: #777; font-size: 0.8rem; font-weight: 500;">
        <span>{v_min:.2f} {unit}</span>
        <span>{((v_min + v_max) / 2):.2f} {unit}</span>
        <span>{v_max:.2f} {unit}</span>
    </div>
</div>
"""
st.markdown(legend_html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#444; font-size:0.8rem; "
    "padding: 12px 0;'>"
    "🛰️ <strong>ISRO Climate Digital Twin</strong> — Proof of Concept  ·  "
    "Data: IMD (2022–2024) + ISRO Soil Moisture (RZSM)  ·  "
    "Models: Random Forest Regressor (scikit-learn)"
    "</div>",
    unsafe_allow_html=True,
)

