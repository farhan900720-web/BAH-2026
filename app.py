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

import streamlit as st
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
        )

        # Compute DTR from the prediction and attach to result
        pred["dtr"] = round(pred["tmax"] - pred["tmin"], 2)
        pred["icon"] = get_weather_icon(pred["rainfall"], pred["dtr"])
        pred["day_of_year"] = features["day_of_year"]
        forecast.append(pred)

        # ── Prepare features for the NEXT day ────────────────────────
        features = copy.deepcopy(base_features)  # start from base coords
        features["day_of_year"] = base_features["day_of_year"] + day + 1
        features["tmax_lag1"]   = pred["tmax"]
        features["tmin_lag1"]   = pred["tmin"]
        features["rainfall_lag1"] = pred["rainfall"]

        # Update rolling 7-day rainfall:
        # add today's predicted rainfall, subtract the oldest day's contribution
        prev_7d = base_features["rainfall_7d"] if day == 0 else features["rainfall_7d"]
        features["rainfall_7d"] = round(
            prev_7d + pred["rainfall"] - features.get("rainfall_lag1", 0.0), 2
        )
        features["dtr"] = round(pred["tmax"] - pred["tmin"], 2)

    return forecast


# ──────────────────────────────────────────────────────────────────────
# Climatology lookup — realistic baseline lag features by city & season
# ──────────────────────────────────────────────────────────────────────
# Each city maps to a list of (day_start, day_end, tmax_lag1, tmin_lag1,
# rainfall_lag1, rainfall_7d) tuples.  Seasons are checked in order;
# the first matching range wins.  Day ranges wrap around year boundary
# (e.g. winter = day 330-365 + 1-59).

CLIMATOLOGY = {
    "Delhi": [
        #           day_start  day_end  tmax   tmin   rain_lag  rain_7d
        ("winter",   330,  59,  20.0,   7.0,   0.0,    0.5),
        ("spring",    60, 119,  32.0,  17.0,   0.2,    1.5),
        ("summer",   120, 180,  42.0,  28.0,   0.5,    3.0),
        ("monsoon",  181, 273,  35.0,  26.0,   8.0,   55.0),
        ("autumn",   274, 329,  32.0,  18.0,   0.5,    3.0),
    ],
    "Mumbai": [
        ("winter",   330,  59,  33.0,  19.0,   0.0,    0.2),
        ("spring",    60, 119,  34.0,  23.0,   0.0,    0.3),
        ("summer",   120, 180,  34.0,  27.0,   1.0,    5.0),
        ("monsoon",  181, 273,  31.0,  25.0,  20.0,  130.0),
        ("autumn",   274, 329,  34.0,  23.0,   2.0,   10.0),
    ],
    "Chennai": [
        ("winter",   330,  59,  29.0,  21.0,   3.0,   20.0),
        ("spring",    60, 119,  33.0,  24.0,   0.5,    2.0),
        ("summer",   120, 180,  38.0,  28.0,   0.3,    2.0),
        ("monsoon",  181, 273,  35.0,  26.0,   3.0,   15.0),
        ("autumn",   274, 329,  31.0,  23.0,   6.0,   40.0),
    ],
    "Kolkata": [
        ("winter",   330,  59,  26.0,  13.0,   0.2,    1.0),
        ("spring",    60, 119,  33.0,  22.0,   1.0,    5.0),
        ("summer",   120, 180,  36.0,  27.0,   3.0,   15.0),
        ("monsoon",  181, 273,  33.0,  26.0,  10.0,   70.0),
        ("autumn",   274, 329,  31.0,  22.0,   1.5,    8.0),
    ],
    "Bengaluru": [
        ("winter",   330,  59,  28.0,  16.0,   0.2,    1.0),
        ("spring",    60, 119,  34.0,  20.0,   0.5,    3.0),
        ("summer",   120, 180,  34.0,  21.0,   3.0,   18.0),
        ("monsoon",  181, 273,  29.0,  20.0,   5.0,   30.0),
        ("autumn",   274, 329,  28.0,  19.0,   2.0,   12.0),
    ],
    "Hyderabad": [
        ("winter",   330,  59,  30.0,  15.0,   0.2,    1.0),
        ("spring",    60, 119,  36.0,  22.0,   0.3,    1.5),
        ("summer",   120, 180,  40.0,  27.0,   1.0,    5.0),
        ("monsoon",  181, 273,  32.0,  23.0,   6.0,   40.0),
        ("autumn",   274, 329,  31.0,  19.0,   1.5,    8.0),
    ],
    "Jaipur": [
        ("winter",   330,  59,  22.0,   8.0,   0.0,    0.3),
        ("spring",    60, 119,  34.0,  19.0,   0.1,    0.5),
        ("summer",   120, 180,  42.0,  28.0,   0.5,    3.0),
        ("monsoon",  181, 273,  34.0,  25.0,   6.0,   40.0),
        ("autumn",   274, 329,  33.0,  19.0,   0.3,    2.0),
    ],
}

# Fallback for Custom Location — all-India average by season
_DEFAULT_CLIM = [
    ("winter",   330,  59,  27.0,  14.0,   0.5,    3.0),
    ("spring",    60, 119,  34.0,  21.0,   0.5,    3.0),
    ("summer",   120, 180,  38.0,  26.0,   1.0,    6.0),
    ("monsoon",  181, 273,  33.0,  24.0,   7.0,   45.0),
    ("autumn",   274, 329,  31.0,  19.0,   1.5,    8.0),
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
    """Return a complete 8-feature dict with realistic lag values
    based on the selected city and time of year."""

    seasons = CLIMATOLOGY.get(city, _DEFAULT_CLIM)

    # Find the matching season
    tmax_lag = 33.0
    tmin_lag = 22.0
    rain_lag = 1.0
    rain_7d  = 5.0

    for _name, day_start, day_end, tmax, tmin, rain_l, rain_7 in seasons:
        if _day_in_range(day_of_year, day_start, day_end):
            tmax_lag = tmax
            tmin_lag = tmin
            rain_lag = rain_l
            rain_7d  = rain_7
            break

    return {
        "lat":           lat,
        "lon":           lon,
        "day_of_year":   day_of_year,
        "tmax_lag1":     tmax_lag,
        "tmin_lag1":     tmin_lag,
        "rainfall_lag1": rain_lag,
        "rainfall_7d":   rain_7d,
        "dtr":           round(tmax_lag - tmin_lag, 2),
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

    st.markdown("---")

    # Scenario status indicator
    is_modified = temp_anomaly != 0.0 or rain_multiplier != 1.0
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
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown("---")


# ──────────────────────────────────────────────────────────────────────
# 3e. Prediction metric cards
# ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📊 Climate Predictions</p>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="🌡️ Max Temperature (Tmax)",
        value=f"{scenario_pred['tmax']} °C",
        delta=f"{delta_tmax:+.2f} °C" if delta_tmax != 0 else None,
        delta_color="inverse",  # red for increasing temp = bad
    )

with col2:
    st.metric(
        label="❄️ Min Temperature (Tmin)",
        value=f"{scenario_pred['tmin']} °C",
        delta=f"{delta_tmin:+.2f} °C" if delta_tmin != 0 else None,
        delta_color="inverse",
    )

with col3:
    st.metric(
        label="🌧️ Rainfall",
        value=f"{scenario_pred['rainfall']} mm/day",
        delta=f"{delta_rain:+.2f} mm" if delta_rain != 0 else None,
        delta_color="normal",
    )

# Baseline reference note
if is_modified:
    st.caption(
        f"_Baseline predictions — "
        f"Tmax: {baseline_pred['tmax']} °C  |  "
        f"Tmin: {baseline_pred['tmin']} °C  |  "
        f"Rainfall: {baseline_pred['rainfall']} mm/day_"
    )


# ──────────────────────────────────────────────────────────────────────
# 4. Downstream Risk Indicators
# ──────────────────────────────────────────────────────────────────────
st.markdown("")
st.markdown('<p class="section-header">⚠️ Downstream Hazard Projections</p>',
            unsafe_allow_html=True)

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
        <p style="color: #777; font-size: 0.85rem; margin-top: 10px; margin-bottom: 0;">
            Predicted Tmax: <strong>{tmax_val} °C</strong> — Crop stress threshold: 35 °C. Severe yield loss expected above 45 °C.
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
        <p style="color: #777; font-size: 0.85rem; margin-top: 10px; margin-bottom: 0;">
            Predicted Rainfall: <strong>{rain_val} mm/day</strong> — Flash flood threshold: 60 mm/day. Urban waterlogging begins above 30 mm/day.
        </p>
    </div>
    """
    st.markdown(card_html_f, unsafe_allow_html=True)


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
# Footer
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#444; font-size:0.8rem; "
    "padding: 12px 0;'>"
    "🛰️ <strong>ISRO Climate Digital Twin</strong> — Proof of Concept  ·  "
    "Data: India Meteorological Department (IMD 2022–2024)  ·  "
    "Models: Random Forest Regressor (scikit-learn)"
    "</div>",
    unsafe_allow_html=True,
)
