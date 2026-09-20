"""
Inbound Flight Reliability -- decision-support dashboard.

Two tabs:
  1. Dashboard        -- per-route model results (global, then per-airline).
  2. Live Predictor   -- today's Open-Meteo forecast fed into the trained
                         models for a live buffer recommendation.

Reads the artifacts written by src/train_model.py (models/<route>/tier_manifest.json
+ per-tier/per-quantile model + preprocessing files) and src/features.py
(data/processed/features.parquet). 
"""

import datetime as dt
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# Allow imports from project root when running:
# streamlit run app/streamlit_app.py
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import config
from src.features import WEATHER_FEATURES
from src.airlines import airline_label
from src.fetch_weather import fetch_live_weather, nearest_hour_weather, LIVE_WEATHER_URL


st.set_page_config(
    page_title="Inbound Flight Reliability",
    page_icon="✈️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data source (local checkout, or synced from the HF dataset repo in
# deployment -- see docker-compose.yml / DATA_SOURCE env var)
# ---------------------------------------------------------------------------

DATA_SOURCE = os.getenv("DATA_SOURCE", "local").lower()


@st.cache_resource
def sync_from_hf():
    from src.hf_storage import download_from_hf

    return download_from_hf(local_dir="data_cache")


DATA_SYNC_ERROR = None  

if DATA_SOURCE == "hf":  
    try:  
        DATA_ROOT = sync_from_hf()  
    except Exception as exc:  
        DATA_ROOT = PROJECT_ROOT  
        DATA_SYNC_ERROR = str(exc)  
else:
    DATA_ROOT = PROJECT_ROOT

if DATA_SYNC_ERROR:  
    st.error(  
        "Couldn't load data from Hugging Face.\n\n"
        f"**{DATA_SYNC_ERROR}**\n\n"
        "This usually means `HF_TOKEN` and/or `DATA_SOURCE=hf` aren't set "
        "in this app's own **Settings → Secrets** on Streamlit Cloud "
        "(a separate secrets store from GitHub Actions)."
    )  
    st.stop()  


# ---------------------------------------------------------------------------
# Route configuration
# ---------------------------------------------------------------------------

def route_options():
    """[(origin, destination), ...] straight from config -- first entry is
    the default selection everywhere."""
    return list(config.ROUTES)


def route_key(origin, destination):
    """Matches the "->" convention written into features.parquet's "route"
    column by src.fetch_opensky.build_route_dataset (`df["route"] =
    f"{origin}->{destination}"`)."""
    return f"{origin}->{destination}"


def route_label(origin, destination):
    return f"{origin} \u2192 {destination}"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def route_model_dir(origin, destination):
    return os.path.join(
        DATA_ROOT,
        "models",
        config.route_key(origin, destination),
    )


def tier_manifest_path(origin, destination):
    return os.path.join(
        route_model_dir(origin, destination),
        "tier_manifest.json",
    )


def _absolute_model_path(path):
    if not path:
        return None

    if os.path.isabs(path):
        return path

    return os.path.join(DATA_ROOT, path)


def features_path():
    return os.path.join(DATA_ROOT, config.PROCESSED_DIR, "features.parquet")


# ---------------------------------------------------------------------------
# Generic loaders
# ---------------------------------------------------------------------------

def load_json(path):
    if not path or not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_tier_manifest(origin, destination):
    return load_json(tier_manifest_path(origin, destination))


@st.cache_data
def load_features():
    path = features_path()

    if not os.path.exists(path):
        return None

    return pd.read_parquet(path)



@st.cache_resource
def load_tier_models(origin, destination):

    """
    Load and cache XGBoost regression models and preprocessing pipelines for a route.

    Parameters:
        origin (str): Departure airport ICAO code (e.g., "EDDF").
        destination (str): Arrival airport ICAO code (e.g., "LGTS").

    Returns:
        dict: A nested structure mapping tiers to quantiles and their model assets:
            {
                tier_name (str): {
                    quantile (float): {
                        "model": xgb.XGBRegressor,
                        "preprocessing": dict
                    }
                }
            }
            Where tier_name is "all_carriers" or an airline code (e.g., "AEE").
            Returns {} if the route manifest or model files are missing.
    """

    manifest = load_tier_manifest(origin, destination)

    if not manifest:
        return {}

    tiers = {"all_carriers": manifest.get("all_carriers")}
    tiers.update(manifest.get("by_carrier", {}).get("carriers", {}))

    loaded = {}

    for tier, entry in tiers.items():
        if not isinstance(entry, dict) or not entry.get("trained"):
            continue

        per_quantile = {}

        for q_str, q_entry in entry.get("quantiles", {}).items():
            production = q_entry.get("production", {}) if isinstance(q_entry, dict) else {}

            model_file = _absolute_model_path(production.get("model_path"))
            preprocessing_file = _absolute_model_path(production.get("preprocessing_path"))

            if not model_file or not os.path.exists(model_file):
                continue

            if not preprocessing_file or not os.path.exists(preprocessing_file):
                continue

            model = xgb.XGBRegressor()
            model.load_model(model_file)

            preprocessing = joblib.load(preprocessing_file)

            try:
                quantile = float(q_str)
            except (TypeError, ValueError):
                continue

            per_quantile[quantile] = {
                "model": model,
                "preprocessing": preprocessing,
            }

        if per_quantile:
            loaded[tier] = per_quantile

    return loaded


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def build_model_input(preprocessing, base_row):
    
    """
    Reproduce, for a single live row, exactly what src.train_model.transform_features() does at training time 
    """
    columns = preprocessing["feature_columns"]

    row = {col: base_row.get(col) for col in columns}

    if preprocessing.get("pooled"):
        encoding = preprocessing["airline_encoding"]
        airline = base_row.get("airline")
        row["airline"] = encoding["mapping"].get(str(airline), encoding["fallback"])

    stats = preprocessing.get("imputer_statistics", {})

    for col in columns:
        value = row.get(col)

        if value is None or (isinstance(value, float) and np.isnan(value)):
            row[col] = stats.get(col, 0.0)

    return pd.DataFrame([[row[c] for c in columns]], columns=columns)


def predict_tier(tier_models, base_row):

    """Returns ({quantile: predicted_duration_anomaly_minutes}, route_median_minutes)."""

    predictions = {}
    route_median = None

    for quantile, bundle in sorted(tier_models.items()):
        X = build_model_input(bundle["preprocessing"], base_row)
        predictions[quantile] = float(bundle["model"].predict(X)[0])
        route_median = bundle["preprocessing"].get("route_median", route_median)

    return predictions, route_median


def typical_row_for_route(features_df, origin, destination):

    """Average weather + modal hour + current month for this route --
    used for the Dashboard tab's at-a-glance model results (no live
    weather needed there, unlike the Live Predictor tab)."""

    route_df = features_df[features_df["route"] == route_key(origin, destination)]

    if route_df.empty:
        return None

    weather_cols = [c for c in route_df.columns if c.startswith("dep_") or c.startswith("arr_")]

    row = {
        col: float(route_df[col].mean())
        for col in weather_cols
        if col in route_df.columns and pd.api.types.is_numeric_dtype(route_df[col])
    }
    row["hour_of_day"] = int(route_df["hour_of_day"].mode().iloc[0]) if "hour_of_day" in route_df.columns else 12
    row["month"] = dt.date.today().month
    row["airline"] = None  # airline-agnostic figure for the Dashboard's "Global model" card

    return row


# ---------------------------------------------------------------------------
# Small display helpers
# ---------------------------------------------------------------------------

def render_quantile_cards(predictions, route_median):
    """P50 / P90 / P95 as recommended total buffered duration."""
    labels = {0.5: "Typical (P50)", 0.9: "Cautious (P90)", 0.95: "Safe (P95)"}

    cols = st.columns(len(predictions) or 1)

    for col, quantile in zip(cols, sorted(predictions)):
        anomaly = predictions[quantile]
        total = (route_median or 0) + anomaly

        with col:
            st.metric(
                labels.get(quantile, f"P{int(quantile * 100)}"),
                f"{total:.0f} min",
                delta=f"{anomaly:+.0f} min vs. median",
                delta_color="inverse",
            )


def eligible_airlines_for_route(features_df, origin, destination):
    if features_df is None:
        return []

    route_df = features_df[features_df["route"] == route_key(origin, destination)]

    if "airline" not in route_df.columns:
        return []

    return sorted(a for a in route_df["airline"].dropna().unique())


@st.cache_data(show_spinner="Fetching today's Open-Meteo forecast...")
def cached_live_weather(lat, lon, cache_date):
    """
    Fetches live weather for a given lat/lon, cached to avoid repeated calls to Open-Meteo for the same day. 
    The cache key is (lat, lon, date).
    
    """
    return fetch_live_weather(lat, lon)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("\u2708\ufe0f Inbound Flight Reliability")


st.markdown(  
    "A decision-support tool for ground-ops buffer planning on inbound "
    "flights. Rather than a single delay estimate, it predicts a **range "
    "of likely flight-duration outcomes** (Typical / Cautious / Safe) "
    "using XGBoost quantile regression trained on historical flight and "
    "weather data — a **global model** for the route, plus a **dedicated "
    "model per airline** where enough history exists. The Live Predictor "
    "tab feeds in today's actual weather forecast for a live recommendation."
)  

st.caption(
    "Data sources: "
    "[OpenSky Network](https://opensky-network.org/) (flight history) \u00b7 "
    "[Open-Meteo](https://open-meteo.com/) (weather, historical + forecast)"
)

st.divider()

tab_dashboard, tab_predictor = st.tabs(["\U0001F4CA Dashboard", "\U0001F52E Live Predictor"])


# ---------------------------------------------------------------------------
# TAB 1 -- Dashboard
# ---------------------------------------------------------------------------

with tab_dashboard:
    routes = route_options()

    if not routes:
        st.info("No routes configured in config.ROUTES yet.")
    else:
        selected = st.selectbox(
            "Route",
            routes,
            format_func=lambda r: route_label(*r),
            key="dashboard_route",
        )
        origin, destination = selected

        manifest = load_tier_manifest(origin, destination)
        tier_models = load_tier_models(origin, destination)
        features_df = load_features()

        if not manifest or not tier_models:
            st.info(
                f"No trained production models found yet for {route_label(origin, destination)}. "
                "Run the monthly-retrain workflow (or `python -m src.train_model` locally) "
                "to populate models/."
            )
        elif features_df is None:
            st.info("No processed feature data found yet (data/processed/features.parquet).")
        else:
            typical_row = typical_row_for_route(features_df, origin, destination)

            # --- Global model -------------------------------------------------
            st.subheader("\U0001F310 Global Model \u2014 All Carriers")
            st.caption(
                "Pooled across every airline on this route; airline is used only "
                "as one input feature, not as a separate model."
            )

            if "all_carriers" in tier_models and typical_row is not None:
                predictions, route_median = predict_tier(tier_models["all_carriers"], typical_row)
                render_quantile_cards(predictions, route_median)
            else:
                st.caption("Not available for this route yet.")

            st.divider()

            # --- Per-airline models --------------------------------------------
            st.subheader("\u2708\ufe0f Dedicated Per-Airline Models")
            st.caption(
                f"A carrier gets its own dedicated model once it has at least "
                f"{config.MIN_ROWS_FOR_MODEL_FEASIBILITY} historical flights on this route; "
                "otherwise the global model above is used for that carrier."
            )

            carrier_tiers = sorted(t for t in tier_models if t != "all_carriers")

            if not carrier_tiers:
                st.caption("No carrier on this route currently meets the dedicated-model threshold.")
            else:
                for tier in carrier_tiers:
                    with st.container(border=True):
                        st.markdown(f"**{airline_label(tier)}**")

                        carrier_row = dict(typical_row) if typical_row else None

                        if carrier_row is not None:
                            predictions, route_median = predict_tier(tier_models[tier], carrier_row)
                            render_quantile_cards(predictions, route_median)


# ---------------------------------------------------------------------------
# TAB 2 -- Live Predictor
# ---------------------------------------------------------------------------

with tab_predictor:
    routes = route_options()

    if not routes:
        st.info("No routes configured in config.ROUTES yet.")
    else:
        selected = st.selectbox(
            "Route",
            routes,
            format_func=lambda r: route_label(*r),
            key="predictor_route",
        )
        origin, destination = selected

        tier_models = load_tier_models(origin, destination)
        features_df = load_features()

        if not tier_models:
            st.info(
                f"No trained production models found yet for {route_label(origin, destination)}."
            )
        else:
            origin_info = config.ORIGIN_AIRPORTS.get(origin, {})
            dest_info = config.DEST_AIRPORTS.get(destination, {})


            today = dt.date.today()

            try:
                dep_weather_df = (
                    cached_live_weather(origin_info["lat"], origin_info["lon"], today)
                    if origin_info else pd.DataFrame()
                )
                arr_weather_df = (
                    cached_live_weather(dest_info["lat"], dest_info["lon"], today)
                    if dest_info else pd.DataFrame()
                )
                weather_fetch_error = None
            except Exception as exc:
                dep_weather_df = pd.DataFrame()
                arr_weather_df = pd.DataFrame()
                weather_fetch_error = str(exc)

            if weather_fetch_error:
                st.error(f"Could not reach the Open-Meteo forecast right now: {weather_fetch_error}")

            st.caption(
                f"Weather source: [Open-Meteo forecast]({LIVE_WEATHER_URL}) "
                f"\u2014 fetched once today ({today.isoformat()}) and reused for every prediction below."
            )

            col_route, col_hour, col_airline = st.columns(3)

            with col_route:
                st.text_input("Route", route_label(origin, destination), disabled=True)

            with col_hour:
                hour_of_day = st.selectbox("Arrival hour (UTC)", list(range(24)), index=dt.datetime.utcnow().hour)

            with col_airline:
                available_airlines = eligible_airlines_for_route(features_df, origin, destination)
                airline_choice = st.selectbox(
                    "Airline",
                    ["All carriers (global model)"] + available_airlines,
                    format_func=lambda a: a if a == "All carriers (global model)" else airline_label(a),
                )

        
            current_month = today.month
            st.caption(f"Month is taken automatically from today's date: {today.strftime('%B')}.")

            ts_utc = pd.Timestamp.combine(today, dt.time(hour=hour_of_day)).tz_localize("UTC")

            dep_weather = nearest_hour_weather(dep_weather_df, ts_utc)
            arr_weather = nearest_hour_weather(arr_weather_df, ts_utc)

            if dep_weather or arr_weather:
                st.markdown("**Forecast values used for this prediction**")

                weather_table = pd.DataFrame(
                    [
                        {
                            "Feature": feature,
                            f"{origin} (departure)": dep_weather.get(feature),
                            f"{destination} (arrival)": arr_weather.get(feature),
                        }
                        for feature in WEATHER_FEATURES
                    ]
                )
                st.dataframe(weather_table, hide_index=True, use_container_width=True)

            if st.button("Predict", type="primary"):
                if not dep_weather or not arr_weather:
                    st.error("Could not retrieve a forecast for this route right now. Try again shortly.")
                else:

                    base_row = {f"dep_{k}": v for k, v in dep_weather.items() if k in WEATHER_FEATURES}
                    base_row.update({f"arr_{k}": v for k, v in arr_weather.items() if k in WEATHER_FEATURES})
                    base_row["hour_of_day"] = hour_of_day
                    base_row["month"] = current_month

                    use_dedicated = (
                        airline_choice != "All carriers (global model)"
                        and airline_choice in tier_models
                    )

                    if use_dedicated:
                        tier_used = airline_choice
                        predictions, route_median = predict_tier(tier_models[airline_choice], base_row)
                    else:
                        tier_used = "all_carriers"
                        base_row["airline"] = (
                            airline_choice if airline_choice != "All carriers (global model)" else None
                        )
                        predictions, route_median = predict_tier(tier_models.get("all_carriers", {}), base_row)

                    if not predictions:
                        st.warning("No trained model available to answer this request.")
                    else:
                        tier_label = (
                            f"Dedicated {airline_label(tier_used)} model"
                            if use_dedicated
                            else "Global (all-carriers) model"
                        )
                        st.success(f"Prediction from: **{tier_label}**")
                        render_quantile_cards(predictions, route_median)
                        st.caption(
                            f"Reference: EU261 delay threshold is "
                            f"{config.EU261_DELAY_THRESHOLD_MINUTES} minutes, shown for context only."
                        )