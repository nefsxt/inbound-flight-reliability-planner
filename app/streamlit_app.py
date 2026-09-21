"""
Inbound Flight Reliability -- decision-support dashboard.

Two tabs:
  1. Dashboard        -- per-route model results (global, then per-airline).
  2. Live Predictor   -- today's Open-Meteo forecast fed into the trained
                         models for a live buffer recommendation.

Data:
  - Local development:
        DATA_SOURCE=local

  - Streamlit Cloud:
        DATA_SOURCE="hf"
        HF_TOKEN="hf_..."

    The private Hugging Face dataset/model repository is downloaded into
    data_cache/ and all data/model reads are performed from that directory.
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


# ---------------------------------------------------------------------------
# Project root / imports
# ---------------------------------------------------------------------------

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
from src.fetch_weather import (
    fetch_live_weather,
    nearest_hour_weather,
    LIVE_WEATHER_URL,
)


# ---------------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Inbound Flight-Time Anomaly Prediction",
    page_icon="🛬",
    layout="wide",
)
###  



# ---------------------------------------------------------------------------
# Configuration / secrets
# ---------------------------------------------------------------------------

def get_setting(name, default=None):
    """
    Read configuration from Streamlit Secrets first, then environment.

    Streamlit Cloud secrets are available through st.secrets and should not
    be assumed to be present in os.environ.
    """

    try:
        value = st.secrets.get(name)

        if value is not None:
            return value

    except Exception:
        # st.secrets can raise when no secrets file exists in local
        # development.
        pass

    return os.getenv(name, default)


DATA_SOURCE = str(
    get_setting("DATA_SOURCE", "local")
).strip().lower()


HF_TOKEN = get_setting("HF_TOKEN")


# If running from Streamlit Cloud and HF_TOKEN exists in Secrets, expose it
# to code that expects the conventional HF_TOKEN environment variable.
if HF_TOKEN:
    os.environ["HF_TOKEN"] = str(HF_TOKEN)


# ---------------------------------------------------------------------------
# Data synchronization
# ---------------------------------------------------------------------------

@st.cache_resource(ttl=300)
def sync_from_hf():
    """
    Download the inference artifacts from the Hugging Face
    data repository.

    The result is cached for 5 minutes.
    """

    from src.hf_storage import download_inference_artifacts

    token = get_setting("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is not configured. Add HF_TOKEN to the Streamlit "
            "Cloud app's Settings → Secrets."
        )

    # Make the token available to hf_storage.py.
    os.environ["HF_TOKEN"] = str(token)

    data_root = download_inference_artifacts(
        local_dir="data_cache"
    )

    if not data_root:
        raise RuntimeError(
            "download_inference_artifacts() returned no local data directory."
        )

    data_root = os.path.abspath(data_root)

    if not os.path.isdir(data_root):
        raise RuntimeError(
            f"HF sync returned a directory that does not exist: {data_root}"
        )

    return data_root


# ---------------------------------------------------------------------------
# Determine DATA_ROOT
# ---------------------------------------------------------------------------

DATA_SYNC_ERROR = None


if DATA_SOURCE == "hf":

    try:
        DATA_ROOT = sync_from_hf()

    except Exception as exc:

        DATA_ROOT = None
        DATA_SYNC_ERROR = str(exc)

else:

    DATA_ROOT = PROJECT_ROOT


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def route_key(origin, destination):
    """
    Route key used by the feature dataset and model directories.

    Example:
        EDDF -> LGTS

    becomes:
        EDDF->LGTS
    """

    return f"{origin}->{destination}"


def route_label(origin, destination):
    return f"{origin} → {destination}"


def route_model_dir(origin, destination):
    """
    Return the model directory for a route.
    """

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


def features_path():
    return os.path.join(
        DATA_ROOT,
        config.PROCESSED_DIR,
        "features.parquet",
    )


def _absolute_model_path(path):
    """
    Resolve a model path stored in tier_manifest.json.

    Absolute paths are preserved.

    Relative paths are interpreted relative to DATA_ROOT.
    """

    if not path:
        return None

    if os.path.isabs(path):
        return path

    return os.path.join(DATA_ROOT, path)


# ---------------------------------------------------------------------------
# Validate synchronized data
# ---------------------------------------------------------------------------

def validate_data_root():
    """
    Verify that the selected data source contains the files/directories
    required by the application.

    This is deliberately strict when DATA_SOURCE=hf so that the app does
    not silently run against the GitHub checkout when HF data is missing.
    """

    if not DATA_ROOT:
        return [
            "DATA_ROOT is not set."
        ]

    required = {
        "models directory": os.path.join(
            DATA_ROOT,
            "models",
        ),

        "features.parquet": features_path(),
    }

    missing = []

    for description, path in required.items():

        if not os.path.exists(path):
            missing.append(
                f"{description}: {path}"
            )

    return missing


if DATA_SOURCE == "hf":

    if DATA_SYNC_ERROR:

        st.error(
            "Could not synchronize the private Hugging Face data repository."
        )

        st.code(
            DATA_SYNC_ERROR,
            language="text",
        )

        st.info(
            "Check that DATA_SOURCE=\"hf\" and HF_TOKEN are configured "
            "in this Streamlit app's Settings → Secrets."
        )

        st.stop()

    missing_data = validate_data_root()

    if missing_data:

        st.error(
            "The Hugging Face synchronization completed, but the expected "
            "data files were not found."
        )

        for item in missing_data:
            st.code(item)

        st.info(
            "The app is intentionally stopping instead of falling back "
            "to the GitHub checkout."
        )

        st.stop()


# ---------------------------------------------------------------------------
# Optional diagnostics
# ---------------------------------------------------------------------------

with st.sidebar:

    st.caption(
        f"Data source: `{DATA_SOURCE}`"
    )

    st.caption(
        f"Data root: `{DATA_ROOT}`"
    )

st.sidebar.write("Data source:", DATA_SOURCE)
st.sidebar.write("Data root:", DATA_ROOT)

models_root = os.path.join(DATA_ROOT, "models")

st.sidebar.write(
    "Models directory exists:",
    os.path.isdir(models_root),
)

if os.path.isdir(models_root):
    st.sidebar.write(
        "Model directories:",
        os.listdir(models_root),
    )

st.sidebar.write(
    "Expected model directory:",
    route_model_dir("EDDF", "LGTS"),
)

st.sidebar.write(
    "Expected manifest:",
    tier_manifest_path("EDDF", "LGTS"),
)

st.sidebar.write(
    "Manifest exists:",
    os.path.isfile(
        tier_manifest_path("EDDF", "LGTS")
    ),
)


with open(
    tier_manifest_path("EDDF", "LGTS"),
    "r",
    encoding="utf-8",
) as f:
    manifest = json.load(f)

#st.sidebar.write("Tier manifest:", manifest)

all_carriers = manifest.get("all_carriers", {})

st.sidebar.write(
    "all_carriers keys:",
    list(all_carriers.keys()),
)

st.sidebar.write(
    "quantile keys:",
    list(all_carriers.get("quantiles", {}).keys()),
)

#for quantile, details in all_carriers.get("quantiles", {}).items():
#    st.sidebar.write(
#        f"Quantile {quantile} keys:",
#        list(details.keys()),
#    )

for quantile, details in all_carriers.get("quantiles", {}).items():
    st.sidebar.write(
        f"Quantile {quantile} production:",
        details.get("production"),
    )

# ---------------------------------------------------------------------------
# Route configuration
# ---------------------------------------------------------------------------

def route_options():
    """
    [(origin, destination), ...]

    Straight from config.ROUTES. The first entry is the default.
    """

    return list(config.ROUTES)


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

    return load_json(
        tier_manifest_path(
            origin,
            destination,
        )
    )


@st.cache_data
def load_features():

    path = features_path()

    if not os.path.exists(path):
        return None

    return pd.read_parquet(path)


@st.cache_resource
def load_tier_models(origin, destination):
    """
    Load and cache XGBoost regression models and preprocessing pipelines
    for a route.

    Returns:

        {
            tier_name: {
                quantile: {
                    "model": xgb.XGBRegressor,
                    "preprocessing": dict
                }
            }
        }

    Tier names are:

        all_carriers

    and/or airline codes such as:

        AEE
        A3
        etc.
    """

    manifest = load_tier_manifest(
        origin,
        destination,
    )

    if not manifest:
        return {}

    tiers = {
        "all_carriers": manifest.get(
            "all_carriers"
        )
    }

    tiers.update(
        manifest.get(
            "by_carrier",
            {}
        ).get(
            "carriers",
            {}
        )
    )

    loaded = {}

    for tier, entry in tiers.items():

        if not isinstance(entry, dict):
            continue

        if not entry.get("trained"):
            continue

        per_quantile = {}

        for q_str, q_entry in entry.get(
            "quantiles",
            {}
        ).items():

            if not isinstance(q_entry, dict):
                continue

            production = q_entry.get(
                "production",
                {}
            )

            if not isinstance(production, dict):
                continue

            model_file = _absolute_model_path(
                production.get(
                    "model_path"
                )
            )

            preprocessing_file = _absolute_model_path(
                production.get(
                    "preprocessing_path"
                )
            )

            if (
                not model_file
                or not os.path.exists(model_file)
            ):
                continue

            if (
                not preprocessing_file
                or not os.path.exists(preprocessing_file)
            ):
                continue

            model = xgb.XGBRegressor()

            model.load_model(
                model_file
            )

            preprocessing = joblib.load(
                preprocessing_file
            )

            try:
                quantile = float(q_str)

            except (
                TypeError,
                ValueError,
            ):
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

def build_model_input(
    preprocessing,
    base_row,
):
    """
    Reproduce the feature transformation performed during training for a
    single live prediction row.
    """

    columns = preprocessing[
        "feature_columns"
    ]

    row = {
        col: base_row.get(col)
        for col in columns
    }

    if preprocessing.get("pooled"):

        encoding = preprocessing[
            "airline_encoding"
        ]

        airline = base_row.get(
            "airline"
        )

        row["airline"] = encoding[
            "mapping"
        ].get(
            str(airline),
            encoding["fallback"],
        )

    stats = preprocessing.get(
        "imputer_statistics",
        {}
    )

    for col in columns:

        value = row.get(col)

        if value is None:

            row[col] = stats.get(
                col,
                0.0,
            )

        elif isinstance(
            value,
            (float, np.floating),
        ) and np.isnan(value):

            row[col] = stats.get(
                col,
                0.0,
            )

    return pd.DataFrame(
        [
            [
                row[c]
                for c in columns
            ]
        ],
        columns=columns,
    )


def predict_tier(
    tier_models,
    base_row,
):
    """
    Returns:

        predictions,
        route_median_minutes
    """

    predictions = {}

    route_median = None

    for quantile, bundle in sorted(
        tier_models.items()
    ):

        X = build_model_input(
            bundle["preprocessing"],
            base_row,
        )

        predictions[quantile] = float(
            bundle["model"].predict(X)[0]
        )

        route_median = bundle[
            "preprocessing"
        ].get(
            "route_median",
            route_median,
        )

    return (
        predictions,
        route_median,
    )


def typical_row_for_route(
    features_df,
    origin,
    destination,
):
    """
    Average weather + modal hour + current month for this route.

    Used by the Dashboard tab's at-a-glance model results.
    """

    route_df = features_df[
        features_df["route"]
        == route_key(
            origin,
            destination,
        )
    ]

    if route_df.empty:
        return None

    weather_cols = [
        c
        for c in route_df.columns
        if c.startswith("dep_")
        or c.startswith("arr_")
    ]

    row = {
        col: float(
            route_df[col].mean()
        )
        for col in weather_cols
        if (
            col in route_df.columns
            and pd.api.types.is_numeric_dtype(
                route_df[col]
            )
        )
    }

    if "hour_of_day" in route_df.columns:

        row["hour_of_day"] = int(
            route_df[
                "hour_of_day"
            ].mode().iloc[0]
        )

    else:

        row["hour_of_day"] = 12

    row["month"] = dt.date.today().month

    # Airline-agnostic dashboard row.
    row["airline"] = None

    return row


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def render_quantile_cards(
    predictions,
    route_median,
):
    """
    P50 / P90 / P95 as recommended total buffered duration.
    """

    labels = {
        0.5: "Typical (P50)",
        0.9: "Cautious (P90)",
        0.95: "Safe (P95)",
    }

    cols = st.columns(
        len(predictions) or 1
    )

    for col, quantile in zip(
        cols,
        sorted(predictions),
    ):

        anomaly = predictions[
            quantile
        ]

        total = (
            route_median or 0
        ) + anomaly

        with col:

            st.metric(
                labels.get(
                    quantile,
                    f"P{int(quantile * 100)}",
                ),
                f"{total:.0f} min",
                delta=(
                    f"{anomaly:+.0f} min "
                    "vs. median"
                ),
                delta_color="inverse",
            )


def eligible_airlines_for_route(
    features_df,
    origin,
    destination,
):

    if features_df is None:
        return []

    route_df = features_df[
        features_df["route"]
        == route_key(
            origin,
            destination,
        )
    ]

    if "airline" not in route_df.columns:
        return []

    return sorted(
        a
        for a in route_df[
            "airline"
        ].dropna().unique()
    )


# ---------------------------------------------------------------------------
# Live weather
# ---------------------------------------------------------------------------

@st.cache_data(
    ttl=86400,
    show_spinner="Fetching today's Open-Meteo forecast...",
)
def cached_live_weather(
    lat,
    lon,
    cache_date,
):
    """
    Fetch live weather for a given location/date.

    cache_date deliberately forms part of the cache key so the forecast
    refreshes on a new calendar day.
    """

    return fetch_live_weather(
        lat,
        lon,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title(
    "🛬 Inbound Flight-Time Anomaly Prediction"
)


st.markdown(
    """
    A proof-of-concept decision-support tool for **inbound ground-operations buffer planning**. 
    Instead of predicting a single delay value, the app combines historical flight data with 
    **real-time weather forecasts** to produce three probabilistic planning scenarios: 
    **Typical (P50), Cautious (P90), and Safe (P95)**.

    The models predict **flight-duration anomalies** relative to the historical median for each route:

    `Flight Duration Anomaly = Actual Gate-to-Gate Duration − Route Historical Median`

    This means the predictions describe deviations from typical observed flight duration—not whether a flight will meet its published commercial schedule.

    The three quantiles provide progressively more conservative references for evaluating current conditions and deciding how much operational buffer may be appropriate.

    See [README.md](#) and [ARCHITECTURE.md](#) for more details.  
    [GitHub Repository](https://github.com/nefsxt/inbound-flight-reliability-planner)
    """
)



st.caption(
    "Data sources: "
    "[OpenSky Network](https://opensky-network.org/) "
    "· "
    "[Open-Meteo](https://open-meteo.com/) "
    "(weather, historical + forecast)"
)


st.divider()


tab_dashboard, tab_predictor = st.tabs(
    [
        "📊 Dashboard",
        "🔮 Live Predictor",
    ]
)


# ===========================================================================
# TAB 1 -- Dashboard
# ===========================================================================

with tab_dashboard:

    routes = route_options()

    if not routes:

        st.info(
            "No routes configured in config.ROUTES yet."
        )

    else:

        selected = st.selectbox(
            "Route",
            routes,
            format_func=lambda r: route_label(
                *r
            ),
            key="dashboard_route",
        )

        origin, destination = selected

        manifest = load_tier_manifest(
            origin,
            destination,
        )

        tier_models = load_tier_models(
            origin,
            destination,
        )

        features_df = load_features()

        if not manifest or not tier_models:

            st.info(
                f"No trained production models found yet "
                f"for {route_label(origin, destination)}. "
                "Run the monthly-retrain workflow (or "
                "`python -m src.train_model` locally) to populate models/."
            )

        elif features_df is None:

            st.info(
                "No processed feature data found yet "
                "(data/processed/features.parquet)."
            )

        else:

            typical_row = typical_row_for_route(
                features_df,
                origin,
                destination,
            )

            # ----------------------------------------------------------------
            # Global model
            # ----------------------------------------------------------------

            st.subheader(
                "🌐 Global Model — All Carriers"
            )

            st.caption(
                "Pooled across every airline on this route; "
                "airline is used only as one input feature, "
                "not as a separate model."
            )

            if (
                "all_carriers" in tier_models
                and typical_row is not None
            ):

                predictions, route_median = predict_tier(
                    tier_models[
                        "all_carriers"
                    ],
                    typical_row,
                )

                render_quantile_cards(
                    predictions,
                    route_median,
                )

            else:

                st.caption(
                    "Not available for this route yet."
                )


            st.divider()


            # ----------------------------------------------------------------
            # Per-airline models
            # ----------------------------------------------------------------

            st.subheader(
                "✈️ Dedicated Per-Airline Models"
            )

            st.caption(
                f"A carrier gets its own dedicated model once it has at least "
                f"{config.MIN_ROWS_FOR_MODEL_FEASIBILITY} historical flights "
                "on this route; otherwise the global model above is used "
                "for that carrier."
            )

            carrier_tiers = sorted(
                t
                for t in tier_models
                if t != "all_carriers"
            )

            if not carrier_tiers:

                st.caption(
                    "No carrier on this route currently meets "
                    "the dedicated-model threshold."
                )

            else:

                for tier in carrier_tiers:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"**{airline_label(tier)}**"
                        )

                        carrier_row = (
                            dict(typical_row)
                            if typical_row
                            else None
                        )

                        if carrier_row is not None:

                            # Explicitly provide the airline code for
                            # dedicated models in case their preprocessing
                            # expects it.
                            carrier_row[
                                "airline"
                            ] = tier

                            predictions, route_median = predict_tier(
                                tier_models[tier],
                                carrier_row,
                            )

                            render_quantile_cards(
                                predictions,
                                route_median,
                            )


# ===========================================================================
# TAB 2 -- Live Predictor
# ===========================================================================

with tab_predictor:

    routes = route_options()

    if not routes:

        st.info(
            "No routes configured in config.ROUTES yet."
        )

    else:

        selected = st.selectbox(
            "Route",
            routes,
            format_func=lambda r: route_label(
                *r
            ),
            key="predictor_route",
        )

        origin, destination = selected

        tier_models = load_tier_models(
            origin,
            destination,
        )

        features_df = load_features()

        if not tier_models:

            st.info(
                f"No trained production models found yet "
                f"for {route_label(origin, destination)}."
            )

        else:

            origin_info = config.ORIGIN_AIRPORTS.get(
                origin,
                {}
            )

            dest_info = config.DEST_AIRPORTS.get(
                destination,
                {}
            )

            today = dt.date.today()

            try:

                dep_weather_df = (
                    cached_live_weather(
                        origin_info["lat"],
                        origin_info["lon"],
                        today,
                    )
                    if origin_info
                    else pd.DataFrame()
                )

                arr_weather_df = (
                    cached_live_weather(
                        dest_info["lat"],
                        dest_info["lon"],
                        today,
                    )
                    if dest_info
                    else pd.DataFrame()
                )

                weather_fetch_error = None

            except Exception as exc:

                dep_weather_df = pd.DataFrame()

                arr_weather_df = pd.DataFrame()

                weather_fetch_error = str(
                    exc
                )


            if weather_fetch_error:

                st.error(
                    "Could not reach the Open-Meteo "
                    f"forecast right now: {weather_fetch_error}"
                )


            st.caption(
                f"Weather source: "
                f"[Open-Meteo forecast]({LIVE_WEATHER_URL}) "
                f"— fetched once today "
                f"({today.isoformat()}) and reused for "
                "every prediction below."
            )


            col_route, col_hour, col_airline = st.columns(
                3
            )


            with col_route:

                st.text_input(
                    "Route",
                    route_label(
                        origin,
                        destination,
                    ),
                    disabled=True,
                )


            with col_hour:

                hour_of_day = st.selectbox(
                    "Arrival hour (UTC)",
                    list(range(24)),
                    index=dt.datetime.utcnow().hour,
                )


            with col_airline:

                available_airlines = (
                    eligible_airlines_for_route(
                        features_df,
                        origin,
                        destination,
                    )
                )

                airline_choice = st.selectbox(
                    "Airline",
                    [
                        "All carriers (global model)"
                    ] + available_airlines,
                    format_func=lambda a:
                        a
                        if a
                        == "All carriers (global model)"
                        else airline_label(a),
                )


            current_month = today.month

            st.caption(
                "Month is taken automatically from today's date: "
                f"{today.strftime('%B')}."
            )


            ts_utc = pd.Timestamp.combine(
                today,
                dt.time(
                    hour=hour_of_day
                ),
            ).tz_localize("UTC")


            dep_weather = nearest_hour_weather(
                dep_weather_df,
                ts_utc,
            )

            arr_weather = nearest_hour_weather(
                arr_weather_df,
                ts_utc,
            )


            if dep_weather or arr_weather:

                st.markdown(
                    "**Forecast values used for this prediction**"
                )

                weather_table = pd.DataFrame(
                    [
                        {
                            "Feature": feature,
                            f"{origin} (departure)": (
                                dep_weather.get(
                                    feature
                                )
                            ),
                            f"{destination} (arrival)": (
                                arr_weather.get(
                                    feature
                                )
                            ),
                        }
                        for feature in WEATHER_FEATURES
                    ]
                )

                st.dataframe(
                    weather_table,
                    hide_index=True,
                    use_container_width=True,
                )


            if st.button(
                "Predict",
                type="primary",
            ):

                if not dep_weather or not arr_weather:

                    st.error(
                        "Could not retrieve a forecast for "
                        "this route right now. Try again shortly."
                    )

                else:

                    base_row = {
                        f"dep_{k}": v
                        for k, v in dep_weather.items()
                        if k in WEATHER_FEATURES
                    }

                    base_row.update(
                        {
                            f"arr_{k}": v
                            for k, v in arr_weather.items()
                            if k in WEATHER_FEATURES
                        }
                    )

                    base_row[
                        "hour_of_day"
                    ] = hour_of_day

                    base_row[
                        "month"
                    ] = current_month


                    use_dedicated = (
                        airline_choice
                        != "All carriers (global model)"
                        and airline_choice
                        in tier_models
                    )


                    if use_dedicated:

                        tier_used = (
                            airline_choice
                        )

                        base_row[
                            "airline"
                        ] = airline_choice

                        predictions, route_median = (
                            predict_tier(
                                tier_models[
                                    airline_choice
                                ],
                                base_row,
                            )
                        )

                    else:

                        tier_used = (
                            "all_carriers"
                        )

                        base_row[
                            "airline"
                        ] = (
                            airline_choice
                            if airline_choice
                            != "All carriers (global model)"
                            else None
                        )

                        predictions, route_median = (
                            predict_tier(
                                tier_models.get(
                                    "all_carriers",
                                    {},
                                ),
                                base_row,
                            )
                        )


                    if not predictions:

                        st.warning(
                            "No trained model available "
                            "to answer this request."
                        )

                    else:

                        tier_label = (
                            f"Dedicated "
                            f"{airline_label(tier_used)} model"
                            if use_dedicated
                            else "Global (all-carriers) model"
                        )

                        st.success(
                            f"Prediction from: **{tier_label}**"
                        )

                        render_quantile_cards(
                            predictions,
                            route_median,
                        )

                        st.caption(
                            "Reference: EU261 delay threshold is "
                            f"{config.EU261_DELAY_THRESHOLD_MINUTES} "
                            "minutes, shown for context only."
                        )

