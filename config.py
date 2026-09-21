"""
Central configuration for Inbound Flight Reliability.

Data sources are OpenSky (flight history) and Open-Meteo (weather) ONLY.
Every path in the pipeline is derived from the helpers at the bottom of
this file so fetch/feature/train scripts never hand-roll a path -- add a
route here and every stage (raw cache, processed flights, weather,
combined feature matrix) picks it up consistently.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- OpenSky credentials -----------------------------------------------
OPENSKY_CLIENT_ID = os.getenv("OPENSKY_CLIENT_ID", "")
OPENSKY_CLIENT_SECRET = os.getenv("OPENSKY_CLIENT_SECRET", "")

# --- Destination airports (this project) --------------------------------
DEST_AIRPORTS = {
    "LGAV": {"name": "Athens Eleftherios Venizelos", "lat": 37.9364, "lon": 23.9445, "tz": "Europe/Athens"},
    "LGTS": {"name": "Thessaloniki Makedonia", "lat": 40.5197, "lon": 22.9709, "tz": "Europe/Athens"},
}

# --- Origin airports (reference set; only ones used by ROUTES below are
#     actually fetched, but it's harmless/useful to keep the rest defined
#     here for when you add a route) --------------------------------------
ORIGIN_AIRPORTS = {
    "EGLL": {"name": "London Heathrow", "lat": 51.470748, "lon": -0.459909, "tz": "Europe/London"},
    "EDDF": {"name": "Frankfurt Main", "lat": 50.026706, "lon": 8.558350, "tz": "Europe/Berlin"},
    "LCLK": {"name": "Larnaca", "lat": 34.875099, "lon": 33.624901, "tz": "Asia/Nicosia"},
    "LGAV": {"name": "Athens International", "lat": 37.936401, "lon": 23.944500, "tz": "Europe/Athens"},
    "LIRF": {"name": "Rome Fiumicino", "lat": 41.804532, "lon": 12.251998, "tz": "Europe/Rome"},
    "VHHH": {"name": "Hong Kong International", "lat": 22.311840, "lon": 113.914862, "tz": "Asia/Hong_Kong"},
    "LGTS": {"name": "Thessaloniki", "lat": 40.519280, "lon": 22.970009, "tz": "Europe/Athens"},
}

# --- Active routes: (origin_icao, destination_icao) ----------------------
# Every configured pipeline stage (fetch_opensky, fetch_weather, features,
# train_model) iterates this list. Only routes with real collected history
# should be listed here -- add a route once you've backfilled it, don't
# just add it and expect data to appear.
ROUTES = [
    ("EDDF", "LGTS"),  # Frankfurt -> Thessaloniki (3 years of real history)
]

# --- Date range for the historical backfill --------------------------------
HIST_START_DATE = "2023-07-31"
HIST_END_DATE = "2026-07-31"

# --- OpenSky credit management ---------------------------------------------
# OpenSky's documented free allotment is ~4000 credits/day for registered
# users, metered by the time span queried on /flights and /tracks endpoints.
# This is a conservative self-imposed chunk size, not an official credit
# count -- tune it down if you see 429s.
OPENSKY_QUERY_CHUNK_DAYS = 1
OPENSKY_REQUEST_PAUSE_SECONDS = 2

# Network configuration
OPENSKY_AUTH_TIMEOUT_SECONDS = 30
OPENSKY_API_TIMEOUT_SECONDS = 60

# --- Daily incremental update ------------------------------------------
# See src/update_data.py. If a daily update fails (network error, API
# outage, etc.), we don't hammer the source again the next day -- we wait
# this many days before retrying, at which point we again pull everything
# from the last successfully stored date up to "today".
UPDATE_RETRY_BACKOFF_DAYS = 2

# --- Paths -----------------------------------------------------------------
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

# --- Modeling --------------------------------------------------------------
QUANTILES = [0.5, 0.9, 0.95]

# Minimum historical flights required before we consider a model trustworthy
# enough to train and surface in the app -- applies uniformly to the
# all-carriers route model AND to any single carrier's dedicated model.
# Below this, the app falls back to the plain historical baseline
# (percentiles of actual duration anomaly) instead of a model prediction.
MIN_ROWS_FOR_MODEL_FEASIBILITY = 500

# Number of expanding-window walk-forward backtest folds used to report
# per-fold calibration/error metrics in the Model Card (see
# src/train_model.run_walkforward_backtest). 3 folds keeps each fold's
# training window a meaningfully different size while still leaving enough
# rows per test fold to trust the reported coverage percentage.
BACKTEST_N_FOLDS = 3


#  Tuning & Loss Penalties 
# Weights used inside the custom Optuna objective function to penalize 
# high calibration errors and high fold-to-fold variance.
COVERAGE_ERROR_WEIGHT = 10
FOLD_BALANCE_WEIGHT = 10

# Default number of optimization trials run by Optuna during hyperparameter search.
OPTUNA_TRIALS = 40


# HF Spaces Dataset repo configuration -- see src/hf_storage.py for upload/download logic.
HF_REPO_ID = "nefelisxt/inbound-flight-data"
HF_REPO_TYPE = "dataset"




# --- Airline-tier modeling ---------------------------------------------------
# Carrier identity is treated as a first-class signal, not an optional
# add-on: every trained model includes airline_code as a feature at
# minimum (the "all carriers" tier), and a carrier with enough
# route-specific history (>= MIN_ROWS_FOR_MODEL_FEASIBILITY flights) gets
# its own dedicated model on top of that. See NOTES.md "Airline modeling
# tiers" for the fallback logic.


# --- EU261-style reference threshold (for dashboard framing only) ---------
EU261_DELAY_THRESHOLD_MINUTES = 180  # 3 hours


# ---------------------------------------------------------------------------
# Path helpers -- every script imports THESE instead of building its own
# f-strings, so raw/processed layout can only drift from one place.
#
#   data/raw/<ORIGIN>_<DEST>/
#       arrivals/                 <- cached OpenSky arrival chunks
#       weather.parquet           <- consolidated Open-Meteo route weather
#   data/raw/weather_hist_chunks/<AIRPORT>/
#                                 <- per-airport Open-Meteo chunk cache,
#                                    shared across any route touching that
#                                    airport (no point fetching Frankfurt's
#                                    weather twice for two different routes)
#   data/processed/<ORIGIN>_<DEST>/
#       flights.parquet           <- sanity-filtered OpenSky arrivals
#   data/processed/features.parquet
#                                 <- combined, pooled feature matrix across
#                                    every route in ROUTES (model training
#                                    input)
# ---------------------------------------------------------------------------

def route_key(origin_icao: str, destination_icao: str) -> str:
    return f"{origin_icao}_{destination_icao}"


def route_raw_dir(origin_icao: str, destination_icao: str) -> str:
    return os.path.join(RAW_DIR, route_key(origin_icao, destination_icao))


def route_arrivals_dir(origin_icao: str, destination_icao: str) -> str:
    return os.path.join(route_raw_dir(origin_icao, destination_icao), "arrivals")


def route_weather_path(origin_icao: str, destination_icao: str) -> str:
    return os.path.join(route_processed_dir(origin_icao, destination_icao), "weather.parquet")

def route_processed_dir(origin_icao: str, destination_icao: str) -> str:
    return os.path.join(PROCESSED_DIR, route_key(origin_icao, destination_icao))


def route_flights_path(origin_icao: str, destination_icao: str) -> str:
    return os.path.join(route_processed_dir(origin_icao, destination_icao), "flights.parquet")


def features_path() -> str:
    return os.path.join(PROCESSED_DIR, "features.parquet")
