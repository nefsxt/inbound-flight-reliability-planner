"""
Weather data via Open-Meteo -- route-level historical weather dataset
creation.

Two different endpoints on purpose:
  - Historical Forecast API (historical-forecast-api.open-meteo.com) for
    TRAINING data. This mirrors the live Forecast API's model outputs
    (available from ~2021 onward), so it includes fields like visibility
    that the plain ERA5 reanalysis archive does not reliably provide. Using
    the same model family for training and inference avoids a train/serve
    schema mismatch.
  - Forecast API (api.open-meteo.com) for LIVE predictions in the dashboard.
  - IMPORTANT NOTE: using historical forecast ensures no data leakage w.r.t historical weather: the model sees the forcasts for the day, not the actual weather observed
                            -> this is meant to match the available weather information available to predict the target for a future scenario

Both are free, keyless.

Route-level datasets live at:

    data/raw/<ORIGIN>_<DEST>/weather.parquet

Individual historical API chunks are cached separately, per airport (shared
across any route touching that airport):

    data/raw/weather_hist_chunks/<AIRPORT>/
        <AIRPORT>_2021-01-01_2021-01-14.parquet
        <AIRPORT>_2021-01-15_2021-01-28.parquet
        ...

This means a failed/interrupted multi-year download can resume without
re-downloading chunks that were already successfully retrieved.

Usage:
    python -m src.fetch_weather                      # backfill/refresh every route in config.ROUTES
    python -m src.fetch_weather --route EDDF LGTS
    python -m src.fetch_weather --inspect --route EDDF LGTS
"""

import os
import sys
import time
import argparse
import logging
import requests
import pandas as pd
import config

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_or_none,
    save_parquet,
    inspect_cached_parquet,
    record_stage_stats,
)


HIST_WEATHER_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LIVE_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Historical API chunk size. Two weeks keeps individual requests reasonably
# small while still avoiding an excessive number of HTTP requests.
HIST_CHUNK_DAYS = 14

# Retry configuration for transient API failures.
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2.0
REQUEST_TIMEOUT = 60

WEATHER_LOG_PATH = os.path.join(config.PROCESSED_DIR, "fetch_weather.log")


def _get_logger() -> logging.Logger:

    """Return the persistent weather-fetch logger."""

    logger = logging.getLogger("fetch_weather")

    if logger.handlers:
        return logger

    os.makedirs(os.path.dirname(WEATHER_LOG_PATH), exist_ok=True)

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(WEATHER_LOG_PATH)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(handler)

    return logger


logger = _get_logger()


HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "weather_code",
    "pressure_msl",
    "cloud_cover_low",
    "cloud_cover_high",
    "visibility",
    "wind_speed_10m",
    "wind_speed_180m",
    "wind_direction_10m",
    "wind_direction_180m",
    "cape",
    "geopotential_height_850hPa",
]


def historical_chunk_path(airport_icao: str, chunk_start: pd.Timestamp, chunk_end: pd.Timestamp) -> str:

    """Cache path for one historical API chunk (shared across routes)."""

    airport_dir = os.path.join(config.RAW_DIR, "weather_hist_chunks", airport_icao)
    os.makedirs(airport_dir, exist_ok=True)

    start_str = chunk_start.strftime("%Y-%m-%d")
    end_str = chunk_end.strftime("%Y-%m-%d")

    return os.path.join(
        airport_dir,
        f"{airport_icao}_{start_str}_{end_str}.parquet",
    )


def _request_historical_chunk(params: dict) -> dict:

    """
    Execute one Historical Forecast API request with retry/backoff.

    Retries transient HTTP errors (429, 500, 502, 503, 504). Other HTTP
    errors are raised immediately.
    """
    delay = INITIAL_RETRY_DELAY

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                HIST_WEATHER_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if resp.status_code == 429:
                if attempt == MAX_RETRIES:
                    logger.error(
                        "HTTP 429 after %s attempts: %s",
                        MAX_RETRIES,
                        resp.text[:500],
                    )
                    resp.raise_for_status()

                retry_after = resp.headers.get("Retry-After")
                try:
                    sleep_time = (
                        float(retry_after)
                        if retry_after is not None
                        else delay
                    )
                except ValueError:
                    sleep_time = delay

                msg = (
                    f"Rate limited by Open-Meteo. Retrying in {sleep_time:.1f}s "
                    f"(attempt {attempt}/{MAX_RETRIES})..."
                )
                print(f"  {msg}")
                logger.warning(msg)

                time.sleep(sleep_time)
                delay *= 2
                continue

            if resp.status_code in {500, 502, 503, 504}:
                if attempt == MAX_RETRIES:
                    logger.error(
                        "HTTP %s after %s attempts: %s",
                        resp.status_code,
                        MAX_RETRIES,
                        resp.text[:500],
                    )
                    resp.raise_for_status()

                msg = (
                    f"Open-Meteo returned HTTP {resp.status_code}. "
                    f"Retrying in {delay:.1f}s "
                    f"(attempt {attempt}/{MAX_RETRIES})..."
                )
                print(f"  {msg}")
                logger.warning(msg)

                time.sleep(delay)
                delay *= 2
                continue

            if not resp.ok:
                logger.error(
                    "Open-Meteo HTTP %s: %s",
                    resp.status_code,
                    resp.text[:500],
                )
                resp.raise_for_status()

            return resp.json()

        except requests.exceptions.Timeout:
            if attempt == MAX_RETRIES:
                logger.exception(
                    "Open-Meteo request timed out after %s attempts.",
                    MAX_RETRIES,
                )
                raise

            msg = (
                f"Request timed out. Retrying in {delay:.1f}s "
                f"(attempt {attempt}/{MAX_RETRIES})..."
            )
            print(f"  {msg}")
            logger.warning(msg)

            time.sleep(delay)
            delay *= 2

        except requests.exceptions.ConnectionError:
            if attempt == MAX_RETRIES:
                logger.exception(
                    "Open-Meteo connection failed after %s attempts.",
                    MAX_RETRIES,
                )
                raise

            msg = (
                f"Connection error. Retrying in {delay:.1f}s "
                f"(attempt {attempt}/{MAX_RETRIES})..."
            )
            print(f"  {msg}")
            logger.warning(msg)

            time.sleep(delay)
            delay *= 2

    raise RuntimeError(
        f"Historical Open-Meteo request failed after {MAX_RETRIES} attempts."
    )


def fetch_historical_weather(airport_icao: str, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:

    """
    Hourly historical weather for one airport across the given date range.

    The date range is divided into small chunks; every successful chunk is
    cached independently before the next request is made, so an interrupted
    multi-year download is resumable.

    A missing/empty API response is treated as a failure rather than valid
    data. This is important for scheduled incremental updates: the
    orchestrator must know that the route did not actually catch up.
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    chunks = []
    current_start = start_dt

    while current_start <= end_dt:
        current_end = min(
            current_start + pd.Timedelta(days=HIST_CHUNK_DAYS - 1),
            end_dt,
        )

        chunk_path = historical_chunk_path(
            airport_icao,
            current_start,
            current_end,
        )
        chunk_cached = load_or_none(chunk_path)

        if chunk_cached is not None:
            print(
                f"  {airport_icao}: cached chunk "
                f"{current_start.strftime('%Y-%m-%d')} -> "
                f"{current_end.strftime('%Y-%m-%d')}"
            )
            chunks.append(chunk_cached)
            current_start = current_end + pd.Timedelta(days=1)
            continue

        print(
            f"  {airport_icao}: fetching "
            f"{current_start.strftime('%Y-%m-%d')} -> "
            f"{current_end.strftime('%Y-%m-%d')}"
        )

        logger.info(
            "FETCH airport=%s start=%s end=%s",
            airport_icao,
            current_start.strftime("%Y-%m-%d"),
            current_end.strftime("%Y-%m-%d"),
        )

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": current_start.strftime("%Y-%m-%d"),
            "end_date": current_end.strftime("%Y-%m-%d"),
            "hourly": ",".join(HOURLY_VARS),
            "timezone": "UTC",
        }

        try:
            data = _request_historical_chunk(params)
        except Exception:
            logger.exception(
                "FAILED airport=%s start=%s end=%s",
                airport_icao,
                current_start.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d"),
            )
            raise

        hourly = data.get("hourly", {})
        df_chunk = pd.DataFrame(hourly)

        if df_chunk.empty or "time" not in df_chunk.columns:
            msg = (
                f"Open-Meteo returned no usable weather data for "
                f"{airport_icao} "
                f"{current_start.strftime('%Y-%m-%d')} -> "
                f"{current_end.strftime('%Y-%m-%d')}"
            )
            print(f"  ERROR: {msg}")
            logger.error(msg)
            raise RuntimeError(msg)

        df_chunk["time"] = pd.to_datetime(df_chunk["time"], utc=True)
        df_chunk["airport"] = airport_icao

        save_parquet(df_chunk, chunk_path)
        chunks.append(df_chunk)

        logger.info(
            "SUCCESS airport=%s start=%s end=%s rows=%s",
            airport_icao,
            current_start.strftime("%Y-%m-%d"),
            current_end.strftime("%Y-%m-%d"),
            len(df_chunk),
        )

        current_start = current_end + pd.Timedelta(days=1)

    if not chunks:
        raise RuntimeError(
            f"No historical weather chunks available for {airport_icao} "
            f"between {start_date} and {end_date}."
        )

    df = pd.concat(chunks, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = (
        df.drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )
    df["airport"] = airport_icao

    return df


def fetch_live_weather(lat: float, lon: float, forecast_hours: int = 48) -> pd.DataFrame:

    """Live/forecast weather for the dashboard's live predictor. Not cached
    to disk since it's time-sensitive by definition."""

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "forecast_hours": forecast_hours,
        "timezone": "UTC",
    }

    resp = requests.get(
        LIVE_WEATHER_URL,
        params=params,
        timeout=30,
    )

    if not resp.ok:
        logger.error(
            "Live Open-Meteo HTTP %s: %s",
            resp.status_code,
            resp.text[:500],
        )
        print(f"Open-Meteo error {resp.status_code}: {resp.text}")
        resp.raise_for_status()

    data = resp.json()
    df = pd.DataFrame(data.get("hourly", {}))

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)

    return df


def nearest_hour_weather(weather_df: pd.DataFrame, ts_utc: pd.Timestamp) -> dict:

    """Look up the weather row closest to a given UTC timestamp. Returns an
    empty dict rather than raising if weather_df is empty."""

    if weather_df is None or weather_df.empty:
        return {}

    idx = (weather_df["time"] - ts_utc).abs().idxmin()
    return weather_df.loc[idx].to_dict()


def _get_airport_info(airport_icao: str) -> dict:

    """Look up airport configuration regardless of whether the airport is
    listed as an origin or destination."""

    if airport_icao in config.ORIGIN_AIRPORTS:
        return config.ORIGIN_AIRPORTS[airport_icao]
    if airport_icao in config.DEST_AIRPORTS:
        return config.DEST_AIRPORTS[airport_icao]

    raise ValueError(
        f"Airport {airport_icao} was not found in "
        f"config.ORIGIN_AIRPORTS or config.DEST_AIRPORTS."
    )


def _fetch_airport_worker(item):

    """
    Unpack airport configuration details, download historical weather metrics 
    via a network request, and record operational logs for tracking.
    """

    icao, info, start_date, end_date = item

    print(
        f"Fetching historical weather for {icao} ({info['name']})..."
    )
    logger.info(
        "AIRPORT_START airport=%s start=%s end=%s",
        icao,
        start_date,
        end_date,
    )

    try:
        df = fetch_historical_weather(
            icao,
            info["lat"],
            info["lon"],
            start_date,
            end_date,
        )
    except Exception:
        logger.exception(
            "AIRPORT_FAILED airport=%s start=%s end=%s",
            icao,
            start_date,
            end_date,
        )
        raise

    logger.info(
        "AIRPORT_SUCCESS airport=%s start=%s end=%s rows=%s",
        icao,
        start_date,
        end_date,
        len(df),
    )

    return icao, df


def build_route_weather(origin_icao: str, destination_icao: str, start_date: str, end_date: str) -> dict:

    """
    Fetch historical weather for both airports in a route, for the given
    date range. Airport pulls run sequentially and each one's chunks are
    cached independently.

    NOTE: pulls are sequential to not overwhelm the Open-Meteo API

    If either airport fails, the route-level call fails. Already-cached
    chunks remain on disk and will be reused by the next attempt.
    """
    airports = {
        origin_icao: _get_airport_info(origin_icao),
        destination_icao: _get_airport_info(destination_icao),
    }

    logger.info(
        "ROUTE_START route=%s start=%s end=%s",
        config.route_key(origin_icao, destination_icao),
        start_date,
        end_date,
    )

    result = {}

    try:
        for icao, info in airports.items():
            # Unpack the returned tuple (icao, df)
            _, df = _fetch_airport_worker(
                (icao, info, start_date, end_date)
            )
            result[icao] = df
            logger.info(
                "AIRPORT_SUCCESS airport=%s rows=%d",
                icao,
                len(df),  
            )

    except Exception:
        logger.exception(
            "ROUTE_FAILED route=%s start=%s end=%s",
            config.route_key(origin_icao, destination_icao),
            start_date,
            end_date,
        )
        raise

    logger.info(
        "ROUTE_SUCCESS route=%s start=%s end=%s",
        config.route_key(origin_icao, destination_icao),
        start_date,
        end_date,
    )

    return result


def get_last_weather_date(origin_icao: str, destination_icao: str):

    """Max hourly timestamp already stored in this route's weather.parquet,
    as a date, or None if nothing's been built yet."""

    path = config.route_weather_path(origin_icao, destination_icao)
    df = load_or_none(path)

    if df is None or df.empty or "time" not in df.columns:
        return None

    return pd.to_datetime(df["time"], utc=True).max().date()


def create_route_dataset(origin_icao: str, destination_icao: str,start_date: str = None, end_date: str = None) -> pd.DataFrame:

    """
    Create (or incrementally extend) the historical weather dataset for one
    route.

    Existing data is merged rather than overwritten. API failures propagate
    to the caller so the daily orchestrator can record the route as failed
    and apply its retry backoff.
    """
    start_date = start_date or config.HIST_START_DATE
    end_date = end_date or config.HIST_END_DATE

    path = config.route_weather_path(origin_icao, destination_icao)
    route = config.route_key(origin_icao, destination_icao)

    print(
        f"Fetching weather for {origin_icao} -> {destination_icao} "
        f"({start_date} -> {end_date})..."
    )

    logger.info(
        "DATASET_START route=%s start=%s end=%s",
        route,
        start_date,
        end_date,
    )

    try:
        weather_dfs = build_route_weather(
            origin_icao,
            destination_icao,
            start_date,
            end_date,
        )

        new_frames = [
            d for d in weather_dfs.values()
            if d is not None and not d.empty
        ]

        if not new_frames:
            raise RuntimeError(
                f"No weather data retrieved for route {route} "
                f"between {start_date} and {end_date}."
            )

        new_df = pd.concat(new_frames, ignore_index=True)
        new_df["time"] = pd.to_datetime(new_df["time"], utc=True)

        existing = load_or_none(path)

        if existing is not None and not existing.empty:
            route_df = pd.concat(
                [existing, new_df],
                ignore_index=True,
            )
        else:
            route_df = new_df

        route_df = (
            route_df
            .drop_duplicates(subset=["airport", "time"])
            .sort_values(["time", "airport"])
            .reset_index(drop=True)
        )

        route_df["route"] = route

        save_parquet(route_df, path)

        print(
            f"Saved route weather dataset: {path} "
            f"({len(route_df)} total hourly rows)"
        )

        record_stage_stats(
            "2_fetch_weather",
            route_df,
            extra={
                "route": route,
                "airports": sorted(
                    route_df["airport"].unique().tolist()
                ),
                "date_range": [
                    str(route_df["time"].min()),
                    str(route_df["time"].max()),
                ],
            },
        )

        logger.info(
            "DATASET_SUCCESS route=%s start=%s end=%s rows=%s",
            route,
            start_date,
            end_date,
            len(route_df),
        )

        return route_df

    except Exception:
        logger.exception(
            "DATASET_FAILED route=%s start=%s end=%s",
            route,
            start_date,
            end_date,
        )
        raise


def inspect_route_weather(origin_icao: str, destination_icao: str):

    """Inspect an already-created route weather dataset on disk. Never
    calls the Open-Meteo API."""

    path = config.route_weather_path(origin_icao, destination_icao)

    return inspect_cached_parquet(
        path,
        label=(
            f"Open-Meteo route weather: {origin_icao} -> {destination_icao} "
            f"({config.HIST_START_DATE} -> {config.HIST_END_DATE})"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--route",
        nargs=2,
        metavar=("ORIGIN", "DEST"),
        help=(
            "Only fetch/build this one route instead of every route "
            "in config.ROUTES"
        ),
    )

    parser.add_argument(
        "--start",
        help="Start date YYYY-MM-DD (defaults to config.HIST_START_DATE)",
    )

    parser.add_argument(
        "--end",
        help="End date YYYY-MM-DD (defaults to config.HIST_END_DATE)",
    )

    parser.add_argument(
        "--inspect",
        action="store_true",
        help=(
            "Just show the already-created route dataset(s) on disk "
            "-- does not call the Open-Meteo API."
        ),
    )

    args = parser.parse_args()

    routes = (
        [tuple(a.upper() for a in args.route)]
        if args.route
        else config.ROUTES
    )

    if args.inspect:
        for origin, dest in routes:
            inspect_route_weather(origin, dest)
    else:
        for origin, dest in routes:
            create_route_dataset(
                origin,
                dest,
                args.start,
                args.end,
            )