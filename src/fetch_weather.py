"""
Weather data via Open-Meteo.

Two different endpoints on purpose:
  - Historical Forecast API (historical-forecast-api.open-meteo.com) for
    TRAINING data. This mirrors the live Forecast API's model outputs
    (available from ~2021 onward), so it includes fields like visibility
    that the plain ERA5 reanalysis archive (archive-api.open-meteo.com)
    does not reliably provide. Using the same model family for training
    and inference avoids a train/serve schema mismatch.
  - Forecast API (api.open-meteo.com) for LIVE predictions in the dashboard.

Both are free, keyless.
"""
import os
import sys
import argparse
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import (
    cache_path, load_or_none, save_parquet,
    inspect_cached_parquet, inspect_cached_glob,
)

HIST_WEATHER_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LIVE_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS =  [
    "temperature_2m",
    "dewpoint_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "sea_level_pressure",
    "cloud_cover_low",
    "cloud_cover_high",
    "visibility",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "wind_speed_200m",
    "wind_direction_200m",
    "wind_gusts_200m",
    "cape",
    "geopotential_height_850hPa"
]


def fetch_historical_weather(airport_icao: str, lat: float, lon: float,
                              start_date: str, end_date: str) -> pd.DataFrame:
    """Hourly historical weather for one airport across the full date range.
    Safely chunks the 3-year window into annual bins to respect the payload restrictions
    of the Historical Forecast API model archive and avoid HTTP 400 Bad Requests."""
    key = f"{airport_icao}_{start_date}_{end_date}"
    path = cache_path(config.RAW_DIR, "weather_hist", key)
    cached = load_or_none(path)
    if cached is not None:
        return cached

    # Chunk 3 years into annual slices to avoid Open-Meteo payload volume exceptions
    date_range = pd.date_range(start=start_date, end=end_date, freq='YE')
    chunks = []
    current_start = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Establish time bounds and ensure the final absolute end date is caught
    boundaries = [d for d in date_range if d > current_start and d < end_dt] + [end_dt]

    for boundary in boundaries:
        chunk_start_str = current_start.strftime("%Y-%m-%d")
        chunk_end_str = boundary.strftime("%Y-%m-%d")
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": chunk_start_str,
            "end_date": chunk_end_str,
            "hourly": ",".join(HOURLY_VARS),
            "timezone": "UTC",
        }
        
        resp = requests.get(HIST_WEATHER_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        df_chunk = pd.DataFrame(hourly)
        if not df_chunk.empty:
            chunks.append(df_chunk)
            
        current_start = boundary + pd.Timedelta(days=1)

    if not chunks:
        return pd.DataFrame()

    # Consolidate chunks, drop overlapping boundary artifacts, and clean up sorting
    df = pd.concat(chunks, ignore_index=True)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    df["airport"] = airport_icao
    
    save_parquet(df, path)
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
    resp = requests.get(LIVE_WEATHER_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data.get("hourly", {}))
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def nearest_hour_weather(weather_df: pd.DataFrame, ts_utc: pd.Timestamp) -> dict:
    """Look up the weather row closest to a given UTC timestamp. Returns an
    empty dict (rather than raising) if weather_df is empty, so callers can
    proceed with NaN features instead of crashing on a data gap.
    
    NOTE: At 3 years of scale, looping this per flight creates an O(N) bottleneck.
    Use the vectorized batch_join_weather_to_flights function instead for training runs."""
    if weather_df is None or weather_df.empty:
        return {}
    idx = (weather_df["time"] - ts_utc).abs().idxmin()
    row = weather_df.loc[idx]
    return row.to_dict()


def batch_join_weather_to_flights(flights_df: pd.DataFrame, weather_dfs: dict, airport_col: str, time_col: str) -> pd.DataFrame:
    """
    Blazing-fast vectorized binary matching using pd.merge_asof.
    Eliminates the loop overhead of nearest_hour_weather for 3-year feature mapping.
    
    Args:
        flights_df: DataFrame containing flight entries.
        weather_dfs: Dictionary returned by build_all_weather() -> {icao: DataFrame}
        airport_col: Column indicating airport ICAO ('origin_airport' or 'dest_airport')
        time_col: Column tracking arrival/departure datetime ('departure_time' or 'arrival_time')
    """
    flights_processed = flights_df.copy()
    flights_processed[time_col] = pd.to_datetime(flights_processed[time_col], utc=True)
    
    # Pack individual airport dataframes into a unified frame sorted by timestamp
    all_weather = pd.concat(weather_dfs.values(), ignore_index=True)
    all_weather = all_weather.sort_values(by="time")
    
    # Vectorized lookups keyed simultaneously on airport identity and time proximity
    merged = pd.merge_asof(
        flights_processed.sort_values(by=time_col),
        all_weather,
        left_on=time_col,
        right_on="time",
        by=airport_col,
        direction="nearest",
        suffixes=("", "_weather")
    )
    return merged


def _fetch_airport_worker(item):
    """Internal parallel thread pool worker function execution."""
    icao, info = item
    print(f"Fetching historical weather for {icao} ({info['name']})...")
    df = fetch_historical_weather(
        icao, info["lat"], info["lon"],
        config.HIST_START_DATE, config.HIST_END_DATE,
    )
    return icao, df


def build_all_weather() -> dict:
    """Fetch historical weather for every airport referenced in MVP_ROUTES
    (both origins and destinations) using concurrent worker threads to speed up I/O.
    Returns {icao: DataFrame}."""
    airports = {}
    airports.update(config.DEST_AIRPORTS)
    for origin, _ in config.MVP_ROUTES:
        if origin in config.ORIGIN_AIRPORTS:
            airports[origin] = config.ORIGIN_AIRPORTS[origin]

    result = {}
    # Use max_workers=4 to safely maximize transfer speed without hitting request blocks
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = executor.map(_fetch_airport_worker, airports.items())
        for icao, df in futures:
            result[icao] = df
            print(f"  {len(df)} hourly rows retrieved/cached for {icao}")
            
    return result


def inspect_weather(airport_icao: str = None):
    """
    Look at Open-Meteo historical weather that's already cached on disk --
    never calls the API. With no argument, shows every weather_hist_*
    cache file found; pass an ICAO code (e.g. "LGAV") to inspect just that
    airport's cached pull for the configured HIST_START_DATE/HIST_END_DATE.
    """
    if airport_icao:
        key = f"{airport_icao}_{config.HIST_START_DATE}_{config.HIST_END_DATE}"
        path = cache_path(config.RAW_DIR, "weather_hist", key)
        return inspect_cached_parquet(
            path, label=f"Open-Meteo historical weather: {airport_icao} "
                        f"({config.HIST_START_DATE} -> {config.HIST_END_DATE})"
        )
    return inspect_cached_glob(config.RAW_DIR, "weather_hist_*.parquet",
                                label="Open-Meteo historical weather (all cached airports)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true",
                         help="Just show what's already cached on disk (df.info(), "
                              "sample rows, null rates) -- does not call the API.")
    parser.add_argument("--airport", help="Restrict --inspect to one airport ICAO code")
    args = parser.parse_args()

    if args.inspect:
        inspect_weather(args.airport)
    else:
        build_all_weather()
        
