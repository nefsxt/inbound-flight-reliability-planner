"""
Builds the final pooled flight/weather dataframe from raw OpenSky flights +
Open-Meteo weather data for every route in config.ROUTES.

This module is intentionally a DATA PREPARATION module, not an ML
preprocessing/training module.

The output of build_feature_matrix() is the final `flights` dataframe that
can subsequently be passed to the ML training pipeline.

IMPORTANT DATA-LEAKAGE RULE
---------------------------
Nothing in this module learns a value from the full flight population.

In particular, this module does NOT calculate:

- route medians
- target variables
- target anomalies
- imputers
- scalers
- encoders learned from the data
- feature selection
- seasonal features derived during the training process
- any cross-fold statistics

Those operations belong after the chronological train/validation/test split
and must be fitted using training data only.

The weather join is not a learned operation. It is an exact lookup against
the cached Open-Meteo observations using rounded UTC hourly keys.

OUTPUT
------
The final dataframe has exactly these 53 columns:

    icao24
    firstSeen
    estDepartureAirport
    lastSeen
    estArrivalAirport
    callsign
    estDepartureAirportHorizDistance
    estDepartureAirportVertDistance
    estArrivalAirportHorizDistance
    estArrivalAirportVertDistance
    departureAirportCandidatesCount
    arrivalAirportCandidatesCount
    destination_airport
    route
    firstSeen_hourly_utc
    lastSeen_hourly_utc
    dep_merge_key
    arr_merge_key
    dep_temperature_2m
    dep_relative_humidity_2m
    dep_dew_point_2m
    dep_precipitation
    dep_weather_code
    dep_pressure_msl
    dep_cloud_cover_low
    dep_cloud_cover_high
    dep_visibility
    dep_wind_speed_10m
    dep_wind_speed_180m
    dep_wind_direction_10m
    dep_wind_direction_180m
    dep_cape
    dep_geopotential_height_850hPa
    arr_temperature_2m
    arr_relative_humidity_2m
    arr_dew_point_2m
    arr_precipitation
    arr_weather_code
    arr_pressure_msl
    arr_cloud_cover_low
    arr_cloud_cover_high
    arr_visibility
    arr_wind_speed_10m
    arr_wind_speed_180m
    arr_wind_direction_10m
    arr_wind_direction_180m
    arr_cape
    arr_geopotential_height_850hPa
    flight_duration
    airline
    hour_of_day
    month
    flight_date_parsed

No additional ML-derived columns are written.

The final features will be chosen within the train_model.py script because some important features must be calculated per fold.
"""

import os
import sys

import numpy as np
import pandas as pd


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.utils import save_parquet, record_stage_stats
from src.airlines import extract_airline_code


# ---------------------------------------------------------------------------
# Weather schema
# ---------------------------------------------------------------------------

# The exact 15 Open-Meteo hourly variables used by the notebook.
WEATHER_FEATURES = [
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

DEPARTURE_WEATHER_COLUMNS = [
    f"dep_{column}" for column in WEATHER_FEATURES
]

ARRIVAL_WEATHER_COLUMNS = [
    f"arr_{column}" for column in WEATHER_FEATURES
]


# ---------------------------------------------------------------------------
# Final schema
# ---------------------------------------------------------------------------

FINAL_COLUMNS = [
    "icao24",
    "firstSeen",
    "estDepartureAirport",
    "lastSeen",
    "estArrivalAirport",
    "callsign",
    "estDepartureAirportHorizDistance",
    "estDepartureAirportVertDistance",
    "estArrivalAirportHorizDistance",
    "estArrivalAirportVertDistance",
    "departureAirportCandidatesCount",
    "arrivalAirportCandidatesCount",
    "destination_airport",
    "route",
    "firstSeen_hourly_utc",
    "lastSeen_hourly_utc",
    "dep_merge_key",
    "arr_merge_key",
    *DEPARTURE_WEATHER_COLUMNS,
    *ARRIVAL_WEATHER_COLUMNS,
    "flight_duration",
    "airline",
    "hour_of_day",
    "month",
    "flight_date_parsed",
]


# Columns that must exist in the raw OpenSky dataframe before feature
# construction begins.
REQUIRED_FLIGHT_COLUMNS = [
    "icao24",
    "firstSeen",
    "estDepartureAirport",
    "lastSeen",
    "estArrivalAirport",
    "callsign",
    "estDepartureAirportHorizDistance",
    "estDepartureAirportVertDistance",
    "estArrivalAirportHorizDistance",
    "estArrivalAirportVertDistance",
    "departureAirportCandidatesCount",
    "arrivalAirportCandidatesCount",
    "destination_airport",
    "route",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_required_columns(df: pd.DataFrame, required_columns: list[str], dataframe_name: str,) -> None:

    """
    Raises a clear error if a dataframe does not contain the columns required
    by the next pipeline stage.
    """
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required columns: {missing}"
        )


def parse_opensky_timestamp(series: pd.Series, column_name: str,) -> pd.Series:

    """
    Converts an OpenSky timestamp column into timezone-aware UTC datetimes.

    OpenSky data can arrive either as:

        - Unix timestamps in seconds
        - pandas datetime values

    The distinction is important.

    Calling pd.to_datetime(..., utc=True) on Unix-second integers without
    specifying unit='s' interprets them as nanoseconds and can silently
    produce dates around 1970.

    Conversely, supplying unit='s' to an already-datetime Series is invalid.

    This function therefore handles the two representations explicitly and
    always returns datetime64[ns, UTC]-compatible values.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        result = pd.to_datetime(series, utc=True, errors="coerce")
    elif pd.api.types.is_numeric_dtype(series):
        result = pd.to_datetime(
            series,
            unit="s",
            utc=True,
            errors="coerce",
        )
    else:
        result = pd.to_datetime(
            series,
            utc=True,
            errors="coerce",
        )

    if result.isna().any():
        invalid_count = int(result.isna().sum())

        raise ValueError(
            f"{column_name} contains {invalid_count} values that could not "
            "be parsed as UTC timestamps."
        )

    return result


def validate_timestamp_range(df: pd.DataFrame) -> None:

    """
    Validates the fundamental temporal relationship of each flight.

    A flight must:

        firstSeen < lastSeen

    Zero-duration and negative-duration observations are invalid and are
    rejected rather than silently repaired.
    """
    invalid_mask = df["lastSeen"] <= df["firstSeen"]

    if invalid_mask.any():
        invalid_count = int(invalid_mask.sum())

        examples = (
            df.loc[
                invalid_mask,
                ["icao24", "firstSeen", "lastSeen", "route"],
            ]
            .head(5)
            .to_dict("records")
        )

        raise ValueError(
            f"Found {invalid_count} flights with lastSeen <= firstSeen. "
            f"Examples: {examples}"
        )


def validate_no_duplicate_flights(df: pd.DataFrame) -> None:

    """
    Rejects duplicate flight observations.

    The combination below is used as the practical flight identity because
    a single icao24 can perform multiple flights over the dataset period.
    """
    duplicate_mask = df.duplicated(
        subset=[
            "icao24",
            "firstSeen",
            "lastSeen",
            "estDepartureAirport",
            "estArrivalAirport",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())

        examples = (
            df.loc[
                duplicate_mask,
                [
                    "icao24",
                    "firstSeen",
                    "lastSeen",
                    "estDepartureAirport",
                    "estArrivalAirport",
                ],
            ]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )

        raise ValueError(
            f"Found {duplicate_count} duplicated flight observations. "
            f"Examples: {examples}"
        )


# ---------------------------------------------------------------------------
# Flight preparation
# ---------------------------------------------------------------------------

def prepare_flights(df: pd.DataFrame) -> pd.DataFrame:

    """
    Validates and normalizes the raw OpenSky flight dataframe.

    No population-level statistics are calculated here.

    The function:

        1. validates the raw schema
        2. converts firstSeen/lastSeen to UTC
        3. validates chronological ordering
        4. rejects duplicate observations
        5. creates rounded hourly UTC timestamps
        6. creates timezone-aware UTC hourly merge keys used by the weather join
        7. creates hour_of_day and month
        8. creates flight_duration
        9. creates the airline feature
        10. creates flight_date_parsed

    The temporary departure_utc / arrival_utc columns are removed before
    returning the dataframe.
    """
    df = df.copy()

    validate_required_columns(
        df,
        REQUIRED_FLIGHT_COLUMNS,
        "Flights",
    )

    # Normalize OpenSky timestamps safely. This explicitly avoids the
    # seconds-vs-nanoseconds timestamp bug.
    df["firstSeen"] = parse_opensky_timestamp(
        df["firstSeen"],
        "firstSeen",
    )

    df["lastSeen"] = parse_opensky_timestamp(
        df["lastSeen"],
        "lastSeen",
    )

    validate_timestamp_range(df)
    validate_no_duplicate_flights(df)

    # ------------------------------------------------------------------
    # Rounded UTC timestamps
    # ------------------------------------------------------------------

    df["firstSeen_hourly_utc"] = df["firstSeen"].dt.round("h")
    df["lastSeen_hourly_utc"] = df["lastSeen"].dt.round("h")

    # Timezone-aware UTC hourly merge keys are retained in the final dataframe.
    # Both flight and weather keys use the same datetime representation so that
    #the MultiIndex weather lookup is type-safe.
    
    df["dep_merge_key"] = df["firstSeen_hourly_utc"]
    df["arr_merge_key"] = df["lastSeen_hourly_utc"]

    # ------------------------------------------------------------------
    # Time features.
    #
    # These are direct calendar properties of departure time.
    # ------------------------------------------------------------------

    df["hour_of_day"] = df["firstSeen"].dt.hour
    df["month"] = df["firstSeen"].dt.month

    # Date representation used for chronological splitting downstream.
    df["flight_date_parsed"] = (
        df["firstSeen"]
        .dt.normalize()
        .dt.date
    )

    df["flight_date_parsed"] = pd.to_datetime(
        df["flight_date_parsed"]
    )

    # ------------------------------------------------------------------
    # Actual flight duration.
    #
    # This is an observation-derived quantity, not a target transformation.
    #
    # ------------------------------------------------------------------

    df["flight_duration"] = (
        df["lastSeen"] - df["firstSeen"]
    ).dt.total_seconds() / 60.0

    # ------------------------------------------------------------------
    # Airline.
    #
    # Keep the requested final column name `airline`.
    # ------------------------------------------------------------------

    if "callsign" in df.columns:
        df["airline"] = df["callsign"].apply(extract_airline_code)
    else:
        df["airline"] = None

    return df


# ---------------------------------------------------------------------------
# Weather preparation
# ---------------------------------------------------------------------------

def prepare_weather_index(weather_df: pd.DataFrame,) -> pd.DataFrame:

    """
    Builds the exact (airport, rounded UTC hour) weather lookup index.

    The weather dataframe is expected to contain:

        airport
        time
        WEATHER_FEATURES

    Weather timestamps are normalized to UTC and rounded to the nearest hour,
    matching the notebook's merge logic.

    Duplicate (airport, hour) observations are rejected because allowing
    them would make the flight-to-weather join ambiguous.
    """
    weather_df = weather_df.copy()

    validate_required_columns(
        weather_df,
        ["airport", "time"],
        "Weather",
    )

    weather_df["time"] = pd.to_datetime(
        weather_df["time"],
        utc=True,
        errors="coerce", # converts bad entris to NaT - can easily be dropped at a later stage
    )

    if weather_df["time"].isna().any():
        invalid_count = int(weather_df["time"].isna().sum())

        raise ValueError(
            f"Weather contains {invalid_count} invalid timestamp values."
        )

    weather_df["weather_merge_key"] = (
        weather_df["time"].dt.round("h")
    )

    # The final flight dataframe has a fixed weather schema. Missing source
    # columns are represented as NaN here, but the final quality validation
    # will reject rows that consequently lack complete weather.
    for column in WEATHER_FEATURES:
        if column not in weather_df.columns:
            weather_df[column] = np.nan

    duplicate_mask = weather_df.duplicated(
        subset=["airport", "weather_merge_key"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())

        examples = (
            weather_df.loc[
                duplicate_mask,
                ["airport", "weather_merge_key"],
            ]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )

        raise ValueError(
            "Weather contains duplicate (airport, weather_merge_key) "
            f"keys. Found {duplicate_count} affected rows. "
            f"Examples: {examples}"
        )

    return weather_df.set_index(
        ["airport", "weather_merge_key"]
    )[WEATHER_FEATURES]


# ---------------------------------------------------------------------------
# Weather merge
# ---------------------------------------------------------------------------

def merge_route_weather(flights_df: pd.DataFrame, weather_df: pd.DataFrame, origin_icao: str, destination_icao: str,) -> pd.DataFrame:

    """
    Adds departure and arrival weather using a MultiIndex-based merge.

    """
    df = flights_df.copy()

    required = [
        "estDepartureAirport",
        "estArrivalAirport",
        "firstSeen_hourly_utc",
        "lastSeen_hourly_utc",
    ]

    validate_required_columns(df, required, "Flights")

    w_indexed = prepare_weather_index(weather_df)

    relevant_airports = {origin_icao, destination_icao}

    w_indexed = w_indexed[
        w_indexed.index.get_level_values("airport").isin(
            relevant_airports
        )
    ]
    
    # Left join departure weather on ("airport", "weather_merge_key") == ("estDepartureAirport", "dep_merge_key")
    # Right_index=True tells pandas to match the flights keys directly against the fast weather MultiIndex
    # This retains the speed of index lookups while keeping the original flight column order intact

    departure_weather = w_indexed.add_prefix("dep_")
    df= df.merge(
        departure_weather,
        left_on=["estDepartureAirport", "dep_merge_key"],
        right_index=True,
        how="left",
    )
    
    # Left join arrival weather on ("airport", "weather_merge_key") == ("estArrivalAirport", "arr_merge_key")
    # Right_index=True leverages the same pre-built weather lookup index for maximum memory efficiency

    arrival_weather = w_indexed.add_prefix("arr_")
    df = df.merge(
         arrival_weather ,
        left_on=["estArrivalAirport", "arr_merge_key"],
        right_index=True,
     how="left",
    )


    # Guarantee the expected schema even when an upstream weather
    # variable is absent.
    for column in (
        DEPARTURE_WEATHER_COLUMNS
        + ARRIVAL_WEATHER_COLUMNS
    ):
        if column not in df.columns:
            df[column] = np.nan

    return df

# ---------------------------------------------------------------------------
# Final dataframe validation
# ---------------------------------------------------------------------------

def validate_final_flights(df: pd.DataFrame,) -> pd.DataFrame:

    """
    Performs the final data-quality gate before the dataframe is returned.

    This function deliberately does NOT impute missing values.

    Missing values are a data-quality problem at this stage and should be
    investigated rather than silently filled using statistics calculated
    over the complete dataset.

    The returned dataframe is guaranteed to contain exactly FINAL_COLUMNS.
    """
    missing_columns = [
        column
        for column in FINAL_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Final flights dataframe is missing columns: "
            f"{missing_columns}"
        )

    # Do not allow unexpected columns to leak into the final artifact.
    df = df[FINAL_COLUMNS].copy()

    # ------------------------------------------------------------------
    # Required identifiers / categorical fields
    # ------------------------------------------------------------------

    required_non_null_columns = [
        "icao24",
        "firstSeen",
        "estDepartureAirport",
        "lastSeen",
        "estArrivalAirport",
        "callsign",
        "destination_airport",
        "route",
        "dep_merge_key",
        "arr_merge_key",
        "airline",
        "flight_date_parsed",
    ]

    missing_required = (
        df[required_non_null_columns]
        .isna()
        .any(axis=1)
    )

    if missing_required.any():
        count = int(missing_required.sum())

        examples = (
            df.loc[
                missing_required,
                required_non_null_columns,
            ]
            .head(5)
            .to_dict("records")
        )

        raise ValueError(
            f"Found {count} flights with missing required values. "
            f"Examples: {examples}"
        )

    # ------------------------------------------------------------------
    # Weather completeness
    #
    # For the final dataset we want the complete weather matrix.
    # ------------------------------------------------------------------

    weather_columns = (
        DEPARTURE_WEATHER_COLUMNS
        + ARRIVAL_WEATHER_COLUMNS
    )

    incomplete_weather = (
        df[weather_columns]
        .isna()
        .any(axis=1)
    )

    if incomplete_weather.any():
        count = int(incomplete_weather.sum())

        examples = (
            df.loc[
                incomplete_weather,
                [
                    "icao24",
                    "firstSeen",
                    "lastSeen",
                    "route",
                    "dep_merge_key",
                    "arr_merge_key",
                ],
            ]
            .head(5)
            .to_dict("records")
        )

        raise ValueError(
            f"Found {count} flights with incomplete departure/arrival "
            f"weather. No imputation is performed in features.py. "
            f"Examples: {examples}"
        )

    # ------------------------------------------------------------------
    # Numeric sanity checks
    # ------------------------------------------------------------------

    if (~np.isfinite(df["flight_duration"])).any():
        raise ValueError(
            "flight_duration contains non-finite values."
        )

    if (df["flight_duration"] <= 0).any():
        raise ValueError(
            "flight_duration contains zero or negative durations."
        )

    # ------------------------------------------------------------------
    # Final duplicate check after all joins.
    #
    # The weather merge must never multiply flight rows.
    # ------------------------------------------------------------------

    duplicate_count = int(
        df.duplicated(
            subset=[
                "icao24",
                "firstSeen",
                "lastSeen",
                "estDepartureAirport",
                "estArrivalAirport",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            "The final flights dataframe contains "
            f"{duplicate_count} duplicate flight observations."
        )

    return df


# ---------------------------------------------------------------------------
# Single-route pipeline
# ---------------------------------------------------------------------------

def build_route_features(origin_icao: str, destination_icao: str,) -> pd.DataFrame:

    """
    Builds the final flight/weather dataframe for one route.

    Returns an empty DataFrame when one of the required cached input files
    does not exist so build_feature_matrix() can skip that route.

    No ML preprocessing or population-level statistics are performed.
    """
    flights_path = config.route_flights_path(
        origin_icao,
        destination_icao,
    )

    weather_path = config.route_weather_path(
        origin_icao,
        destination_icao,
    )

    if not os.path.exists(flights_path):
        print(
            f"[SKIP] {origin_icao}->{destination_icao}: "
            f"{flights_path} not found -- run "
            f"`python -m src.fetch_opensky --route "
            f"{origin_icao} {destination_icao}` first."
        )
        return pd.DataFrame()

    if not os.path.exists(weather_path):
        print(
            f"[SKIP] {origin_icao}->{destination_icao}: "
            f"{weather_path} not found -- run "
            f"`python -m src.fetch_weather --route "
            f"{origin_icao} {destination_icao}` first."
        )
        return pd.DataFrame()

    flights_df = pd.read_parquet(flights_path)
    weather_df = pd.read_parquet(weather_path)

    print(
        f"[{origin_icao}->{destination_icao}] "
        f"Loaded {len(flights_df)} raw flights, "
        f"{len(weather_df)} weather rows."
    )

    if flights_df.empty:
        print(
            f"[SKIP] {origin_icao}->{destination_icao}: "
            "flight dataset is empty."
        )
        return pd.DataFrame()

    if weather_df.empty:
        print(
            f"[SKIP] {origin_icao}->{destination_icao}: "
            "weather dataset is empty."
        )
        return pd.DataFrame()

    # 1. Normalize and validate flights.
    flights_df = prepare_flights(flights_df)

    # 2. Exact weather lookup.
    flights_df = merge_route_weather(
        flights_df,
        weather_df,
        origin_icao,
        destination_icao,
    )

    # 3. Final quality gate and exact schema.
    flights_df = validate_final_flights(flights_df)

    return flights_df


# ---------------------------------------------------------------------------
# Pooled pipeline
# ---------------------------------------------------------------------------

def build_feature_matrix() -> pd.DataFrame:

    """
    Builds and pools the final flights dataframe across every route in
    config.ROUTES.

    This is intentionally the end of feature construction.

    The returned dataframe is suitable as the input artifact for the next
    stage, where chronological train/validation/test splitting and all
    training-only transformations must occur.

    The dataframe is saved to config.features_path().
    """
    frames = []

    for origin, destination in config.ROUTES:
        route_df = build_route_features(
            origin,
            destination,
        )

        if not route_df.empty:
            frames.append(route_df)

    if not frames:
        raise RuntimeError(
            "No route produced a final flights dataframe. "
            "Run `python -m src.fetch_opensky` and "
            "`python -m src.fetch_weather` for at least one route "
            "in config.ROUTES first."
        )

    flights = pd.concat(
        frames,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # Final pooled validation.
    #
    # Re-running the final schema/quality gate after concatenation catches
    # accidental route-specific schema differences.
    # ------------------------------------------------------------------

    flights = validate_final_flights(flights)

    # Sort chronologically so the persisted artifact has a deterministic
    # temporal ordering. This does NOT constitute a train/test split.
    flights = (
        flights
        .sort_values(
            ["firstSeen", "icao24"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Stage statistics.
    #
    # These are diagnostics only. Nothing here is used to transform data.
    # ------------------------------------------------------------------

    airline_counts = (
        flights["airline"]
        .value_counts(dropna=False)
        .to_dict()
    )

    record_stage_stats(
        "3_build_features",
        flights,
        extra={
            "routes": sorted(
                flights["route"].unique().tolist()
            ),
            "flights_per_airline": {
                str(key): int(value)
                for key, value in airline_counts.items()
            },
            "pct_rows_with_complete_departure_weather": round(
                100
                * flights[
                    DEPARTURE_WEATHER_COLUMNS
                ]
                .notna()
                .all(axis=1)
                .mean(),
                1,
            ),
            "pct_rows_with_complete_arrival_weather": round(
                100
                * flights[
                    ARRIVAL_WEATHER_COLUMNS
                ]
                .notna()
                .all(axis=1)
                .mean(),
                1,
            ),
            "note": (
                "This stage performs data preparation only. No route "
                "median, target, imputation, scaling, encoding, feature "
                "selection, or other population-level ML transformation "
                "is performed here. All training-dependent operations "
                "must be fitted using training rows only after the "
                "chronological split."
            ),
        },
    )

    # ------------------------------------------------------------------
    # Persist only the final 53-column flights dataframe.
    # ------------------------------------------------------------------

    out_path = config.features_path()

    save_parquet(
        flights,
        out_path,
    )

    print(
        f"Saved final flights dataframe with "
        f"{len(flights)} rows, "
        f"{len(flights.columns)} columns to {out_path}"
    )

    return flights


if __name__ == "__main__":
    build_feature_matrix()