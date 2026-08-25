"""
Pulls historical arrival data for our configured routes (config.ROUTES) from
the OpenSky Network REST API and caches everything to
data/raw/<ORIGIN>_<DEST>/arrivals/ so we never re-spend credits re-fetching a
date range we already have on disk.

OpenSky auth: OAuth2 client-credentials grant (username/password auth is
deprecated for accounts created after March 2025). Get a client id/secret
from your account page at https://opensky-network.org and put them in .env.

Endpoints used:
  - Token:
    https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token
  - Arrivals:
    https://opensky-network.org/api/flights/arrival?airport=...&begin=...&end=...

NOTE ON LIMITS: OpenSky daily API call limit for OAuth2 accounts is ~4000
credits/day. This code is conservative about chunking and pausing to
avoid hitting that limit, but if you see 429s you may need to reduce
OPENSKY_QUERY_CHUNK_DAYS or increase OPENSKY_REQUEST_PAUSE_SECONDS in
config.py.

Usage:
    python -m src.fetch_opensky
    python -m src.fetch_opensky --route EDDF LGTS
    python -m src.fetch_opensky --start 2026-07-01 --end 2026-07-31
    python -m src.fetch_opensky --inspect
"""

import os
import sys
import argparse
import datetime as dt
import logging

import requests
import pandas as pd


sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import config

from src.utils import (
    daterange_chunks,
    to_unix,
    cache_path,
    load_or_none,
    save_parquet,
    log_credit_usage,
    polite_sleep,
    record_stage_stats,
    inspect_cached_parquet,
    inspect_cached_glob,
)


TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network"
    "/protocol/openid-connect/token"
)

API_BASE = "https://opensky-network.org/api"


_token_cache = {
    "access_token": None,
}


def _get_logger() -> logging.Logger:

    """Create and configure the module logger for console and file output."""

    logger = logging.getLogger("fetch_opensky")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger.addHandler(console)

    log_dir = config.PROCESSED_DIR
    os.makedirs(
        log_dir,
        exist_ok=True,
    )

    file_handler = logging.FileHandler(
        os.path.join(
            log_dir,
            "update.log",
        )
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.propagate = False

    return logger


logger = _get_logger()


def get_token() -> str:

    """Fetch and cache an OpenSky OAuth2 access token"""

    if _token_cache["access_token"]:
        return _token_cache["access_token"]

    if (
        not config.OPENSKY_CLIENT_ID
        or not config.OPENSKY_CLIENT_SECRET
    ):
        raise RuntimeError(
            "Missing OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET. "
            "Copy .env.example to .env and fill in your credentials."
        )

    logger.info(
        "Requesting new OpenSky OAuth2 access token."
    )

    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": config.OPENSKY_CLIENT_ID,
            "client_secret": config.OPENSKY_CLIENT_SECRET,
        },
        timeout=config.OPENSKY_AUTH_TIMEOUT_SECONDS,
    )

    resp.raise_for_status()

    token = resp.json()["access_token"]

    _token_cache["access_token"] = token

    logger.info(
        "OpenSky OAuth2 token acquired."
    )

    return token


def _get(endpoint: str, params: dict) -> list:

    """Send an authenticated OpenSky API request and handle token expiry and empty responses."""

    def make_request(token: str):
        return requests.get(
            f"{API_BASE}/{endpoint}",
            params=params,
            headers={
                "Authorization": f"Bearer {token}"
            },
            timeout=config.OPENSKY_API_TIMEOUT_SECONDS,
        )

    token = get_token()

    resp = make_request(token)

    # Token expired: refresh once and retry.
    if resp.status_code == 401:
        logger.warning(
            "OpenSky token expired. "
            "Refreshing token and retrying request."
        )

        _token_cache["access_token"] = None

        token = get_token()
        resp = make_request(token)

        if resp.status_code == 401:
            raise RuntimeError(
                "OpenSky authentication failed after token refresh. "
                "Check client credentials."
            )

    # OpenSky uses 404 to indicate no flights in the requested window.
    if resp.status_code == 404:
        return []

    resp.raise_for_status()

    return resp.json()


def fetch_arrivals_for_airport(airport_icao: str, start_date: str, end_date: str, cache_dir: str,) -> pd.DataFrame:

    """
    Fetch arrivals for one airport over a date range using cached chunks
    where available, retrying temporary API failures and caching successful
    responses including empty results.
    """
    all_rows = []
    chunks_failed = 0

    for chunk_start, chunk_end in daterange_chunks(
        start_date,
        end_date,
        config.OPENSKY_QUERY_CHUNK_DAYS,
    ):
        key = (
            f"{airport_icao}_"
            f"{chunk_start}_"
            f"{chunk_end}"
        )

        path = cache_path(
            cache_dir,
            "arrivals",
            key,
        )

        cached = load_or_none(path)

        if cached is not None:
            logger.info(
                "%s: cached chunk %s -> %s (%d rows)",
                airport_icao,
                chunk_start,
                chunk_end,
                len(cached),
            )

            if not cached.empty:
                all_rows.append(cached)

            continue

        logger.info(
            "%s: fetching chunk %s -> %s",
            airport_icao,
            chunk_start,
            chunk_end,
        )

        begin_ts = to_unix(
            chunk_start,
            0,
            0,
        )

        end_ts = to_unix(
            chunk_end,
            23,
            59,
        )

        data = None
        max_retries = 3

        for attempt in range(max_retries):
            try:
                data = _get(
                    "flights/arrival",
                    {
                        "airport": airport_icao,
                        "begin": begin_ts,
                        "end": end_ts,
                    },
                )

                break

            except requests.exceptions.HTTPError as e:
                status = (
                    e.response.status_code
                    if e.response is not None
                    else None
                )

                body = (
                    e.response.text[:300]
                    if e.response is not None
                    else str(e)
                )

                if status == 429:
                    if attempt == max_retries - 1:
                        raise RuntimeError(
                            f"OpenSky rate limit persisted after "
                            f"{max_retries} retries for "
                            f"{airport_icao} "
                            f"{chunk_start}->{chunk_end}."
                        ) from e

                    wait = (
                        config.OPENSKY_REQUEST_PAUSE_SECONDS
                        * (2 ** (attempt + 1))
                    )

                    logger.warning(
                        "%s: HTTP 429 for %s->%s. "
                        "Backing off %.1fs "
                        "(attempt %d/%d).",
                        airport_icao,
                        chunk_start,
                        chunk_end,
                        wait,
                        attempt + 1,
                        max_retries,
                    )

                    polite_sleep(wait)
                    continue

                if status == 400:
                    raise RuntimeError(
                        f"OpenSky returned HTTP 400 for "
                        f"{airport_icao} "
                        f"{chunk_start}->{chunk_end}: "
                        f"{body}"
                    ) from e

                logger.error(
                    "%s: HTTP %s for %s->%s: %s",
                    airport_icao,
                    status,
                    chunk_start,
                    chunk_end,
                    body,
                )

                raise

            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ) as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"OpenSky request failed after "
                        f"{max_retries} retries for "
                        f"{airport_icao} "
                        f"{chunk_start}->{chunk_end}: {e}"
                    ) from e

                wait = (
                    config.OPENSKY_REQUEST_PAUSE_SECONDS
                    * (2 ** (attempt + 1))
                )

                logger.warning(
                    "%s: %s for %s->%s. "
                    "Backing off %.1fs "
                    "(attempt %d/%d).",
                    airport_icao,
                    type(e).__name__,
                    chunk_start,
                    chunk_end,
                    wait,
                    attempt + 1,
                    max_retries,
                )

                polite_sleep(wait)

        if data is None:
            raise RuntimeError(
                "OpenSky returned no usable response for "
                f"{airport_icao} "
                f"{chunk_start}->{chunk_end}."
            )

        if not data:
            logger.info(
                "%s: no arrivals found between %s and %s.",
                airport_icao,
                chunk_start,
                chunk_end,
            )

            df_chunk = pd.DataFrame()

        else:
            df_chunk = pd.DataFrame(data)

            logger.info(
                "%s: received %d arrivals for %s -> %s.",
                airport_icao,
                len(df_chunk),
                chunk_start,
                chunk_end,
            )

        # Only successful API responses reach this point.
        # Therefore an empty frame means "successfully checked; no flights."
        save_parquet(
            df_chunk,
            path,
        )

        span_days = (
            chunk_end - chunk_start
        ).days + 1

        log_credit_usage(
            config.RAW_DIR,
            "flights/arrival",
            span_days,
            note=(
                f"{airport_icao} "
                f"{chunk_start}->{chunk_end}, "
                f"{len(df_chunk)} rows"
            ),
        )

        if not df_chunk.empty:
            all_rows.append(df_chunk)

        polite_sleep(
            config.OPENSKY_REQUEST_PAUSE_SECONDS
        )

    if chunks_failed > 0:
        logger.warning(
            "%s: %d chunk(s) failed during this fetch.",
            airport_icao,
            chunks_failed,
        )

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(
        all_rows,
        ignore_index=True,
    )


def _sanity_filter(df: pd.DataFrame,) -> pd.DataFrame:

    """
    Remove flight records missing required fields or having implausible
    flight durations outside the configured 5-minute to 5-hour range.
    """
    df = df.dropna(
        subset=[
            "firstSeen",
            "lastSeen",
            "estDepartureAirport",
        ]
    )

    duration_min = (
        df["lastSeen"] - df["firstSeen"]
    ) / 60.0

    return df[
        (duration_min > 5)
        & (duration_min <= 300)
    ].copy()


def _last_cached_chunk_date(airport_icao: str, cache_dir: str,):

    """Find the latest end date represented by successfully cached arrival chunks."""

    if not os.path.isdir(cache_dir):
        return None

    latest = None

    prefix = (
        f"arrivals_{airport_icao}_"
    )

    for filename in os.listdir(cache_dir):
        if (
            not filename.startswith(prefix)
            or not filename.endswith(".parquet")
        ):
            continue

        stem = filename[:-8]
        parts = stem.split("_")

        if len(parts) < 4:
            continue

        try:
            chunk_end = dt.date.fromisoformat(
                parts[-1]
            )

        except ValueError:
            continue

        latest = (
            max(latest, chunk_end)
            if latest
            else chunk_end
        )

    return latest


def get_last_flight_date(origin_icao: str, destination_icao: str,):

    """
    Determine the latest date successfully checked for a route using both
    processed flight data and raw cached arrival chunks.
    """
    path = config.route_flights_path(
        origin_icao,
        destination_icao,
    )

    df = load_or_none(path)

    processed_date = None

    if (
        df is not None
        and not df.empty
        and "firstSeen" in df.columns
    ):
        processed_date = (
            pd.to_datetime(
                df["firstSeen"],
                unit="s",
                utc=True,
            )
            .max()
            .date()
        )

    cache_dir = config.route_arrivals_dir(
        origin_icao,
        destination_icao,
    )

    cached_date = _last_cached_chunk_date(
        destination_icao,
        cache_dir,
    )

    dates = [
        d
        for d in [
            processed_date,
            cached_date,
        ]
        if d is not None
    ]

    return max(dates) if dates else None


def build_route_dataset(origin_icao: str, destination_icao: str, start_date: str = None, end_date: str = None,) -> pd.DataFrame:

    """
    Fetch and process arrivals for one route, merge them with any existing
    processed dataset, remove duplicate flights, and save the updated
    route-level parquet dataset.
    """
    start_date = (
        start_date
        or config.HIST_START_DATE
    )

    end_date = (
        end_date
        or config.HIST_END_DATE
    )

    cache_dir = config.route_arrivals_dir(
        origin_icao,
        destination_icao,
    )

    flights_path = config.route_flights_path(
        origin_icao,
        destination_icao,
    )

    logger.info(
        "Fetching route %s -> %s (%s -> %s).",
        origin_icao,
        destination_icao,
        start_date,
        end_date,
    )

    df = fetch_arrivals_for_airport(
        destination_icao,
        start_date,
        end_date,
        cache_dir=cache_dir,
    )

    if df.empty:
        logger.info(
            "No arrival data returned for destination %s "
            "in this range.",
            destination_icao,
        )

        existing = load_or_none(
            flights_path
        )

        return (
            existing
            if existing is not None
            else pd.DataFrame()
        )

    # Keep only flights whose estimated departure airport
    # matches the origin airport of this route.
    df = df[
        df["estDepartureAirport"]
        == origin_icao
    ].copy()

    if df.empty:
        logger.info(
            "No flights found for route %s -> %s "
            "in this range.",
            origin_icao,
            destination_icao,
        )

        existing = load_or_none(
            flights_path
        )

        return (
            existing
            if existing is not None
            else pd.DataFrame()
        )

    # Add route metadata to each retained flight.
    df["destination_airport"] = (
        destination_icao
    )

    df["route"] = (
        f"{origin_icao}->{destination_icao}"
    )

    n_before_sanity_filter = len(df)

    df = _sanity_filter(df)

    n_after_sanity_filter = len(df)

    existing = load_or_none(
        flights_path
    )

    if (
        existing is not None
        and not existing.empty
    ):
        combined = pd.concat(
            [
                existing,
                df,
            ],
            ignore_index=True,
        )

        dedup_keys = [
            c
            for c in [
                "icao24",
                "firstSeen",
                "lastSeen",
            ]
            if c in combined.columns
        ]

        combined = (
            combined
            .drop_duplicates(
                subset=dedup_keys or None
            )
            .reset_index(drop=True)
        )

    else:
        combined = df

    combined = (
        combined
        .sort_values("firstSeen")
        .reset_index(drop=True)
    )

    save_parquet(
        combined,
        flights_path,
    )

    logger.info(
        "Saved %d total flights for route %s -> %s "
        "(%d new/updated rows this run).",
        len(combined),
        origin_icao,
        destination_icao,
        len(df),
    )

    logger.info(
        "Raw cache: %s",
        cache_dir,
    )

    logger.info(
        "Processed data: %s",
        flights_path,
    )

    record_stage_stats(
        "1_fetch_opensky",
        combined,
        extra={
            "route": (
                f"{origin_icao}->{destination_icao}"
            ),
            "origin": origin_icao,
            "destination": destination_icao,
            "date_range_this_run": [
                start_date,
                end_date,
            ],
            "rows_before_sanity_filter_this_run": (
                n_before_sanity_filter
            ),
            "rows_after_sanity_filter_this_run": (
                n_after_sanity_filter
            ),
            "rows_total_after_merge": len(combined),
        },
    )

    return combined


def inspect_route(origin_icao: str, destination_icao: str, start_date: str = None, end_date: str = None,):

    """
    Inspect cached raw and processed data for one route without making any
    OpenSky API requests.
    """
    cache_dir = config.route_arrivals_dir(
        origin_icao,
        destination_icao,
    )

    flights_path = config.route_flights_path(
        origin_icao,
        destination_icao,
    )

    print(
        f"\n{'=' * 80}\n"
        f"ROUTE INSPECTION: "
        f"{origin_icao} -> {destination_icao}\n"
        f"{'=' * 80}"
    )

    print(
        f"\nRaw arrivals directory:\n"
        f"{cache_dir}"
    )

    if start_date and end_date:
        frames = []

        for chunk_start, chunk_end in daterange_chunks(
            start_date,
            end_date,
            config.OPENSKY_QUERY_CHUNK_DAYS,
        ):
            key = (
                f"{destination_icao}_"
                f"{chunk_start}_"
                f"{chunk_end}"
            )

            path = cache_path(
                cache_dir,
                "arrivals",
                key,
            )

            df = inspect_cached_parquet(
                path,
                label=(
                    f"RAW "
                    f"{origin_icao}->{destination_icao} "
                    f"{chunk_start}->{chunk_end}"
                ),
            )

            if (
                df is not None
                and not df.empty
            ):
                frames.append(df)

        raw_combined = (
            pd.concat(
                frames,
                ignore_index=True,
            )
            if frames
            else pd.DataFrame()
        )

        if frames:
            print(
                f"\n{'=' * 80}\n"
                f"COMBINED RAW DATA: "
                f"{origin_icao}->{destination_icao}\n"
                f"{'=' * 80}"
            )

            print(
                f"{len(raw_combined)} rows across "
                f"{len(frames)} non-empty chunk(s)."
            )

        else:
            print(
                f"\nNo raw cached data found for "
                f"{origin_icao}->{destination_icao} "
                f"within {start_date}->{end_date}."
            )

    else:
        raw_combined = inspect_cached_glob(
            cache_dir,
            "arrivals_*.parquet",
            label=(
                f"RAW OpenSky arrivals: "
                f"{origin_icao}->{destination_icao}"
            ),
        )

    print(
        f"\nProcessed dataset:\n"
        f"{flights_path}"
    )

    processed = inspect_cached_parquet(
        flights_path,
        label=(
            f"PROCESSED: "
            f"{origin_icao}->{destination_icao} "
            f"(sanity-filtered)"
        ),
    )

    print(
        f"\n{'=' * 80}\n"
        f"ROUTE SUMMARY: "
        f"{origin_icao} -> {destination_icao}\n"
        f"{'=' * 80}"
    )

    print(
        f"Raw rows inspected: {len(raw_combined)}"
    )

    print(
        "Processed rows:     "
        f"{len(processed) if processed is not None else 0}"
    )

    last_date = get_last_flight_date(
        origin_icao,
        destination_icao,
    )

    print(
        "Last stored flight date: "
        f"{last_date if last_date else 'n/a'}"
    )

    return {
        "raw": raw_combined,
        "processed": processed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--route",
        nargs=2,
        metavar=("ORIGIN", "DEST"),
        help=(
            "Only fetch/build this one route "
            "instead of every route in config.ROUTES"
        ),
    )

    parser.add_argument(
        "--start",
        help=(
            "Start date YYYY-MM-DD "
            "(defaults to config.HIST_START_DATE)"
        ),
    )

    parser.add_argument(
        "--end",
        help=(
            "End date YYYY-MM-DD "
            "(defaults to config.HIST_END_DATE)"
        ),
    )

    parser.add_argument(
        "--inspect",
        action="store_true",
        help=(
            "Just show what's already cached on disk "
            "-- does not call the API."
        ),
    )

    args = parser.parse_args()

    routes = (
        [tuple(args.route)]
        if args.route
        else config.ROUTES
    )

    if args.inspect:
        for origin, dest in routes:
            inspect_route(
                origin,
                dest,
                args.start,
                args.end,
            )

    else:
        for origin, dest in routes:
            build_route_dataset(
                origin,
                dest,
                args.start,
                args.end,
            )