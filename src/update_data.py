"""
Daily incremental data fetch for every route in config.ROUTES.

Each run:

  1. For every route, finds the last date already stored on disk by checking
     both the processed flights and route weather datasets.

  2. Pulls OpenSky arrivals and Open-Meteo weather for the missing range.

  3. Each fetcher is treated as an independent stage. If one succeeds and the
     other fails, the successful stage is kept. The route is marked failed,
     and the next run resumes from whichever source is still behind.

  4. On failure, the route is recorded in the checkpoint and persistent
     per-run log. Other configured routes are still attempted.

  5. A failed route is skipped during the configured backoff window unless
     --force is supplied.

Both fetchers are responsible for their own chunk-level caching and retry
behaviour. This orchestration layer deliberately does not retry failed
fetchers immediately.

State:
    data/processed/update_checkpoint.json

Persistent per-run logs:
    data/processed/update_logs/YYYY-MM-DD_HH-MM-SS.json

Usage:
    python -m src.update_data
    python -m src.update_data --route EDDF LGTS
    python -m src.update_data --force
"""

import os
import sys
import json
import argparse
import datetime as dt
import logging
import tempfile
import config

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


from src.fetch_opensky import (
    build_route_dataset,
    get_last_flight_date,
)
from src.fetch_weather import (
    create_route_dataset,
    get_last_weather_date,
)


CHECKPOINT_PATH = os.path.join(
    config.PROCESSED_DIR,
    "update_checkpoint.json",
)

UPDATE_LOG_DIR = os.path.join(
    config.PROCESSED_DIR,
    "update_logs",
)


def _utc_now() -> dt.datetime: 

    """Return the current timezone-aware UTC timestamp."""

    return dt.datetime.now(dt.timezone.utc)


def _setup_logging():

    """Configure normal console logging for GitHub Actions."""

    logger = logging.getLogger("update_data")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)sZ [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


logger = _setup_logging()


def _write_run_log(run_record: dict):

    """
    Write one compact JSON file describing a completed update run.

    Each run gets its own file, so logs never grow indefinitely.
    """
    os.makedirs(UPDATE_LOG_DIR, exist_ok=True)

    started_at = dt.datetime.fromisoformat(
        run_record["started_utc"]
    )

    filename = (started_at.strftime("%Y-%m-%d_%H-%M-%S") + ".json")

    target_path = os.path.join(UPDATE_LOG_DIR, filename,)

    fd, tmp_path = tempfile.mkstemp(
        prefix="update_run_",
        suffix=".tmp",
        dir=UPDATE_LOG_DIR,
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                run_record,
                f,
                indent=2,
                default=str,
            )

            f.flush()
            os.fsync(f.fileno())

        os.replace(
            tmp_path,
            target_path,
        )

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _load_checkpoint() -> dict:

    """Load route update state from disk."""

    if not os.path.exists(CHECKPOINT_PATH):
        return {}

    try:
        with open(
            CHECKPOINT_PATH,
            encoding="utf-8",
        ) as f:
            return json.load(f)

    except (
        json.JSONDecodeError,
        OSError,
    ) as e:
        logger.warning(
            f"Could not read checkpoint "
            f"{CHECKPOINT_PATH}: {e}"
        )
        return {}


def _save_checkpoint(state: dict):

    """
    Atomically replace the checkpoint file.

    The JSON is flushed and fsynced before replacement.
    """
    checkpoint_dir = os.path.dirname(
        CHECKPOINT_PATH
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    fd, tmp_path = tempfile.mkstemp(
        prefix="update_checkpoint_",
        suffix=".tmp",
        dir=checkpoint_dir,
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                state,
                f,
                indent=2,
                default=str,
            )

            f.flush()
            os.fsync(f.fileno())

        os.replace(
            tmp_path,
            CHECKPOINT_PATH,
        )

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _in_backoff_window(route_state: dict,) -> bool:

    """Return True if the route is still inside its failure cooldown."""

    last_failure = route_state.get("last_failure_utc")

    if not last_failure:
        return False

    failed_at = dt.datetime.fromisoformat(
        last_failure
    )

    if failed_at.tzinfo is None:
        failed_at = failed_at.replace(
            tzinfo=dt.timezone.utc
        )

    elapsed = _utc_now() - failed_at

    return elapsed < dt.timedelta(days=config.UPDATE_RETRY_BACKOFF_DAYS)


def _save_route_state(key: str, route_state: dict,):

    """Update one route's checkpoint state atomically."""

    checkpoint = _load_checkpoint()
    checkpoint[key] = route_state
    _save_checkpoint(checkpoint)


def _stage_state(route_state: dict, stage: str, status: str, error: str = None,):

    """Record the most recent state of one fetcher stage."""

    stages = route_state.setdefault(
        "stages",
        {},
    )

    stages[stage] = {
        "status": status,
        "updated_utc": _utc_now().isoformat(),
        "error": error,
    }


def update_route(origin: str, destination: str, force: bool = False,) -> dict:

    """
    Update OpenSky and Weather independently for one route.

    A successful stage is retained even when the other stage fails.
    """

    key = config.route_key(origin, destination,)

    route_started = _utc_now()

    checkpoint = _load_checkpoint()
    route_state = checkpoint.get(
        key,
        {},
    )

    result = {
        "route": key,
        "status": "failed",
        "open_sky": None,
        "weather": None,
        "started_utc": route_started.isoformat(),
    }

    # --------------------------------------------------------------
    # Backoff
    # --------------------------------------------------------------

    if not force and _in_backoff_window(
        route_state
    ):
        failed_at = dt.datetime.fromisoformat(
            route_state["last_failure_utc"]
        )

        if failed_at.tzinfo is None:
            failed_at = failed_at.replace(
                tzinfo=dt.timezone.utc
            )

        retry_at = failed_at + dt.timedelta(
            days=config.UPDATE_RETRY_BACKOFF_DAYS
        )

        logger.info(
            f"[{key}] Skipping -- last attempt failed at "
            f"{failed_at.isoformat()}, waiting out the "
            f"{config.UPDATE_RETRY_BACKOFF_DAYS}-day backoff "
            f"(next retry at {retry_at.isoformat()})."
        )

        result.update({
            "status": "skipped_backoff",
            "retry_at_utc": retry_at.isoformat(),
        })

        result["duration_seconds"] = (
            _utc_now() - route_started
        ).total_seconds()

        return result

    # --------------------------------------------------------------
    # Determine missing range
    # --------------------------------------------------------------

    today = dt.date.today()

    last_flight_date = get_last_flight_date(origin, destination,)
    last_weather_date = get_last_weather_date(origin, destination,)

    candidate_dates = [
        d
        for d in (
            last_flight_date,
            last_weather_date,
        )
        if d is not None
    ]

    if candidate_dates:
        resume_from = (
            min(candidate_dates)
            + dt.timedelta(days=1)
        )
    else:
        resume_from = None

    start_date = (
        resume_from.isoformat()
        if resume_from
        else None
    )

    end_date = today.isoformat()

    result["requested_start"] = start_date
    result["requested_end"] = end_date

    # --------------------------------------------------------------
    # Already up to date
    # --------------------------------------------------------------

    if (
        resume_from is not None
        and resume_from > today
    ):
        last_stored = max(
            candidate_dates
        )

        logger.info(
            f"[{key}] Already up to date "
            f"(OpenSky={last_flight_date}, "
            f"Weather={last_weather_date})."
        )

        route_state.update({
            "last_success_utc": (
                _utc_now().isoformat()
            ),
            "last_success_date_fetched_through": (
                last_stored.isoformat()
            ),
            "last_failure_utc": None,
            "last_failure_message": None,
        })

        _save_route_state(
            key,
            route_state,
        )

        result["status"] = (
            "already_up_to_date"
        )

        result["duration_seconds"] = (
            _utc_now() - route_started
        ).total_seconds()

        return result

    logger.info(
        f"[{key}] Updating "
        f"{'(first-time backfill)' if start_date is None else start_date} "
        f"-> {end_date}"
    )

    logger.info(
        f"[{key}] Current stored dates: "
        f"OpenSky={last_flight_date}, "
        f"Weather={last_weather_date}"
    )

    # --------------------------------------------------------------
    # OpenSky
    # --------------------------------------------------------------

    logger.info(
        f"[{key}] [OpenSky] Starting fetch."
    )

    try:

        build_route_dataset(origin,destination,start_date=start_date, end_date=end_date,)

        new_last_flight_date = (
            get_last_flight_date(
                origin,
                destination,
            )
        )

        _stage_state(route_state, "opensky", "success",)

        result["open_sky"] = "success"
        result["opensky_stored_through"] = (
            new_last_flight_date.isoformat()
            if new_last_flight_date
            else None
        )

        logger.info(
            f"[{key}] [OpenSky] Completed successfully. "
            f"Stored through {new_last_flight_date}."
        )

    except Exception as e:

        error = str(e)

        _stage_state(route_state, "opensky", "failed", error,)

        logger.exception(
            f"[{key}] [OpenSky] FAILED: {error}"
        )

        route_state.update({
            "last_failure_utc": (
                _utc_now().isoformat()
            ),
            "last_failure_message": (
                f"OpenSky: {error}"
            ),
        })

        _save_route_state(key, route_state,)

        result["open_sky"] = "failed"
        result["error"] = error
        result["duration_seconds"] = (
            _utc_now() - route_started
        ).total_seconds()

        return result

    # --------------------------------------------------------------
    # Weather
    # --------------------------------------------------------------

    logger.info(
        f"[{key}] [Weather] Starting fetch."
    )

    try:

        create_route_dataset(origin, destination, start_date=start_date, end_date=end_date,)

        new_last_weather_date = (
            get_last_weather_date(
                origin,
                destination,
            )
        )

        _stage_state(route_state, "weather", "success",)

        result["weather"] = "success"
        result["weather_stored_through"] = (
            new_last_weather_date.isoformat()
            if new_last_weather_date
            else None
        )

        logger.info(
            f"[{key}] [Weather] Completed successfully. "
            f"Stored through {new_last_weather_date}."
        )

    except Exception as e:

        error = str(e)

        _stage_state(route_state, "weather", "failed", error,)

        logger.exception(
            f"[{key}] [Weather] FAILED: {error}"
        )

        route_state.update({
            "last_failure_utc": (
                _utc_now().isoformat()
            ),
            "last_failure_message": (
                f"Weather: {error}"
            ),
        })

        _save_route_state(key, route_state,)

        result["weather"] = "failed"
        result["error"] = error
        result["duration_seconds"] = (
            _utc_now() - route_started
        ).total_seconds()

        return result

    # --------------------------------------------------------------
    # Both stages succeeded.
    #
    # The datasets do NOT need to have identical final dates.
    # --------------------------------------------------------------

    final_flight_date = (
        get_last_flight_date(
            origin,
            destination,
        )
    )

    final_weather_date = (
        get_last_weather_date(
            origin,
            destination,
        )
    )

    final_dates = [
        d
        for d in (
            final_flight_date,
            final_weather_date,
        )
        if d is not None
    ]

    if not final_dates:
        error = (
            "Both fetchers completed but no stored "
            "data was found."
        )

        logger.error(
            f"[{key}] {error}"
        )

        route_state.update({
            "last_failure_utc": (
                _utc_now().isoformat()
            ),
            "last_failure_message": error,
        })

        _save_route_state(key, route_state,)

        result["error"] = error
        result["duration_seconds"] = (
            _utc_now() - route_started
        ).total_seconds()

        return result

    fetched_through = min(final_dates)

    route_state.update({
        "last_success_utc": (
            _utc_now().isoformat()
        ),
        "last_success_date_fetched_through": (
            fetched_through.isoformat()
        ),
        "last_failure_utc": None,
        "last_failure_message": None,
    })

    _save_route_state(key, route_state,)

    logger.info(
        f"[{key}] Update succeeded. "
        f"OpenSky={final_flight_date}, "
        f"Weather={final_weather_date}."
    )

    result["status"] = "success"

    result["fetched_through"] = {
        "opensky": (
            final_flight_date.isoformat()
            if final_flight_date
            else None
        ),
        "weather": (
            final_weather_date.isoformat()
            if final_weather_date
            else None
        ),
    }

    result["duration_seconds"] = (
        _utc_now() - route_started
    ).total_seconds()

    return result


def run_daily_update(routes=None, force: bool = False,) -> list:

    """Run the update for every configured route."""

    routes = routes or config.ROUTES

    run_started = _utc_now()

    logger.info(
        f"Starting daily update for {len(routes)} route(s). "
        f"force={force}"
    )

    results = []

    for origin, dest in routes:
        try:
            results.append(
                update_route(
                    origin,
                    dest,
                    force=force,
                )
            )

        except Exception as e:
            key = config.route_key(
                origin,
                dest,
            )

            logger.exception(
                f"[{key}] UNEXPECTED ROUTE-LEVEL FAILURE: {e}"
            )

            results.append({
                "route": key,
                "status": "failed",
                "error": str(e),
            })

    n_ok = sum(
        1
        for r in results
        if r["status"] in (
            "success",
            "already_up_to_date",
        )
    )

    n_failed = sum(
        1
        for r in results
        if r["status"] == "failed"
    )

    n_skipped = sum(
        1
        for r in results
        if r["status"] == "skipped_backoff"
    )

    run_finished = _utc_now()

    logger.info(
        f"[SUMMARY] {n_ok} ok, "
        f"{n_failed} failed, "
        f"{n_skipped} skipped (backoff) "
        f"out of {len(results)} route(s)."
    )

    # --------------------------------------------------------------
    # Persistent per-run log
    # --------------------------------------------------------------

    run_record = {
        "started_utc": run_started.isoformat(),
        "finished_utc": run_finished.isoformat(),
        "duration_seconds": (
            run_finished - run_started
        ).total_seconds(),
        "force": force,
        "route_count": len(routes),
        "summary": {
            "ok": n_ok,
            "failed": n_failed,
            "skipped_backoff": n_skipped,
        },
        "routes": results,
    }

    try:
        _write_run_log(run_record)

        logger.info(
            f"Persistent run log written to "
            f"{UPDATE_LOG_DIR}"
        )

    except Exception:
        # Logging failure should not hide the actual update result.
        logger.exception(
            "Failed to write persistent run log."
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--route",
        nargs=2,
        metavar=("ORIGIN", "DEST"),
        help=(
            "Only update this one route instead of every route "
            "in config.ROUTES"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ignore the failure-backoff window and "
            "retry immediately"
        ),
    )

    args = parser.parse_args()

    routes = (
        [tuple(a.upper() for a in args.route)]
        if args.route
        else None
    )

    results = run_daily_update(
        routes=routes,
        force=args.force,
    )

    # All routes are attempted before this point.
    # GitHub Actions receives a failure status if any route failed.
    if any(
        r["status"] == "failed"
        for r in results
    ):
        sys.exit(1)

    sys.exit(0)