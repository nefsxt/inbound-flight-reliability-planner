"""
Tests for src.update_data.

The external OpenSky and Open-Meteo fetchers are mocked so these tests focus
on orchestration, checkpoint state, retry/backoff decisions, and persistent
run logging.
"""

import datetime as dt
import json

import pandas as pd
import pytest

import src.update_data as update


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config(monkeypatch, tmp_path):

    """Provide isolated checkpoint/log paths and deterministic configuration."""

    monkeypatch.setattr(update.config, "PROCESSED_DIR", str(tmp_path / "processed"),)
    monkeypatch.setattr(update.config, "UPDATE_RETRY_BACKOFF_DAYS", 2,)

    monkeypatch.setattr(
        update.config,
        "ROUTES",
        [
            ("EDDF", "LGTS"),
            ("LGTS", "EDDF"),
        ],
    )

    monkeypatch.setattr(update.config, "route_key", lambda origin, destination: f"{origin}->{destination}",)
    monkeypatch.setattr(update, "CHECKPOINT_PATH", str(tmp_path / "processed" / "update_checkpoint.json"),)
    monkeypatch.setattr(update, "UPDATE_LOG_DIR", str(tmp_path / "processed" / "update_logs"),)

    return tmp_path


@pytest.fixture
def fixed_now(monkeypatch):

    """Provide a deterministic current UTC timestamp."""

    now = dt.datetime(
        2026,
        8,
        21,
        12,
        0,
        tzinfo=dt.timezone.utc,
    )

    monkeypatch.setattr(update, "_utc_now", lambda: now,)

    return now


# ---------------------------------------------------------------------------
# _utc_now
# ---------------------------------------------------------------------------


def test_utc_now_returns_timezone_aware_utc_datetime(monkeypatch):

    """Verify that _utc_now returns a timezone-aware UTC datetime."""

    expected = dt.datetime(
        2026,
        8,
        21,
        12,
        30,
        tzinfo=dt.timezone.utc,
    )

    class FakeDateTime:
        @classmethod
        def now(cls, timezone):
            return expected

    monkeypatch.setattr(update.dt, "datetime", FakeDateTime)

    result = update._utc_now()

    assert result == expected
    assert result.tzinfo == dt.timezone.utc


# ---------------------------------------------------------------------------
# _write_run_log
# ---------------------------------------------------------------------------


def test_write_run_log_creates_timestamped_json_file(mock_config, fixed_now,):

    """Verify that a run record is written as a timestamped JSON file."""

    run_record = {
        "started_utc": fixed_now.isoformat(),
        "finished_utc": fixed_now.isoformat(),
        "route_count": 1,
        "summary": {
            "ok": 1,
            "failed": 0,
            "skipped_backoff": 0,
        },
        "routes": [],
    }

    update._write_run_log(run_record)

    log_dir = mock_config / "processed" / "update_logs"
    log_path = log_dir / "2026-08-21_12-00-00.json"

    assert log_path.exists()

    with open(log_path, encoding="utf-8") as f:
        saved = json.load(f)

    assert saved == run_record


def test_write_run_log_does_not_leave_temporary_file(mock_config, fixed_now,):

    """Verify that atomic run-log writing leaves no temporary file behind."""

    run_record = {
        "started_utc": fixed_now.isoformat(),
    }

    update._write_run_log(run_record)

    files = list(
        (mock_config / "processed" / "update_logs").iterdir()
    )

    assert len(files) == 1
    assert files[0].suffix == ".json"


# ---------------------------------------------------------------------------
# _load_checkpoint
# ---------------------------------------------------------------------------


def test_load_checkpoint_returns_empty_dict_when_missing(mock_config):

    """Verify that a missing checkpoint is treated as empty state."""

    result = update._load_checkpoint()

    assert result == {}


def test_load_checkpoint_reads_valid_json(mock_config):

    """Verify that valid checkpoint JSON is loaded correctly."""

    checkpoint = {
        "EDDF->LGTS": {
            "last_success_utc": "2026-08-20T12:00:00+00:00",
        }
    }

    path = mock_config / "processed" / "update_checkpoint.json"
    path.parent.mkdir(parents=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f)

    result = update._load_checkpoint()

    assert result == checkpoint


@pytest.mark.parametrize(
    "contents",
    [
        "{invalid json",
        "",
    ],
)
def test_load_checkpoint_returns_empty_dict_for_invalid_json(mock_config, contents,):

    """Verify that malformed checkpoint JSON does not crash the updater."""

    path = mock_config / "processed" / "update_checkpoint.json"
    path.parent.mkdir(parents=True)

    path.write_text(contents, encoding="utf-8")

    result = update._load_checkpoint()

    assert result == {}


def test_load_checkpoint_returns_empty_dict_for_os_error(monkeypatch, mock_config,):

    """Verify that checkpoint read errors are handled gracefully."""

    monkeypatch.setattr(update.os.path, "exists", lambda path: True,)

    def failing_open(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr("builtins.open", failing_open)

    result = update._load_checkpoint()

    assert result == {}


# ---------------------------------------------------------------------------
# _save_checkpoint
# ---------------------------------------------------------------------------


def test_save_checkpoint_writes_json_atomically(mock_config,):

    """Verify that checkpoint state is persisted as JSON."""

    state = {
        "EDDF->LGTS": {
            "last_success_utc": "2026-08-21T12:00:00+00:00",
            "last_failure_utc": None,
        }
    }

    update._save_checkpoint(state)

    path = mock_config / "processed" / "update_checkpoint.json"

    assert path.exists()

    with open(path, encoding="utf-8") as f:
        saved = json.load(f)

    assert saved == state


def test_save_checkpoint_creates_parent_directory(mock_config):

    """Verify that the checkpoint directory is created automatically."""

    state = {"EDDF->LGTS": {"status": "success"}}

    update._save_checkpoint(state)

    assert (mock_config / "processed").is_dir()


def test_save_checkpoint_does_not_leave_temporary_file(mock_config,):

    """Verify that atomic checkpoint writing leaves no temporary file."""

    update._save_checkpoint({"route": "state"})

    files = list(
        (mock_config / "processed").iterdir()
    )

    assert files == [
        mock_config / "processed" / "update_checkpoint.json"
    ]


# ---------------------------------------------------------------------------
# _in_backoff_window
# ---------------------------------------------------------------------------


def test_in_backoff_window_returns_false_without_failure(mock_config, fixed_now,):

    """Verify that routes without a recorded failure are not in backoff."""

    assert update._in_backoff_window({}) is False


def test_in_backoff_window_returns_true_for_recent_failure(mock_config, fixed_now,):

    """Verify that a recent failure triggers the configured cooldown."""

    route_state = {
        "last_failure_utc": (
            fixed_now - dt.timedelta(days=1)
        ).isoformat()
    }

    assert update._in_backoff_window(route_state) is True


def test_in_backoff_window_returns_false_after_backoff(mock_config, fixed_now,):

    """Verify that the route becomes eligible after the cooldown expires."""

    route_state = {
        "last_failure_utc": (
            fixed_now - dt.timedelta(days=3)
        ).isoformat()
    }

    assert update._in_backoff_window(route_state) is False


def test_in_backoff_window_handles_naive_failure_timestamp(mock_config, fixed_now,):

    """Verify that naive timestamps are interpreted as UTC."""

    route_state = {
        "last_failure_utc": "2026-08-20T12:00:00"
    }

    assert update._in_backoff_window(route_state) is True


# ---------------------------------------------------------------------------
# _save_route_state
# ---------------------------------------------------------------------------


def test_save_route_state_updates_only_requested_route(monkeypatch, mock_config,):

    """Verify that one route can be updated without replacing other routes."""

    existing = {
        "EDDF->LGTS": {
            "status": "old",
        },
        "LGTS->EDDF": {
            "status": "keep",
        },
    }

    monkeypatch.setattr(update, "_load_checkpoint", lambda: existing.copy(),)

    saved = {}

    monkeypatch.setattr(update, "_save_checkpoint", lambda state: saved.update(state),)

    route_state = {
        "status": "new",
    }

    update._save_route_state(
        "EDDF->LGTS",
        route_state,
    )

    assert saved["EDDF->LGTS"] == route_state
    assert saved["LGTS->EDDF"] == {
        "status": "keep"
    }


# ---------------------------------------------------------------------------
# _stage_state
# ---------------------------------------------------------------------------


def test_stage_state_records_success():

    """Verify that a successful stage gets timestamped state."""

    route_state = {}

    fixed = dt.datetime(
        2026,
        8,
        21,
        12,
        0,
        tzinfo=dt.timezone.utc,
    )

    update._stage_state(
        route_state,
        "opensky",
        "success",
    )

    assert route_state["stages"]["opensky"]["status"] == "success"
    assert route_state["stages"]["opensky"]["error"] is None
    assert "updated_utc" in route_state["stages"]["opensky"]


def test_stage_state_records_failure_and_error():

    """Verify that a failed stage records its error."""

    route_state = {}

    update._stage_state(
        route_state,
        "weather",
        "failed",
        "API unavailable",
    )

    stage = route_state["stages"]["weather"]

    assert stage["status"] == "failed"
    assert stage["error"] == "API unavailable"
    assert "updated_utc" in stage


def test_stage_state_preserves_other_stages():

    """Verify that updating one stage does not overwrite another."""

    route_state = {
        "stages": {
            "opensky": {
                "status": "success",
                "error": None,
            }
        }
    }

    update._stage_state(
        route_state,
        "weather",
        "failed",
        "timeout",
    )

    assert route_state["stages"]["opensky"]["status"] == "success"
    assert route_state["stages"]["weather"]["status"] == "failed"


# ---------------------------------------------------------------------------
# update_route - already up to date
# ---------------------------------------------------------------------------


def test_update_route_skips_route_inside_backoff(monkeypatch, mock_config, fixed_now,):

    """Verify that failed routes are skipped during the backoff window."""

    checkpoint = {
        "EDDF->LGTS": {
            "last_failure_utc": (
                fixed_now - dt.timedelta(days=1)
            ).isoformat()
        }
    }

    monkeypatch.setattr(update, "_load_checkpoint", lambda: checkpoint,)

    monkeypatch.setattr(
        update,
        "get_last_flight_date",
        lambda *args: pytest.fail(
            "Stored dates should not be queried during backoff"
        ),
    )

    result = update.update_route(
        "EDDF",
        "LGTS",
    )

    assert result["route"] == "EDDF->LGTS"
    assert result["status"] == "skipped_backoff"
    assert "retry_at_utc" in result


def test_update_route_force_bypasses_backoff(monkeypatch, mock_config, fixed_now,):

    """Verify that force=True bypasses the failure cooldown."""

    checkpoint = {
        "EDDF->LGTS": {
            "last_failure_utc": (
                fixed_now - dt.timedelta(days=1)
            ).isoformat()
        }
    }

    monkeypatch.setattr(update, "_load_checkpoint", lambda: checkpoint,)
    monkeypatch.setattr(update, "get_last_flight_date", lambda *args: fixed_now.date(),)
    monkeypatch.setattr(update, "get_last_weather_date", lambda *args: fixed_now.date(),)

    result = update.update_route(
        "EDDF",
        "LGTS",
        force=True,
    )

    assert result["status"] == "already_up_to_date"


def test_update_route_returns_already_up_to_date(monkeypatch, mock_config, fixed_now,):

    """Verify that no fetch occurs when both datasets are current."""

    save_calls = []

    monkeypatch.setattr(update, "_load_checkpoint", lambda: {},)
    monkeypatch.setattr(update, "get_last_flight_date", lambda *args: fixed_now.date(),)
    monkeypatch.setattr(update, "get_last_weather_date", lambda *args: fixed_now.date(),)

    monkeypatch.setattr(
        update,
        "_save_route_state",
        lambda key, state: save_calls.append(
            (key, state.copy())
        ),
    )

    monkeypatch.setattr(
        update,
        "build_route_dataset",
        lambda *args, **kwargs: pytest.fail(
            "OpenSky should not be fetched"
        ),
    )

    monkeypatch.setattr(
        update,
        "create_route_dataset",
        lambda *args, **kwargs: pytest.fail(
            "Weather should not be fetched"
        ),
    )

    result = update.update_route("EDDF", "LGTS",)

    assert result["status"] == "already_up_to_date"
    assert len(save_calls) == 1
    assert save_calls[0][0] == "EDDF->LGTS"
    assert save_calls[0][1]["last_failure_utc"] is None


# ---------------------------------------------------------------------------
# update_route - first-time backfill
# ---------------------------------------------------------------------------

def test_update_route_first_time_backfill_passes_none_start_date(monkeypatch, mock_config, fixed_now):

    """
    Verify that a route with no stored data performs a first-time backfill
    using start_date=None.
    """
    calls = []

    
    monkeypatch.setattr(update, "_load_checkpoint", lambda: {})
    monkeypatch.setattr(update, "_save_route_state", lambda *args: None)

    
    def fake_opensky(*args, **kwargs):
        calls.append(("opensky", args, kwargs))

    def fake_weather(*args, **kwargs):
        calls.append(("weather", args, kwargs))

    monkeypatch.setattr(update, "build_route_dataset", fake_opensky)
    monkeypatch.setattr(update, "create_route_dataset", fake_weather)

    # Flight tracking iterator 
    # 1st call: Returns None to trigger first-time backfill
    # 2nd call: Returns date to verify OpenSky stage success
    # 3rd call: Returns date for final result evaluation
    stored_flight_dates = iter([
        None, 
        fixed_now.date(), 
        fixed_now.date()
    ])
    
    # Weather tracking iterator (
    # 1st call: Returns None to assist the initial backfill logic
    # 2nd call: Returns date to verify Weather stage success
    # 3rd call : Returns date for final result evaluation
    stored_weather_dates = iter([
        None, 
        fixed_now.date(), 
        fixed_now.date()
    ])

    monkeypatch.setattr(update, "get_last_flight_date", lambda *args: next(stored_flight_dates))
    monkeypatch.setattr(update, "get_last_weather_date", lambda *args: next(stored_weather_dates))

    
    result = update.update_route("EDDF", "LGTS")

    # Assert tracking details in the returned response dictionary
    assert result["open_sky"] == "success"
    assert result["weather"] == "success"

    # Verify OpenSky call parameters
    assert calls[0][0] == "opensky"
    assert calls[0][2]["start_date"] is None
    assert calls[0][2]["end_date"] == "2026-08-21"

    # Verify Weather call parameters
    assert calls[1][0] == "weather"
    assert calls[1][2]["start_date"] is None
    assert calls[1][2]["end_date"] == "2026-08-21"



# ---------------------------------------------------------------------------
# update_route - incremental range
# ---------------------------------------------------------------------------


def test_update_route_resumes_from_oldest_dataset(monkeypatch, mock_config, fixed_now,):

    """
    Verify that the missing range starts one day after the older of the two
    stored datasets.
    """
    calls = []

    monkeypatch.setattr(update, "_load_checkpoint", lambda: {},)
    monkeypatch.setattr(update, "get_last_flight_date", lambda *args: dt.date(2026, 8, 18),)
    monkeypatch.setattr(update, "get_last_weather_date", lambda *args: dt.date(2026, 8, 20),)

    monkeypatch.setattr(
        update,
        "build_route_dataset",
        lambda *args, **kwargs: calls.append(
            ("opensky", kwargs)
        ),
    )

    monkeypatch.setattr(
        update,
        "create_route_dataset",
        lambda *args, **kwargs: calls.append(
            ("weather", kwargs)
        ),
    )

    monkeypatch.setattr(
        update,
        "_save_route_state",
        lambda *args: None,
    )

    result = update.update_route("EDDF","LGTS",)

    assert result["requested_start"] == "2026-08-19"
    assert result["requested_end"] == "2026-08-21"

    assert calls[0][1]["start_date"] == "2026-08-19"
    assert calls[0][1]["end_date"] == "2026-08-21"

    assert calls[1][1]["start_date"] == "2026-08-19"
    assert calls[1][1]["end_date"] == "2026-08-21"


# ---------------------------------------------------------------------------
# update_route - OpenSky
# ---------------------------------------------------------------------------


def test_update_route_open_sky_failure_stops_before_weather(monkeypatch, mock_config, fixed_now,):

    """
    Verify that an OpenSky failure is recorded and the weather stage is not
    attempted.
    """
    saved = []

    monkeypatch.setattr(update, "_load_checkpoint", lambda: {},)
    monkeypatch.setattr(update, "get_last_flight_date", lambda *args: None,)
    monkeypatch.setattr(update, "get_last_weather_date", lambda *args: None,)

    monkeypatch.setattr(
        update,
        "build_route_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("OpenSky failed")
        ),
    )

    monkeypatch.setattr(
        update,
        "create_route_dataset",
        lambda *args, **kwargs: pytest.fail(
            "Weather should not run after OpenSky failure"
        ),
    )

    monkeypatch.setattr(
        update,
        "_save_route_state",
        lambda key, state: saved.append(
            (key, state.copy())
        ),
    )

    result = update.update_route("EDDF", "LGTS",)

    assert result["status"] == "failed"
    assert result["open_sky"] == "failed"
    assert result["weather"] is None
    assert result["error"] == "OpenSky failed"

    assert saved[0][1]["stages"]["opensky"]["status"] == "failed"
    assert (
        saved[0][1]["last_failure_message"]
        == "OpenSky: OpenSky failed"
    )


def test_update_route_open_sky_success_records_progress(monkeypatch, mock_config, fixed_now,):

    """Verify that successful OpenSky data is recorded before Weather runs."""

    saved_states = []

    dates = iter(
        [
            dt.date(2026, 8, 20),  # initial flight
            dt.date(2026, 8, 20),  # initial weather
            dt.date(2026, 8, 21),  # after OpenSky
            dt.date(2026, 8, 21),  # after Weather
            dt.date(2026, 8, 21),  # final flight
            dt.date(2026, 8, 21),  # final weather
        ]
    )

    monkeypatch.setattr(update, "_load_checkpoint", lambda: {},)
    monkeypatch.setattr(update, "get_last_flight_date", lambda *args: next(dates),)
    monkeypatch.setattr(update, "get_last_weather_date", lambda *args: next(dates),)
    monkeypatch.setattr(update, "build_route_dataset", lambda *args, **kwargs: None,)
    monkeypatch.setattr(update, "create_route_dataset", lambda *args, **kwargs: None,)

    monkeypatch.setattr(
        update,
        "_save_route_state",
        lambda key, state: saved_states.append(
            (key, state.copy())
        ),
    )

    result = update.update_route("EDDF", "LGTS",)

    assert result["open_sky"] == "success"
    assert result["weather"] == "success"
    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# update_route - Weather
# ---------------------------------------------------------------------------


def test_update_route_weather_failure_retains_opensky_success(monkeypatch, mock_config,):

    """
    Verify the key independent-stage behavior: OpenSky succeeds, Weather
    fails, and the successful OpenSky state is retained.
    """
    saved = []

    dates = iter(
        [
            dt.date(2026, 8, 20),  # initial flight
            dt.date(2026, 8, 20),  # initial weather
            dt.date(2026, 8, 21),  # after OpenSky
        ]
    )

    monkeypatch.setattr(update, "_load_checkpoint", lambda: {},)
    monkeypatch.setattr(update, "get_last_flight_date", lambda *args: next(dates),)
    monkeypatch.setattr(update, "get_last_weather_date", lambda *args: next(dates),)
    monkeypatch.setattr(update, "build_route_dataset", lambda *args, **kwargs: None,)

    monkeypatch.setattr(
        update,
        "create_route_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Weather failed")
        ),
    )

    monkeypatch.setattr(
        update,
        "_save_route_state",
        lambda key, state: saved.append(
            (key, state.copy())
        ),
    )

    result = update.update_route("EDDF", "LGTS",)

    assert result["status"] == "failed"
    assert result["open_sky"] == "success"
    assert result["weather"] == "failed"
    assert result["error"] == "Weather failed"

    assert saved

    final_state = saved[-1][1]

    assert (final_state["stages"]["opensky"]["status"] == "success")
    assert (final_state["stages"]["weather"]["status"] == "failed")
    assert (final_state["last_failure_message"] == "Weather: Weather failed")


def test_update_route_weather_failure_records_error(monkeypatch, mock_config,):

    """Verify that weather exceptions are recorded in route state."""

    saved = []

    monkeypatch.setattr(update, "_load_checkpoint", lambda: {},)
    monkeypatch.setattr(update, "get_last_flight_date", lambda *args: dt.date(2026, 8, 20),)
    monkeypatch.setattr(update, "get_last_weather_date", lambda *args: dt.date(2026, 8, 20),)
    monkeypatch.setattr(update, "build_route_dataset", lambda *args, **kwargs: None,)

    monkeypatch.setattr(
        update,
        "create_route_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("API unavailable")
        ),
    )

    monkeypatch.setattr(
        update,
        "_save_route_state",
        lambda key, state: saved.append(state.copy()),
    )

    result = update.update_route("EDDF", "LGTS",)

    assert result["weather"] == "failed"
    assert result["error"] == "API unavailable"

    assert saved[-1]["stages"]["weather"]["status"] == "failed"
    assert (saved[-1]["stages"]["weather"]["error"] == "API unavailable")


# ---------------------------------------------------------------------------
# update_route - both successful
# ---------------------------------------------------------------------------


def test_update_route_succeeds_when_both_stages_succeed(monkeypatch, mock_config, fixed_now,):

    """Verify the complete successful route update."""

    saved = []

    dates = iter(
        [
            dt.date(2026, 8, 20),  # initial flight
            dt.date(2026, 8, 19),  # initial weather
            dt.date(2026, 8, 21),  # after OpenSky
            dt.date(2026, 8, 21),  # after Weather
            dt.date(2026, 8, 21),  # final flight
            dt.date(2026, 8, 21),  # final weather
        ]
    )

    monkeypatch.setattr(update, "_load_checkpoint", lambda: {},)
    monkeypatch.setattr(update, "get_last_flight_date", lambda *args: next(dates),)
    monkeypatch.setattr(update, "get_last_weather_date", lambda *args: next(dates),)
    monkeypatch.setattr(update, "build_route_dataset", lambda *args, **kwargs: None,)
    monkeypatch.setattr(update, "create_route_dataset", lambda *args, **kwargs: None,)

    monkeypatch.setattr(
        update,
        "_save_route_state",
        lambda key, state: saved.append(
            (key, state.copy())
        ),
    )

    result = update.update_route("EDDF", "LGTS",)

    assert result["status"] == "success"
    assert result["open_sky"] == "success"
    assert result["weather"] == "success"

    assert result["fetched_through"] == {
        "opensky": "2026-08-21",
        "weather": "2026-08-21",
    }

    final_state = saved[-1][1]

    assert final_state["last_success_date_fetched_through"] == ("2026-08-21")
    assert final_state["last_failure_utc"] is None
    assert final_state["last_failure_message"] is None


def test_update_route_success_uses_oldest_final_date_as_fetched_through(monkeypatch, mock_config,):

    """
    Verify that the route-level success date is the minimum of the two final
    datasets, preventing the route from claiming progress beyond its lagging
    source.
    """
    dates = iter(
        [
            dt.date(2026, 8, 20),
            dt.date(2026, 8, 20),
            dt.date(2026, 8, 21),
            dt.date(2026, 8, 21),
            dt.date(2026, 8, 21),
            dt.date(2026, 8, 20),
        ]
    )

    saved = []

    monkeypatch.setattr(update, "_load_checkpoint", lambda: {},)
    monkeypatch.setattr(update, "get_last_flight_date", lambda *args: next(dates),)
    monkeypatch.setattr(update, "get_last_weather_date", lambda *args: next(dates),)
    monkeypatch.setattr(update, "build_route_dataset", lambda *args, **kwargs: None,)
    monkeypatch.setattr(update, "create_route_dataset", lambda *args, **kwargs: None,)
    monkeypatch.setattr(update, "_save_route_state", lambda key, state: saved.append(state.copy()),)

    result = update.update_route("EDDF", "LGTS",)

    assert result["status"] == "success"

    assert result["fetched_through"] == {
        "opensky": "2026-08-21",
        "weather": "2026-08-20",
    }

    assert (saved[-1]["last_success_date_fetched_through"] == "2026-08-20")


def test_update_route_fails_when_both_fetchers_complete_without_stored_data(monkeypatch, mock_config,):

    """
    Verify that successful function completion is not considered a successful
    update if neither dataset actually contains stored data.
    """
    saved = []

    monkeypatch.setattr(update, "_load_checkpoint", lambda: {},)
    monkeypatch.setattr(update, "get_last_flight_date", lambda *args: None,)
    monkeypatch.setattr(update, "get_last_weather_date", lambda *args: None,)
    monkeypatch.setattr(update, "build_route_dataset", lambda *args, **kwargs: None,)
    monkeypatch.setattr(update, "create_route_dataset", lambda *args, **kwargs: None,)
    monkeypatch.setattr(update, "_save_route_state", lambda key, state: saved.append(state.copy()),)

    result = update.update_route("EDDF", "LGTS",)

    assert result["status"] == "failed"
    assert (result["error"] == "Both fetchers completed but no stored data was found.")
    assert (saved[-1]["last_failure_message"] == "Both fetchers completed but no stored data was found.")


# ---------------------------------------------------------------------------
# run_daily_update
# ---------------------------------------------------------------------------


def test_run_daily_update_processes_all_routes(monkeypatch, mock_config,):

    """Verify that every configured route is attempted."""

    calls = []

    def fake_update(origin, destination, force=False):

        calls.append(
            (origin, destination, force)
        )

        return {
            "route": f"{origin}->{destination}",
            "status": "success",
        }

    monkeypatch.setattr(update, "update_route", fake_update,)
    monkeypatch.setattr(update, "_write_run_log", lambda record: None,)

    results = update.run_daily_update(force=True,)

    assert calls == [
        ("EDDF", "LGTS", True),
        ("LGTS", "EDDF", True),
    ]

    assert len(results) == 2
    assert all(
        result["status"] == "success"
        for result in results
    )


def test_run_daily_update_continues_after_route_failure(monkeypatch,mock_config,):

    """
    Verify that an unexpected route-level exception does not prevent later
    routes from running.
    """
    calls = []

    def fake_update(origin, destination, force=False):

        calls.append(
            (origin, destination)
        )

        if origin == "EDDF":
            raise RuntimeError("unexpected route failure")

        return {
            "route": f"{origin}->{destination}",
            "status": "success",
        }

    monkeypatch.setattr(update, "update_route", fake_update,)
    monkeypatch.setattr(update, "_write_run_log", lambda record: None,)

    results = update.run_daily_update(
        routes=[
            ("EDDF", "LGTS"),
            ("LGTS", "EDDF"),
        ]
    )

    assert calls == [
        ("EDDF", "LGTS"),
        ("LGTS", "EDDF"),
    ]

    assert results[0]["status"] == "failed"
    assert results[0]["error"] == "unexpected route failure"

    assert results[1]["status"] == "success"


def test_run_daily_update_counts_success_failure_and_backoff(monkeypatch, mock_config,):

    """Verify the summary counters for all three route outcomes."""

    responses = iter(
        [
            {
                "route": "EDDF->LGTS",
                "status": "success",
            },
            {
                "route": "LGTS->EDDF",
                "status": "failed",
            },
            {
                "route": "AAA->BBB",
                "status": "skipped_backoff",
            },
        ]
    )

    monkeypatch.setattr(update, "update_route", lambda *args, **kwargs: next(responses),)

    records = []

    monkeypatch.setattr(update, "_write_run_log", lambda record: records.append(record),)

    routes = [
        ("EDDF", "LGTS"),
        ("LGTS", "EDDF"),
        ("AAA", "BBB"),
    ]

    results = update.run_daily_update(routes=routes,)

    assert len(results) == 3

    assert records[0]["route_count"] == 3

    assert records[0]["summary"] == {
        "ok": 1,
        "failed": 1,
        "skipped_backoff": 1,
    }


def test_run_daily_update_writes_persistent_run_log(monkeypatch, mock_config, fixed_now,):

    """Verify that the complete run record is passed to the log writer."""

    records = []

    monkeypatch.setattr(
        update,
        "update_route",
        lambda *args, **kwargs: {
            "route": "EDDF->LGTS",
            "status": "success",
        },
    )

    monkeypatch.setattr(update, "_write_run_log", lambda record: records.append(record),)

    results = update.run_daily_update(
        routes=[
            ("EDDF", "LGTS"),
        ],
        force=True,
    )

    assert len(records) == 1

    record = records[0]

    assert record["force"] is True
    assert record["route_count"] == 1
    assert record["routes"] == results

    assert record["summary"] == {
        "ok": 1,
        "failed": 0,
        "skipped_backoff": 0,
    }

    assert "started_utc" in record
    assert "finished_utc" in record
    assert "duration_seconds" in record


def test_run_daily_update_does_not_fail_if_run_log_writing_fails(monkeypatch, mock_config,):

    """
    Verify that failure to persist the run log does not hide the actual update
    results.
    """
    monkeypatch.setattr(
        update,
        "update_route",
        lambda *args, **kwargs: {
            "route": "EDDF->LGTS",
            "status": "success",
        },
    )

    monkeypatch.setattr(
        update,
        "_write_run_log",
        lambda record: (_ for _ in ()).throw(
            OSError("disk full")
        ),
    )

    results = update.run_daily_update(
        routes=[
            ("EDDF", "LGTS"),
        ]
    )

    assert len(results) == 1
    assert results[0]["status"] == "success"


def test_run_daily_update_accepts_explicit_routes(monkeypatch, mock_config,):

    """Verify that explicitly supplied routes override config.ROUTES."""

    calls = []

    monkeypatch.setattr(
        update,
        "update_route",
        lambda origin, destination, force=False: (
            calls.append((origin, destination, force))
            or {
                "route": f"{origin}->{destination}",
                "status": "success",
            }
        ),
    )

    monkeypatch.setattr(update, "_write_run_log", lambda record: None,)

    routes = [
        ("ABCD", "EFGH"),
    ]

    update.run_daily_update(routes=routes, force=False,)

    assert calls == [
        ("ABCD", "EFGH", False),
    ]


def test_run_daily_update_treats_already_up_to_date_as_success(monkeypatch, mock_config,):

    """Verify that already-current routes count toward the successful total."""

    records = []

    monkeypatch.setattr(
        update,
        "update_route",
        lambda *args, **kwargs: {
            "route": "EDDF->LGTS",
            "status": "already_up_to_date",
        },
    )

    monkeypatch.setattr(update, "_write_run_log", lambda record: records.append(record),)

    results = update.run_daily_update(
        routes=[
            ("EDDF", "LGTS"),
        ]
    )

    assert results[0]["status"] == "already_up_to_date"
    assert records[0]["summary"]["ok"] == 1
    assert records[0]["summary"]["failed"] == 0

