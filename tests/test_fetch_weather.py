import os
import time

import pandas as pd
import pytest
import requests

import src.fetch_weather as weather


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config(monkeypatch, tmp_path):

    monkeypatch.setattr(weather.config, "PROCESSED_DIR", str(tmp_path / "processed"),)
    monkeypatch.setattr(weather.config, "RAW_DIR", str(tmp_path / "raw"),)

    monkeypatch.setattr(weather.config, "HIST_START_DATE", "2026-01-01",)
    monkeypatch.setattr(weather.config, "HIST_END_DATE", "2026-01-31",)

    monkeypatch.setattr(
        weather.config,
        "ORIGIN_AIRPORTS",
        {
            "EDDF": {
                "name": "Frankfurt",
                "lat": 50.026706,
                "lon": 8.558350,
            }
        },
    )

    monkeypatch.setattr(
        weather.config,
        "DEST_AIRPORTS",
        {
            "LGTS": {
                "name": "Thessaloniki",
                "lat": 40.519280,
                "lon": 22.970009,
            }
        },
    )

    monkeypatch.setattr(weather.config, "route_key", lambda origin, dest: f"{origin}->{dest}",)
    monkeypatch.setattr(weather.config, "route_weather_path", lambda origin, dest: str(tmp_path / f"{origin}_{dest}" / "weather.parquet"),)

    return tmp_path


# ---------------------------------------------------------------------------
# historical_chunk_path
# ---------------------------------------------------------------------------

def test_historical_chunk_path_builds_expected_path(monkeypatch, mock_config):
    """
    Verify that historical chunk paths use the shared weather cache directory,
    airport ICAO, and inclusive chunk start/end dates.
    """
    start = pd.Timestamp("2026-01-01")
    end = pd.Timestamp("2026-01-14")

    result = weather.historical_chunk_path("LGTS", start, end)

    expected = (
        mock_config
        / "raw"
        / "weather_hist_chunks"
        / "LGTS"
        / "LGTS_2026-01-01_2026-01-14.parquet"
    )

    assert result == str(expected)
    assert expected.parent.is_dir()


# ---------------------------------------------------------------------------
# _request_historical_chunk
# ---------------------------------------------------------------------------

def test_request_historical_chunk_returns_json(monkeypatch):
    """Verify that the raw API response is correctly captured as JSON."""
    calls = []

    class FakeResponse:
        status_code = 200
        ok = True

        def json(self):
            return {"hourly": {"time": ["2026-01-01T00:00"]}}

    def fake_get(url, params, timeout):
        calls.append((url, params, timeout))
        return FakeResponse()

    monkeypatch.setattr(weather.requests, "get", fake_get)

    result = weather._request_historical_chunk({"latitude": 40.5, "longitude": 23.0})

    assert result == {"hourly": {"time": ["2026-01-01T00:00"]}}
    assert len(calls) == 1



def test_request_historical_chunk_retries_429_using_retry_after(monkeypatch, mock_config,):
    """
    Verify that HTTP 429 responses are retried and that a valid Retry-After
    header controls the sleep duration.
    """
    responses = []

    class FakeResponse:
        def __init__(self, status_code, headers=None):
            self.status_code = status_code
            self.headers = headers or {}
            self.text = "Too Many Requests"
            self.ok = status_code < 400

        def json(self):
            return {"hourly": {"time": ["2026-01-01T00:00"]}}

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(response=self)

    responses.extend(
        [
            FakeResponse(429, {"Retry-After": "7"}),
            FakeResponse(200),
        ]
    )

    sleeps = []

    monkeypatch.setattr(weather.requests, "get", lambda *args, **kwargs: responses.pop(0),)
    monkeypatch.setattr(weather.time, "sleep", lambda seconds: sleeps.append(seconds),)

    result = weather._request_historical_chunk({})

    assert result["hourly"]["time"] == ["2026-01-01T00:00"]
    assert sleeps == [7.0]


def test_request_historical_chunk_uses_exponential_backoff_for_429(monkeypatch, mock_config):
    """
    Verify that a 429 without Retry-After uses the configured exponential
    backoff sequence.
    """
    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code
            self.headers = {}
            self.text = "Too Many Requests"
            # Dynamic ok flag: True if 2xx status code
            self.ok = (200 <= status_code < 300)

        def json(self):
            # Provide the structured data your assertion looks for
            if self.status_code == 200:
                return {"hourly": {"time": ["2026-01-01T00:00"]}}
            return {}

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(response=self)

    # Queue up two failures followed by a success block
    responses = [
        FakeResponse(429),
        FakeResponse(429),
        FakeResponse(200),
    ]

    sleeps = []

    # Mock out the network and sleep calls
    monkeypatch.setattr(weather.requests, "get", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr(weather.time, "sleep", lambda seconds: sleeps.append(seconds))

    # Run the function
    result = weather._request_historical_chunk({})

    # Assertions
    assert result["hourly"]["time"] == ["2026-01-01T00:00"]
    assert sleeps == [
        weather.INITIAL_RETRY_DELAY,
        weather.INITIAL_RETRY_DELAY * 2,
    ]

def test_request_historical_chunk_handles_invalid_retry_after(monkeypatch, mock_config):
    """
    Verify that a malformed Retry-After header falls back to the normal
    exponential delay rather than causing a parsing failure.
    """
    responses = []

    class FakeResponse:
        def __init__(self, status_code, headers=None):
            self.status_code = status_code
            self.headers = headers or {}
            self.text = "Too Many Requests"
            # Dynamically set ok to True for 200 codes, False for 429
            self.ok = (200 <= status_code < 300)

        def json(self):
            return {"hourly": {"time": ["2026-01-01T00:00"]}}

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(response=self)

    responses.extend(
        [
            FakeResponse(429, {"Retry-After": "not-a-number"}),
            FakeResponse(200),
        ]
    )

    sleeps = []

    monkeypatch.setattr(weather.requests, "get", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr(weather.time, "sleep", lambda seconds: sleeps.append(seconds))

    weather._request_historical_chunk({})

    assert sleeps == [weather.INITIAL_RETRY_DELAY]



@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_request_historical_chunk_retries_transient_server_errors(monkeypatch, mock_config, status_code,):
    """
    Verify that all documented transient server errors are retried before
    eventually succeeding.
    """
    responses = []

    class FakeResponse:
        def __init__(self, code):
            self.status_code = code
            self.headers = {}
            self.text = "Server Error"
            self.ok = code < 400

        def json(self):
            return {"hourly": {"time": ["2026-01-01T00:00"]}}

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(response=self)

    responses.extend(
        [
            FakeResponse(status_code),
            FakeResponse(200),
        ]
    )

    sleeps = []

    monkeypatch.setattr(weather.requests, "get", lambda *args, **kwargs: responses.pop(0),)
    monkeypatch.setattr(weather.time, "sleep", lambda seconds: sleeps.append(seconds),)

    result = weather._request_historical_chunk({})

    assert result["hourly"]["time"] == ["2026-01-01T00:00"]
    assert sleeps == [weather.INITIAL_RETRY_DELAY]


def test_request_historical_chunk_raises_immediately_on_non_retryable_http_error(monkeypatch, mock_config,):
    """
    Verify that HTTP errors outside the retryable set are immediately passed
    through to requests.raise_for_status().
    """
    calls = []

    class FakeResponse:
        status_code = 400
        ok = False
        text = "Bad Request"
        headers = {}

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(
                "400 Bad Request",
                response=self,
            )

    def fake_get(*args, **kwargs):
        calls.append(1)
        return FakeResponse()

    monkeypatch.setattr(weather.requests, "get", fake_get)

    with pytest.raises(requests.exceptions.HTTPError, match="400 Bad Request"):
        weather._request_historical_chunk({})

    assert len(calls) == 1


def test_request_historical_chunk_retries_timeout(monkeypatch, mock_config):
    """
    Verify that request timeouts are retried using exponential backoff and
    succeed when a later attempt completes.
    """
    calls = []
    sleeps = []

    class FakeResponse:
        status_code = 200
        ok = True

        def json(self):
            return {"hourly": {"time": ["2026-01-01T00:00"]}}

    def fake_get(*args, **kwargs):
        calls.append(1)

        if len(calls) == 1:
            raise requests.exceptions.Timeout("timed out")

        return FakeResponse()

    monkeypatch.setattr(weather.requests, "get", fake_get)
    monkeypatch.setattr(weather.time, "sleep", lambda seconds: sleeps.append(seconds),)

    result = weather._request_historical_chunk({})

    assert len(calls) == 2
    assert sleeps == [weather.INITIAL_RETRY_DELAY]
    assert "hourly" in result


def test_request_historical_chunk_retries_connection_error(monkeypatch, mock_config,):
    """
    Verify that connection failures are retried using exponential backoff.
    """
    calls = []
    sleeps = []

    class FakeResponse:
        status_code = 200
        ok = True

        def json(self):
            return {"hourly": {"time": ["2026-01-01T00:00"]}}

    def fake_get(*args, **kwargs):
        calls.append(1)

        if len(calls) == 1:
            raise requests.exceptions.ConnectionError("connection failed")

        return FakeResponse()

    monkeypatch.setattr(weather.requests, "get", fake_get)
    monkeypatch.setattr(weather.time, "sleep", lambda seconds: sleeps.append(seconds),)

    result = weather._request_historical_chunk({})

    assert len(calls) == 2
    assert sleeps == [weather.INITIAL_RETRY_DELAY]
    assert "hourly" in result


def test_request_historical_chunk_raises_after_persistent_429(monkeypatch, mock_config,):
    """
    Verify the fail-safe retry boundary: a persistent HTTP 429 is raised after
    MAX_RETRIES attempts rather than retrying indefinitely.
    """
    calls = []

    class FakeResponse:
        status_code = 429
        ok = False
        headers = {}
        text = "Still rate limited"

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(
                "429 Too Many Requests",
                response=self,
            )

    monkeypatch.setattr(weather.requests, "get", lambda *args, **kwargs: calls.append(1) or FakeResponse(),)
    monkeypatch.setattr(weather.time, "sleep", lambda seconds: None)

    with pytest.raises(requests.exceptions.HTTPError):
        weather._request_historical_chunk({})

    assert len(calls) == weather.MAX_RETRIES


def test_request_historical_chunk_raises_after_persistent_timeout(monkeypatch, mock_config,):
    """
    Verify that a persistent timeout eventually propagates after the configured
    maximum number of attempts.
    """
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(1)
        raise requests.exceptions.Timeout("persistent timeout")

    monkeypatch.setattr(weather.requests, "get", fake_get)
    monkeypatch.setattr(weather.time, "sleep", lambda seconds: None)

    with pytest.raises(requests.exceptions.Timeout):
        weather._request_historical_chunk({})

    assert len(calls) == weather.MAX_RETRIES


# ---------------------------------------------------------------------------
# fetch_historical_weather
# ---------------------------------------------------------------------------


def make_valid_cached_chunk(times, airport="LGTS"):

    """Helper to create a valid cached historical chunk DataFrame for testing."""

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(times, utc=True),
            "airport": [airport] * len(times),
        }
    )

    for i, column in enumerate(weather.HOURLY_VARS):
        df[column] = float(i + 1)

    return df



def test_fetch_historical_weather_fetches_and_caches_chunk(monkeypatch, mock_config,):
    """
    Verify the happy path: a historical API response is converted to a
    DataFrame, timestamp-normalized, assigned an airport, and cached.
    """
    saved = {}

    def fake_chunk_path(airport, start, end):
        return f"{airport}_{start:%Y-%m-%d}_{end:%Y-%m-%d}.parquet"

    monkeypatch.setattr(weather, "historical_chunk_path", fake_chunk_path)
    monkeypatch.setattr(weather, "load_or_none", lambda path: None)

    monkeypatch.setattr(
        weather,
        "_request_historical_chunk",
        lambda params: {
            "hourly": {
                "time": [
                    "2026-01-01T00:00",
                    "2026-01-01T01:00",
                ],
                "temperature_2m": [5.0, 6.0],
                "visibility": [10000, 11000],
            }
        },
    )

    def fake_save(df, path):
        saved["df"] = df.copy()
        saved["path"] = path

    monkeypatch.setattr(weather, "save_parquet", fake_save)

    result = weather.fetch_historical_weather("LGTS", 40.5, 23.0, "2026-01-01", "2026-01-01",)

    assert len(result) == 2
    assert result["airport"].tolist() == ["LGTS", "LGTS"]
    assert str(result["time"].dtype) == "datetime64[us, UTC]"
    assert len(saved["df"]) == 2


def test_fetch_historical_weather_builds_multiple_chunks(monkeypatch, mock_config,):
    """
    Verify that date ranges longer than HIST_CHUNK_DAYS are split into
    sequential chunks and every chunk is fetched and cached independently.
    """
    requests_seen = []
    saved = []

    monkeypatch.setattr(weather, "load_or_none", lambda path: None)

    def fake_chunk_path(airport, start, end):
        return f"{airport}_{start:%Y-%m-%d}_{end:%Y-%m-%d}.parquet"

    monkeypatch.setattr(weather, "historical_chunk_path", fake_chunk_path)

    def fake_request(params):
        requests_seen.append(params.copy())

        return {
            "hourly": {
                "time": [f"{params['start_date']}T00:00"],
                "temperature_2m": [10.0],
            }
        }

    monkeypatch.setattr(weather, "_request_historical_chunk", fake_request)
    monkeypatch.setattr(weather, "save_parquet", lambda df, path: saved.append((df.copy(), path)),)

    result = weather.fetch_historical_weather("LGTS", 40.5, 23.0, "2026-01-01", "2026-01-20",)

    assert len(requests_seen) == 2
    assert requests_seen[0]["start_date"] == "2026-01-01"
    assert requests_seen[0]["end_date"] == "2026-01-14"
    assert requests_seen[1]["start_date"] == "2026-01-15"
    assert requests_seen[1]["end_date"] == "2026-01-20"

    assert len(saved) == 2
    assert len(result) == 2


def test_fetch_historical_weather_uses_cached_chunks_without_api_call(monkeypatch, mock_config,):
    """
    Verify resumability: previously cached chunks are loaded directly and
    do not trigger another Historical Forecast API request.
    """
    cached = make_valid_cached_chunk(
        ["2026-01-01T00:00:00Z"],
        airport="LGTS",
    )

    monkeypatch.setattr(weather, "historical_chunk_path", lambda *args: "cached.parquet",)
    monkeypatch.setattr(weather, "load_or_none", lambda path: cached)

    def explosive_request(*args, **kwargs):
        pytest.fail(
            "Historical API was called even though the chunk was cached"
        )

    monkeypatch.setattr(weather, "_request_historical_chunk", explosive_request,)

    result = weather.fetch_historical_weather("LGTS", 40.5, 23.0, "2026-01-01", "2026-01-01",)

    assert len(result) == 1
    assert result.iloc[0]["airport"] == "LGTS"


def test_fetch_historical_weather_rejects_empty_api_response(monkeypatch, mock_config,):
    """
    Verify that an empty hourly response is treated as a failure rather than
    being cached as a valid zero-row weather dataset.
    """
    monkeypatch.setattr(weather, "historical_chunk_path", lambda *args: "empty.parquet",)
    monkeypatch.setattr(weather, "load_or_none", lambda path: None)
    monkeypatch.setattr(weather, "_request_historical_chunk", lambda params: {"hourly": {}},)

    monkeypatch.setattr(
        weather,
        "save_parquet",
        lambda *args: pytest.fail(
            "Invalid weather response must not be cached"
        ),
    )

    with pytest.raises(RuntimeError, match="no usable weather data"):
        weather.fetch_historical_weather("LGTS", 40.5, 23.0, "2026-01-01", "2026-01-01",)


def test_fetch_historical_weather_rejects_missing_time_column(monkeypatch, mock_config,):
    """
    Verify that a response containing weather variables but no time column is
    considered unusable and is not cached.
    """
    monkeypatch.setattr(weather, "historical_chunk_path", lambda *args: "missing_time.parquet",)
    monkeypatch.setattr(weather, "load_or_none", lambda path: None)
    monkeypatch.setattr(
        weather,
        "_request_historical_chunk",
        lambda params: {
            "hourly": {
                "temperature_2m": [10.0],
            }
        },
    )

    with pytest.raises(RuntimeError, match="no usable weather data"):
        weather.fetch_historical_weather("LGTS", 40.5, 23.0, "2026-01-01", "2026-01-01",)


def test_fetch_historical_weather_deduplicates_and_sorts_chunks(monkeypatch, mock_config,):
    """
    Verify that the final combined historical dataset is deduplicated on time
    and sorted chronologically.
    """
    chunks = [
        make_valid_cached_chunk(
            [
                "2026-01-02T01:00:00Z",
                "2026-01-02T00:00:00Z",
            ]
        ),
        make_valid_cached_chunk(
            [
                "2026-01-02T00:00:00Z",
                "2026-01-03T00:00:00Z",
            ]
        ),
    ]
    
    monkeypatch.setattr(weather, "historical_chunk_path", lambda *args: "chunk.parquet",)

    cached_index = {"value": 0}

    def fake_load(path):
        value = chunks[cached_index["value"]]
        cached_index["value"] += 1
        return value

    monkeypatch.setattr(weather, "load_or_none", fake_load)

    result = weather.fetch_historical_weather("LGTS", 40.5, 23.0, "2026-01-01", "2026-01-28",)

    assert len(result) == 3
    assert result["time"].tolist() == [
        pd.Timestamp("2026-01-02T00:00:00Z"),
        pd.Timestamp("2026-01-02T01:00:00Z"),
        pd.Timestamp("2026-01-03T00:00:00Z"),
    ]
    assert result["airport"].tolist() == ["LGTS", "LGTS", "LGTS"]


# ---------------------------------------------------------------------------
# fetch_live_weather
# ---------------------------------------------------------------------------

def test_fetch_live_weather_returns_hourly_dataframe(monkeypatch, mock_config):
    """
    Verify the happy path for live weather: the hourly response is converted
    into a DataFrame and timestamps are normalized to UTC.
    """
    calls = []

    class FakeResponse:
        status_code = 200
        ok = True
        text = ""

        def json(self):
            return {
                "hourly": {
                    "time": [
                        "2026-08-20T10:00",
                        "2026-08-20T11:00",
                    ],
                    "temperature_2m": [25.0, 26.0],
                }
            }

        def raise_for_status(self):
            pass

    def fake_get(url, params, timeout):
        calls.append((url, params, timeout))
        return FakeResponse()

    monkeypatch.setattr(weather.requests, "get", fake_get)

    result = weather.fetch_live_weather(40.5, 23.0, forecast_hours=24,)

    assert len(result) == 2
    assert str(result["time"].dtype) == "datetime64[us, UTC]"

    assert calls[0][0] == weather.LIVE_WEATHER_URL
    assert calls[0][1]["latitude"] == 40.5
    assert calls[0][1]["longitude"] == 23.0
    assert calls[0][1]["forecast_hours"] == 24
    assert calls[0][1]["timezone"] == "UTC"
    assert calls[0][2] == 30


def test_fetch_live_weather_handles_response_without_time(monkeypatch, mock_config,):
    """
    Verify that live weather data without a time column is still returned
    without attempting timestamp conversion.
    """
    class FakeResponse:
        status_code = 200
        ok = True
        text = ""

        def json(self):
            return {"hourly": {"temperature_2m": [20.0],}}

    monkeypatch.setattr(weather.requests, "get", lambda *args, **kwargs: FakeResponse(),)

    result = weather.fetch_live_weather(40.5, 23.0)

    assert len(result) == 1
    assert "time" not in result.columns


def test_fetch_live_weather_raises_on_http_error(monkeypatch, mock_config):
    """
    Verify that non-successful live API responses are logged and propagated
    through requests.raise_for_status().
    """
    class FakeResponse:
        status_code = 500
        ok = False
        text = "Internal Server Error"

        def json(self):
            return {}

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(
                "500 Internal Server Error"
            )

    monkeypatch.setattr(weather.requests, "get", lambda *args, **kwargs: FakeResponse(),)

    with pytest.raises(requests.exceptions.HTTPError, match="500 Internal Server Error",):
        weather.fetch_live_weather(40.5, 23.0)


# ---------------------------------------------------------------------------
# nearest_hour_weather
# ---------------------------------------------------------------------------

def test_nearest_hour_weather_returns_closest_row():
    """
    Verify that the weather observation closest to the requested timestamp is
    selected and returned as a dictionary.
    """
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-08-20T10:00:00Z",
                    "2026-08-20T11:00:00Z",
                    "2026-08-20T12:00:00Z",
                ],
                utc=True,
            ),
            "temperature_2m": [20.0, 22.0, 24.0],
        }
    )

    result = weather.nearest_hour_weather(df,pd.Timestamp("2026-08-20T10:40:00Z"),)

    assert result["temperature_2m"] == 22.0
    assert result["time"] == pd.Timestamp("2026-08-20T11:00:00Z")


def test_nearest_hour_weather_returns_empty_dict_for_empty_dataframe():
    """
    Verify that an empty weather dataset is handled gracefully without
    raising an index or timestamp error.
    """
    result = weather.nearest_hour_weather(pd.DataFrame(), pd.Timestamp("2026-08-20T10:00:00Z"),)

    assert result == {}


def test_nearest_hour_weather_returns_empty_dict_for_none():
    """
    Verify that None is treated the same as an unavailable weather dataset.
    """
    result = weather.nearest_hour_weather(None, pd.Timestamp("2026-08-20T10:00:00Z"),)

    assert result == {}


# ---------------------------------------------------------------------------
# _get_airport_info
# ---------------------------------------------------------------------------

def test_get_airport_info_finds_origin_airport(monkeypatch, mock_config):
    """
    Verify that airport information is found when the ICAO code is configured
    as an origin airport.
    """
    result = weather._get_airport_info("EDDF")

    assert result["name"] == "Frankfurt"
    assert result["lat"] == 50.026706
    assert result["lon"] == 8.558350

def test_get_airport_info_finds_destination_airport(monkeypatch, mock_config):
    """
    Verify that airport information is also found when the ICAO code exists
    only in the destination airport configuration.
    """
    result = weather._get_airport_info("LGTS")

    assert result["name"] == "Thessaloniki"
    assert result["lat"] == 40.519280
    assert result["lon"] == 22.970009


def test_get_airport_info_raises_for_unknown_airport(monkeypatch, mock_config,):
    """
    Verify that an unknown ICAO code raises a descriptive ValueError rather
    than silently returning incomplete configuration.
    """
    with pytest.raises(
        ValueError,
        match="Airport XXXX was not found",
    ):
        weather._get_airport_info("XXXX")


# ---------------------------------------------------------------------------
# _fetch_airport_worker
# ---------------------------------------------------------------------------

def test_fetch_airport_worker_returns_airport_and_dataframe(monkeypatch, mock_config,):
    """
    Verify that the worker passes airport coordinates and dates into the
    historical fetcher and returns the ICAO/DataFrame pair.
    """
    calls = []

    expected = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-01T00:00:00Z"],
                utc=True,
            ),
            "airport": ["LGTS"],
        }
    )

    def fake_fetch(icao, lat, lon, start, end):
        calls.append((icao, lat, lon, start, end))
        return expected

    monkeypatch.setattr(weather, "fetch_historical_weather", fake_fetch,)

    result_icao, result_df = weather._fetch_airport_worker(
        (
            "LGTS",
            {
                "name": "Thessaloniki",
                "lat": 40.5,
                "lon": 23.0,
            },
            "2026-01-01",
            "2026-01-14",
        )
    )

    assert result_icao == "LGTS"
    pd.testing.assert_frame_equal(result_df, expected)

    assert calls == [
        (
            "LGTS",
            40.5,
            23.0,
            "2026-01-01",
            "2026-01-14",
        )
    ]


def test_fetch_airport_worker_propagates_fetch_failure(monkeypatch, mock_config,):
    """
    Verify that a historical fetch failure is not swallowed by the worker.
    """
    monkeypatch.setattr(
        weather,
        "fetch_historical_weather",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("weather fetch failed")
        ),
    )

    with pytest.raises(RuntimeError, match="weather fetch failed"):
        weather._fetch_airport_worker(
            (
                "LGTS",
                {
                    "name": "Thessaloniki",
                    "lat": 40.5,
                    "lon": 23.0,
                },
                "2026-01-01",
                "2026-01-14",
            )
        )


# ---------------------------------------------------------------------------
# build_route_weather
# ---------------------------------------------------------------------------

def test_build_route_weather_fetches_both_airports_sequentially(monkeypatch, mock_config,):
    """
    Verify the route orchestrator fetches weather for both origin and
    destination airports and returns both datasets keyed by ICAO.
    """
    calls = []

    eddf = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-01T00:00:00Z"],
                utc=True,
            ),
            "airport": ["EDDF"],
        }
    )

    lgts = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-01T00:00:00Z"],
                utc=True,
            ),
            "airport": ["LGTS"],
        }
    )

    def fake_worker(item):
        icao, info, start, end = item
        calls.append((icao, start, end))

        return (
            icao,
            eddf if icao == "EDDF" else lgts,
        )

    monkeypatch.setattr(weather, "_fetch_airport_worker", fake_worker)

    result = weather.build_route_weather("EDDF", "LGTS", "2026-01-01", "2026-01-14",)

    assert set(result.keys()) == {"EDDF", "LGTS"}
    pd.testing.assert_frame_equal(result["EDDF"], eddf)
    pd.testing.assert_frame_equal(result["LGTS"], lgts)

    assert calls == [
        ("EDDF", "2026-01-01", "2026-01-14"),
        ("LGTS", "2026-01-01", "2026-01-14"),
    ]


def test_build_route_weather_propagates_airport_failure(monkeypatch, mock_config,):
    """
    Verify that a failure for either airport causes the route-level operation
    to fail rather than returning a partially successful result.
    """
    calls = []

    def fake_worker(item):
        icao = item[0]
        calls.append(icao)

        if icao == "LGTS":
            raise RuntimeError("destination weather failed")

        return (
            "EDDF",
            pd.DataFrame(
                {
                    "time": pd.to_datetime(
                        ["2026-01-01T00:00:00Z"],
                        utc=True,
                    ),
                    "airport": ["EDDF"],
                }
            ),
        )

    monkeypatch.setattr(weather, "_fetch_airport_worker", fake_worker)

    with pytest.raises(RuntimeError, match="destination weather failed",):
        weather.build_route_weather("EDDF", "LGTS", "2026-01-01", "2026-01-14",)

    assert calls == ["EDDF", "LGTS"]


# ---------------------------------------------------------------------------
# get_last_weather_date
# ---------------------------------------------------------------------------

def test_get_last_weather_date_returns_latest_date(monkeypatch, mock_config,):
    """
    Verify that the latest timestamp in the route weather dataset is converted
    into a calendar date.
    """
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-08-18T12:00:00Z",
                    "2026-08-20T15:30:00Z",
                    "2026-08-19T08:00:00Z",
                ],
                utc=True,
            )
        }
    )

    monkeypatch.setattr(weather, "load_or_none", lambda path: df)

    result = weather.get_last_weather_date("EDDF", "LGTS")

    assert result == pd.Timestamp("2026-08-20").date()


def test_get_last_weather_date_returns_none_when_dataset_missing(monkeypatch, mock_config,):
    """
    Verify that an absent weather dataset produces None rather than an
    exception.
    """
    monkeypatch.setattr(weather, "load_or_none", lambda path: None)

    result = weather.get_last_weather_date("EDDF", "LGTS")

    assert result is None


def test_get_last_weather_date_returns_none_for_empty_dataset(monkeypatch, mock_config,):
    """
    Verify that an empty weather DataFrame does not produce an invalid
    timestamp/date.
    """
    monkeypatch.setattr(weather, "load_or_none", lambda path: pd.DataFrame(),)

    result = weather.get_last_weather_date("EDDF", "LGTS")

    assert result is None


def test_get_last_weather_date_returns_none_without_time_column(monkeypatch, mock_config,):
    """
    Verify that malformed datasets without the required time column are
    treated as having no stored weather date.
    """
    monkeypatch.setattr(weather, "load_or_none", lambda path: pd.DataFrame({"temperature_2m": [20.0]}),)

    result = weather.get_last_weather_date("EDDF", "LGTS")

    assert result is None


# ---------------------------------------------------------------------------
# create_route_dataset
# ---------------------------------------------------------------------------

def test_create_route_dataset_creates_new_dataset(monkeypatch, mock_config):
    """Verify initial dataset creation when no previous route weather dataset

    exists.
    """
    eddf = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                ],
                utc=True,
            ),
            "airport": ["EDDF", "EDDF"],
            "temperature_2m": [5.0, 6.0],
        }
    )

    lgts = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                ],
                utc=True,
            ),
            "airport": ["LGTS", "LGTS"],
            "temperature_2m": [10.0, 11.0],
        }
    )

   
    monkeypatch.setattr(
        weather,
        "build_route_weather",
        lambda *args, **kwargs: {
            "EDDF": eddf,
            "LGTS": lgts,
        },
    )

    monkeypatch.setattr(weather.config, "route_weather_path", lambda origin, dest: f"data/processed/{origin}_{dest}/weather.parquet",)
    monkeypatch.setattr(weather.config, "route_key", lambda origin, dest: f"{origin}_{dest}",)
    monkeypatch.setattr(weather, "load_or_none", lambda path: None)
    monkeypatch.setattr(weather, "record_stage_stats", lambda *args, **kwargs: None)

    saved = {}

    def fake_save(df, path):
        saved["df"] = df.copy()
        saved["path"] = path

    monkeypatch.setattr(weather, "save_parquet", fake_save)

    result = weather.create_route_dataset("EDDF", "LGTS", "2026-01-01", "2026-01-01",)

    
    assert len(result) == 4
    assert set(result["airport"]) == {"EDDF", "LGTS"}
    assert set(result["route"]) == {"EDDF_LGTS"} 
    assert saved["path"].endswith("processed/EDDF_LGTS/weather.parquet")
    assert len(saved["df"]) == 4


def test_create_route_dataset_merges_existing_data(monkeypatch, mock_config,):
    """
    Verify incremental behavior: new weather rows are appended to an existing
    route dataset rather than replacing it.
    """
    existing = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-01T00:00:00Z"],
                utc=True,
            ),
            "airport": ["EDDF"],
            "temperature_2m": [5.0],
        }
    )

    new_eddf = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-01T01:00:00Z"],
                utc=True,
            ),
            "airport": ["EDDF"],
            "temperature_2m": [6.0],
        }
    )

    new_lgts = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-01T00:00:00Z"],
                utc=True,
            ),
            "airport": ["LGTS"],
            "temperature_2m": [10.0],
        }
    )

    saved = {}

    def fake_save(df, path):
        saved["df"] = df.copy()
        saved["path"] = path


    monkeypatch.setattr(
        weather,
        "save_parquet",
        fake_save,
    )

    monkeypatch.setattr(
        weather,
        "build_route_weather",
        lambda *args, **kwargs: {
            "EDDF": new_eddf,
            "LGTS": new_lgts,
        },
    )
    monkeypatch.setattr(
        weather.config,
        "route_weather_path",
        lambda origin_icao, destination_icao: (
            f"data/processed/{origin_icao}_{destination_icao}/weather.parquet"
        ),
    )
    monkeypatch.setattr(
        weather.config,
        "route_key",
        lambda origin_icao, destination_icao: (
            f"{origin_icao}_{destination_icao}"
        ),
    )
    monkeypatch.setattr(weather, "load_or_none", lambda path: existing)


    result = weather.create_route_dataset("EDDF","LGTS", "2026-01-01", "2026-01-02",)

    assert len(result) == 3
    assert set(result["airport"]) == {"EDDF", "LGTS"}
    assert all(result["route"] == "EDDF_LGTS")
    assert saved["path"].endswith("processed/EDDF_LGTS/weather.parquet")


def test_create_route_dataset_deduplicates_airport_and_time(monkeypatch, mock_config,):
    """
    Verify incremental uniqueness: an exact duplicate identified by the
    airport/time pair is retained only once.
    """
    duplicate_time = pd.Timestamp("2026-01-01T00:00:00Z")

    existing = pd.DataFrame(
        {
            "time": [duplicate_time],
            "airport": ["EDDF"],
            "temperature_2m": [5.0],
        }
    )

    new_eddf = pd.DataFrame(
        {
            "time": [duplicate_time],
            "airport": ["EDDF"],
            "temperature_2m": [5.0],
        }
    )

    new_lgts = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2026-01-01T00:00:00Z")
            ],
            "airport": ["LGTS"],
            "temperature_2m": [10.0],
        }
    )

    monkeypatch.setattr(
        weather,
        "build_route_weather",
        lambda *args, **kwargs: {
            "EDDF": new_eddf,
            "LGTS": new_lgts,
        },
    )

    monkeypatch.setattr(weather, "load_or_none", lambda path: existing)
    monkeypatch.setattr(
        weather,
        "record_stage_stats",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(weather, "save_parquet", lambda *args: None)

    result = weather.create_route_dataset(
        "EDDF",
        "LGTS",
        start_date="2026-01-01",
        end_date="2026-01-01",
    )

    assert len(result) == 2

    eddf_rows = result[result["airport"] == "EDDF"]
    lgts_rows = result[result["airport"] == "LGTS"]

    assert len(eddf_rows) == 1
    assert len(lgts_rows) == 1


def test_create_route_dataset_sorts_by_time_then_airport(monkeypatch, mock_config,):
    """
    Verify deterministic final ordering by timestamp and then airport ICAO.
    """
    eddf = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-02T00:00:00Z"],
                utc=True,
            ),
            "airport": ["EDDF"],
        }
    )

    lgts = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-01T00:00:00Z"],
                utc=True,
            ),
            "airport": ["LGTS"],
        }
    )

    monkeypatch.setattr(
        weather,
        "build_route_weather",
        lambda *args, **kwargs: {
            "EDDF": eddf,
            "LGTS": lgts,
        },
    )
    monkeypatch.setattr(weather, "load_or_none", lambda path: None)
    monkeypatch.setattr(
        weather,
        "record_stage_stats",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(weather, "save_parquet", lambda *args: None)

    result = weather.create_route_dataset("EDDF", "LGTS", "2026-01-01", "2026-01-02",)

    assert result.iloc[0]["airport"] == "LGTS"
    assert result.iloc[1]["airport"] == "EDDF"


def test_create_route_dataset_raises_when_no_weather_frames(monkeypatch, mock_config,):
    """
    Verify that a route with no non-empty airport datasets is treated as a
    failed fetch rather than creating an empty parquet file.
    """
    monkeypatch.setattr(
        weather,
        "build_route_weather",
        lambda *args, **kwargs: {
            "EDDF": pd.DataFrame(),
            "LGTS": pd.DataFrame(),
        },
    )

    monkeypatch.setattr(
        weather,
        "save_parquet",
        lambda *args: pytest.fail(
            "Empty route weather dataset should not be saved"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="No weather data retrieved for route EDDF->LGTS",
    ):
        weather.create_route_dataset(
            "EDDF",
            "LGTS",
            start_date="2026-01-01",
            end_date="2026-01-01",
        )


def test_create_route_dataset_propagates_build_failure(monkeypatch, mock_config,):
    """
    Verify that failures from build_route_weather propagate to the caller so
    the daily orchestrator can classify the route as failed.
    """
    monkeypatch.setattr(
        weather,
        "build_route_weather",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("route weather API failed")
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="route weather API failed",
    ):
        weather.create_route_dataset(
            "EDDF",
            "LGTS",
            start_date="2026-01-01",
            end_date="2026-01-01",
        )


def test_create_route_dataset_uses_default_date_range(monkeypatch, mock_config,):
    """
    Verify that omitted dates fall back to the configured historical start and
    end dates.
    """
    calls = []

    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-01T00:00:00Z"],
                utc=True,
            ),
            "airport": ["EDDF"],
        }
    )

    def fake_build(origin, destination, start, end):
        calls.append((origin, destination, start, end))
        return {
            "EDDF": frame,
            "LGTS": pd.DataFrame(),
        }

    monkeypatch.setattr(weather, "build_route_weather", fake_build,)
    monkeypatch.setattr(weather, "load_or_none", lambda path: None)
    monkeypatch.setattr(weather, "save_parquet", lambda *args: None)
    monkeypatch.setattr(weather,"record_stage_stats",lambda *args, **kwargs: None,)

    result = weather.create_route_dataset(
        "EDDF",
        "LGTS",
    )

    assert len(result) == 1
    assert calls == [
        (
            "EDDF",
            "LGTS",
            "2026-01-01",
            "2026-01-31",
        )
    ]


