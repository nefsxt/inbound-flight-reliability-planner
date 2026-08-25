import datetime as dt

import os
import pandas as pd
import pytest
import requests

import src.fetch_opensky as opensky


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_token_cache():
    opensky._token_cache["access_token"] = None
    yield
    opensky._token_cache["access_token"] = None


@pytest.fixture
def mock_config(monkeypatch, tmp_path):
    monkeypatch.setattr(opensky.config, "PROCESSED_DIR", str(tmp_path / "processed"))
    monkeypatch.setattr(opensky.config, "RAW_DIR", str(tmp_path / "raw"))

    monkeypatch.setattr(opensky.config, "OPENSKY_CLIENT_ID", "test-client")
    monkeypatch.setattr(opensky.config, "OPENSKY_CLIENT_SECRET", "test-secret")

    monkeypatch.setattr(opensky.config, "OPENSKY_AUTH_TIMEOUT_SECONDS", 10)
    monkeypatch.setattr(opensky.config, "OPENSKY_API_TIMEOUT_SECONDS", 10)

    monkeypatch.setattr(opensky.config, "OPENSKY_QUERY_CHUNK_DAYS", 7)
    monkeypatch.setattr(opensky.config, "OPENSKY_REQUEST_PAUSE_SECONDS", 0)

    return tmp_path


# ---------------------------------------------------------------------------
# get_token
# ---------------------------------------------------------------------------

def test_get_token_requests_and_caches_token(monkeypatch, mock_config):
    """
    Verify that get_token fetches a new token on the first call via an HTTP POST,
    and returns the cached token on subsequent calls without hitting the network.
    """
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"access_token": "abc123"}

    def fake_post(url, data, timeout):
        calls.append((url, data, timeout))
        return FakeResponse()

    monkeypatch.setattr(opensky.requests, "post", fake_post)

    token1 = opensky.get_token()
    token2 = opensky.get_token()

    assert token1 == "abc123"
    assert token2 == "abc123"
    assert len(calls) == 1 # Only one request should be made due to caching


def test_get_token_requires_credentials(monkeypatch, mock_config):
    """
    Ensure a RuntimeError is raised with a descriptive message if the 
    OPENSKY_CLIENT_ID configuration setting is missing or empty.
    """
    monkeypatch.setattr(opensky.config, "OPENSKY_CLIENT_ID", None)

    with pytest.raises(RuntimeError, match="Missing OPENSKY_CLIENT_ID"):
        opensky.get_token()


def test_get_token_requires_secret(monkeypatch, mock_config):
    """
    Ensure a RuntimeError is raised with a descriptive message if the 
    OPENSKY_CLIENT_SECRET configuration setting is missing or empty.
    """
    monkeypatch.setattr(opensky.config, "OPENSKY_CLIENT_SECRET", None)

    # Note: using a broader match to match your actual string
    with pytest.raises(RuntimeError, match="Missing OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET"):
        opensky.get_token()


def test_get_token_handles_http_error(monkeypatch, mock_config):
    """
    Verify that any HTTP errors (like 401 Unauthorized or 500 Server Error) 
    thrown by the API are allowed to bubble up naturally from requests.
    """
    class FakeHttpErrorResponse:
        def raise_for_status(self):
            # Simulate a 401 Unauthorized exception from requests
            raise requests.exceptions.HTTPError("401 Client Error: Unauthorized")

    monkeypatch.setattr(opensky.requests, "post", lambda *a, **kw: FakeHttpErrorResponse())

    with pytest.raises(requests.exceptions.HTTPError):
        opensky.get_token()


def test_get_token_handles_malformed_json(monkeypatch, mock_config):
    """
    Ensure that a KeyError is raised if the API responds with a successful 
    status code but leaves out the expected 'access_token' key in the body.
    """
    class FakeMalformedResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"unexpected_key": "oops"} # Missing 'access_token'

    monkeypatch.setattr(opensky.requests, "post", lambda *a, **kw: FakeMalformedResponse())

    with pytest.raises(KeyError):
        opensky.get_token()



# ---------------------------------------------------------------------------
# _get
# ---------------------------------------------------------------------------

def test_get_returns_json(monkeypatch, mock_config):
    """
    Verify the happy path: a successful 200 OK HTTP GET request 
    properly extracts, structures, and returns the JSON payload.
    """
    monkeypatch.setattr(opensky,"get_token",lambda: "token",)

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return [{"icao24": "abc"}]

    monkeypatch.setattr(opensky.requests,"get",lambda *args, **kwargs: FakeResponse(),)

    result = opensky._get(
        "flights/arrival",
        {"airport": "LGTS"},
    )

    assert result == [{"icao24": "abc"}]


def test_get_404_returns_empty_list(monkeypatch, mock_config):
    """
    Ensure that a 404 status code is treated gracefully as an empty dataset
    rather than a failure, returning an empty list per the OpenSky API convention.
    """
    monkeypatch.setattr(opensky, "get_token", lambda: "token")

    class FakeResponse:
        status_code = 404

    monkeypatch.setattr(opensky.requests,"get",lambda *args, **kwargs: FakeResponse(),)

    assert opensky._get("flights/arrival", {}) == []


def test_get_refreshes_token_once_after_401(monkeypatch, mock_config):
    """
    Verify the single-retry loop: when an initial request fails with a 401,
    the cache must be cleared, a fresh token requested, and the call retried.
    """
    tokens = iter(["expired-token", "fresh-token"])
    calls = []

    # Mock the cache so we can observe if it gets cleared
    fake_cache = {"access_token": "expired-token"}
    monkeypatch.setattr(opensky, "_token_cache", fake_cache)
    monkeypatch.setattr(opensky, "get_token", lambda: next(tokens))

    class FakeResponse:
        def __init__(self, status_code, data=None):
            self.status_code = status_code
            self._data = data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError(response=self)

        def json(self):
            return self._data

    responses = [
        FakeResponse(401),
        FakeResponse(200, [{"icao24": "abc"}]),
    ]

    def fake_get(*args, **kwargs):
        # After the first request (401), but before the second request executes,
        # the token cache should have been wiped by the implementation.
        if len(calls) == 1:
            assert fake_cache["access_token"] is None
            
        calls.append(kwargs["headers"]["Authorization"])
        return responses.pop(0)

    monkeypatch.setattr(opensky.requests, "get", fake_get)

    result = opensky._get(
        "flights/arrival",
        {"airport": "LGTS"},
    )

    assert result == [{"icao24": "abc"}]
    assert calls == [
        "Bearer expired-token",
        "Bearer fresh-token",
    ]

def test_get_raises_after_second_401(monkeypatch, mock_config):
    """
    Prevent infinite loops: confirm that a RuntimeError is thrown if a 401 
    unauthorized error persists even after obtaining a fresh token.
    """
    monkeypatch.setattr(opensky,"get_token",lambda: "token",)

    class FakeResponse:
        status_code = 401
        text = "Unauthorized"

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(response=self)

    monkeypatch.setattr(opensky.requests,"get",lambda *args, **kwargs: FakeResponse(),)

    with pytest.raises(RuntimeError, match="authentication failed"):
        opensky._get("flights/arrival", {})


def test_get_bubbles_up_non_401_errors(monkeypatch, mock_config):
    """
    Ensure non-authentication HTTP errors (like a 500 Server Error or 429 Rate Limit)
    bypass the token refresh retry loop and bubble up instantly via raise_for_status.
    """
    monkeypatch.setattr(opensky, "get_token", lambda: "token")

    class FakeServerErrorResponse:
        status_code = 500
        
        def raise_for_status(self):
            raise requests.exceptions.HTTPError("500 Internal Server Error")

    monkeypatch.setattr(opensky.requests,"get",lambda *args, **kwargs: FakeServerErrorResponse(),)

    with pytest.raises(requests.exceptions.HTTPError, match="500 Internal Server Error"):
        opensky._get("flights/arrival", {})


# ---------------------------------------------------------------------------
# fetch_arrivals_for_airport
# ---------------------------------------------------------------------------

def test_fetch_arrivals_successfully_caches_chunk(monkeypatch, mock_config):
    """
    Verify the happy path: API fetches a valid chunk of flight rows, 
    persists it into a local parquet file, and outputs a complete DataFrame.
    """
    saved = {}

    monkeypatch.setattr(opensky,"daterange_chunks",lambda start, end, days: [(dt.date(2026, 8, 1), dt.date(2026, 8, 2))],)
    monkeypatch.setattr(opensky,"_get",lambda endpoint, params: [{"icao24": "abc", "firstSeen": 100, "lastSeen": 200}],)
    monkeypatch.setattr(opensky, "cache_path", lambda *args: "chunk.parquet")
    monkeypatch.setattr(opensky, "load_or_none", lambda path: None)

    def fake_save(df, path):
        saved["df"] = df
        saved["path"] = path

    monkeypatch.setattr(opensky, "save_parquet", fake_save)
    monkeypatch.setattr(opensky, "log_credit_usage", lambda *a, **k: None)
    monkeypatch.setattr(opensky, "polite_sleep", lambda *a, **k: None)

    result = opensky.fetch_arrivals_for_airport("LGTS", "2026-08-01", "2026-08-02", "cache")

    assert len(result) == 1
    assert result.iloc[0]["icao24"] == "abc"
    assert len(saved["df"]) == 1


def test_empty_api_response_is_cached(monkeypatch, mock_config):
    """
    Ensure that a successful API return with 0 entries is still cached locally 
    as an empty DataFrame, recording that the date window contains no flights.
    """
    saved = {}

    monkeypatch.setattr(opensky,"daterange_chunks",lambda start, end, days: [(dt.date(2026, 8, 1), dt.date(2026, 8, 2))],)
    monkeypatch.setattr(opensky, "_get", lambda endpoint, params: [])
    monkeypatch.setattr(opensky, "cache_path", lambda *args: "empty_chunk.parquet")
    monkeypatch.setattr(opensky, "load_or_none", lambda path: None)
    monkeypatch.setattr(opensky, "log_credit_usage", lambda *a, **k: None)
    monkeypatch.setattr(opensky, "polite_sleep", lambda *a, **k: None)
    monkeypatch.setattr(opensky, "save_parquet", lambda df, path: saved.update(df=df, path=path))

    result = opensky.fetch_arrivals_for_airport("LGTS", "2026-08-01", "2026-08-02", "cache")

    assert result.empty
    assert "df" in saved
    assert saved["df"].empty


def test_fetch_reads_from_cache_and_bypasses_api(monkeypatch, mock_config):
    """
    Verify performance efficiency: if a chunk is already cached on disk, 
    the function must load it directly and never initiate an outbound API call.
    """
    monkeypatch.setattr(opensky,"daterange_chunks",lambda start, end, days: [(dt.date(2026, 8, 1), dt.date(2026, 8, 2))],)

    # Return a dummy dataframe from the cache simulator
    monkeypatch.setattr(opensky, "load_or_none", lambda path: pd.DataFrame([{"icao24": "cached-plane"}]))
    monkeypatch.setattr(opensky, "cache_path", lambda *args: "cached_chunk.parquet")

    # If _get is called, fail the test immediately
    def explosive_get(*a, **kw):
        pytest.fail("Outbound API called even though chunk was completely cached!")
    
    monkeypatch.setattr(opensky, "_get", explosive_get)

    result = opensky.fetch_arrivals_for_airport("LGTS", "2026-08-01", "2026-08-02", "cache")
    assert len(result) == 1
    assert result.iloc[0]["icao24"] == "cached-plane"


def test_fetch_handles_temporary_429_rate_limiting(monkeypatch, mock_config):
    """
    Validate error recovery loops: if the network returns an HTTP 429 Rate Limit error, 
    the engine should sleep to back off, retry the request, and succeed if clear.
    """
    sleep_calls = []
    api_attempts = []

    class Fake429Response:
        status_code = 429
        text = "Too Many Requests"

    def mock_get_flaky(*a, **kw):
        api_attempts.append(1)
        if len(api_attempts) == 1:
            # First attempt fails with rate limit
            err = requests.exceptions.HTTPError()
            err.response = Fake429Response()
            raise err
        # Second attempt succeeds
        return [{"icao24": "retry-success"}]

    monkeypatch.setattr(opensky,"daterange_chunks",lambda start, end, days: [(dt.date(2026, 8, 1), dt.date(2026, 8, 2))],)
    monkeypatch.setattr(opensky, "_get", mock_get_flaky)
    monkeypatch.setattr(opensky, "polite_sleep", lambda secs: sleep_calls.append(secs))
    monkeypatch.setattr(opensky, "cache_path", lambda *args: "retry_chunk.parquet")
    monkeypatch.setattr(opensky, "load_or_none", lambda path: None)
    monkeypatch.setattr(opensky, "save_parquet", lambda *a, **k: None)
    monkeypatch.setattr(opensky, "log_credit_usage", lambda *a, **k: None)

    # Force OPENSKY_REQUEST_PAUSE_SECONDS to a distinct value to track it uniquely
    monkeypatch.setattr(opensky.config, "OPENSKY_REQUEST_PAUSE_SECONDS", 5)

    result = opensky.fetch_arrivals_for_airport("LGTS", "2026-08-01", "2026-08-02", "cache")
    
    assert len(api_attempts) == 2  # Proves it tried a second time
    
    # The first sleep call must be the exponential backoff calculation:
    # config.OPENSKY_REQUEST_PAUSE_SECONDS * (2 ** (attempt + 1)) -> 5 * (2 ** 1) = 10
    assert sleep_calls[0] == 10    
    
    # The second sleep call is the trailing chunk pacing check at the loop floor
    assert sleep_calls[1] == 5
    
    assert result.iloc[0]["icao24"] == "retry-success"


def test_fetch_raises_runtime_error_on_persistent_429(monkeypatch, mock_config):
    """
    Ensure fail-safe thresholds: check that a clear RuntimeError is raised if 
    the target server maintains a 429 Rate Limit block throughout all retry limits.
    """
    class Fake429Response:
        status_code = 429
        text = "Still Rate Limited"

    def mock_get_permanently_blocked(*a, **kw):
        err = requests.exceptions.HTTPError()
        err.response = Fake429Response()
        raise err

    monkeypatch.setattr(opensky,"daterange_chunks",lambda start, end, days: [(dt.date(2026, 8, 1), dt.date(2026, 8, 2))],)
    monkeypatch.setattr(opensky, "_get", mock_get_permanently_blocked)
    monkeypatch.setattr(opensky, "polite_sleep", lambda secs: None)
    monkeypatch.setattr(opensky, "cache_path", lambda *args: "failed_chunk.parquet")
    monkeypatch.setattr(opensky, "load_or_none", lambda path: None)

    with pytest.raises(RuntimeError, match="rate limit persisted after 3 retries"):
        opensky.fetch_arrivals_for_airport("LGTS", "2026-08-01", "2026-08-02", "cache")


def test_fetch_raises_immediately_on_400_bad_request(monkeypatch, mock_config):
    """
    Verify immediate termination constraints: an HTTP 400 means structural formatting errors 
    (e.g., malformed airport string), so it must throw a RuntimeError instantly without retrying.
    """
    api_calls = []

    class Fake400Response:
        status_code = 400
        text = "Bad Request Parameter"

    def mock_get_bad_request(*a, **kw):
        api_calls.append(1)
        err = requests.exceptions.HTTPError()
        err.response = Fake400Response()
        raise err

    monkeypatch.setattr(opensky,"daterange_chunks",lambda start, end, days: [(dt.date(2026, 8, 1), dt.date(2026, 8, 2))],)
    monkeypatch.setattr(opensky, "_get", mock_get_bad_request)
    monkeypatch.setattr(opensky, "cache_path", lambda *args: "400_chunk.parquet")
    monkeypatch.setattr(opensky, "load_or_none", lambda path: None)

    with pytest.raises(RuntimeError, match="returned HTTP 400"):
        opensky.fetch_arrivals_for_airport("LGTS", "2026-08-01", "2026-08-02", "cache")
        
    assert len(api_calls) == 1  # Confirms no retries were wasted on structural exceptions


# ---------------------------------------------------------------------------
# _sanity_filter
# ---------------------------------------------------------------------------

def test_sanity_filter_removes_invalid_rows():
    """
    Verify structural filtering logic: confirm that rows with missing required attributes 
    (timestamps/airports) or an impossibly long flight duration (> 300 minutes) are dropped.
    """
    df = pd.DataFrame(
        [
            # Row 0: Valid row (Duration = 100s / 1.6min) -> NOTE: This will be caught by the 5-min bug below!
            {
                "firstSeen": 100,
                "lastSeen": 700,  # 700 - 100 = 600 seconds (10 minutes) -> Valid
                "estDepartureAirport": "EDDF",
            },
            # Row 1: Invalid long flight (Duration = 301 minutes) -> Should be dropped
            {
                "firstSeen": 100,
                "lastSeen": 100 + 60 * 301,
                "estDepartureAirport": "EDDF",
            },
            # Row 2: Missing timestamp -> Should be dropped
            {
                "firstSeen": None,
                "lastSeen": 200,
                "estDepartureAirport": "EDDF",
            },
            # Row 3: Missing airport -> Should be dropped
            {
                "firstSeen": 100,
                "lastSeen": 200,
                "estDepartureAirport": None,
            },
        ]
    )

    result = opensky._sanity_filter(df)

    # Only Row 0 meets all sanity conditions
    assert len(result) == 1


def test_sanity_filter_drops_short_flight_boundaries():
    """
    Verify boundary data validation: ensure that flights lasting exactly 5 minutes or less 
    are caught and removed as implausible transponder noise or logging artifacts.
    """
    df = pd.DataFrame(
        [
            # Row 0: Exactly 5 minutes (300 seconds) -> Should be dropped due to strict inequality (> 5)
            {
                "firstSeen": 100,
                "lastSeen": 400,
                "estDepartureAirport": "EDDF",
            },
            # Row 1: Under 5 minutes (240 seconds) -> Should be dropped
            {
                "firstSeen": 100,
                "lastSeen": 340,
                "estDepartureAirport": "EDDF",
            },
            # Row 2: Just over 5 minutes (301 seconds) -> Valid, must be kept
            {
                "firstSeen": 100,
                "lastSeen": 401,
                "estDepartureAirport": "EDDF",
            }
        ]
    )

    result = opensky._sanity_filter(df)

    assert len(result) == 1
    assert result.iloc[0]["lastSeen"] == 401


# ---------------------------------------------------------------------------
# _last_cached_chunk_date Units
# ---------------------------------------------------------------------------

def test_last_cached_chunk_date_extracts_maximum_date(monkeypatch, mock_config):
    """
    Verify the happy path: scanning a directory containing multiple valid airport chunks 
    properly parses the filenames and identifies the absolute latest calendar date.
    """
    test_dir = mock_config / "cache"
    os.makedirs(test_dir)

    # Populate zero-byte files following the pattern: arrivals_{icao}_{start}_{end}.parquet
    valid_files = [
        "arrivals_LGTS_2026-08-01_2026-08-07.parquet",
        "arrivals_LGTS_2026-08-08_2026-08-14.parquet",  # The latest tracking date
        "arrivals_LGTS_2026-07-01_2026-07-07.parquet",
    ]
    
    for filename in valid_files:
        with open(test_dir / filename, "w") as f:
            f.write("")

    result = opensky._last_cached_chunk_date("LGTS", str(test_dir))
    assert result == dt.date(2026, 8, 14)


def test_last_cached_chunk_date_skips_malformed_or_unrelated_files(monkeypatch, mock_config):
    """
    Ensure robust file filtering: non-parquet formats, alternative destinations, or 
    corrupted string structures must be bypassed without crashing the loop.
    """
    test_dir = mock_config / "cache"
    os.makedirs(test_dir)

    invalid_files = [
        "arrivals_EDDF_2026-08-01_2026-08-14.parquet",  # Wrong airport code
        "arrivals_LGTS_2026-08-01_2026-08-14.tmp",      # Invalid extension type
        "arrivals_LGTS_broken-date-string.parquet",     # Unparsable date fields
        "random_system_log.txt"                          # Total mismatch
    ]
    
    for filename in invalid_files:
        with open(test_dir / filename, "w") as f:
            f.write("")

    result = opensky._last_cached_chunk_date("LGTS", str(test_dir))
    assert result is None


def test_last_cached_chunk_date_returns_none_if_directory_missing(monkeypatch, mock_config):
    """
    Verify safety boundary condition: if the targeted cache directory path 
    does not exist, return None rather than raising an OS file exception.
    """
    non_existent_path = "/this/path/does/not/exist/anywhere"
    result = opensky._last_cached_chunk_date("LGTS", non_existent_path)
    assert result is None


# ---------------------------------------------------------------------------
# get_last_flight_date Units
# ---------------------------------------------------------------------------

def test_get_last_flight_date_uses_latest_processed_date(monkeypatch, mock_config):
    """
    Ensure that when a processed dataset with active flight records exists, 
    the maximum date is cleanly derived using its 'firstSeen' Unix timestamps.
    """
    # 1,700,000,000 Unix timestamp translates to 2023-11-14 UTC and 1,700,100,000 to 2023-11-16 
    df = pd.DataFrame({"firstSeen": [1_700_000_000, 1_700_100_000]})

    monkeypatch.setattr(opensky, "load_or_none", lambda path: df)
    monkeypatch.setattr(opensky.config, "route_flights_path", lambda *a: "flights.parquet")
    monkeypatch.setattr(opensky.config, "route_arrivals_dir", lambda *a: "cache")
    monkeypatch.setattr(opensky, "_last_cached_chunk_date", lambda *a: None)

    result = opensky.get_last_flight_date("EDDF", "LGTS")
    assert isinstance(result, dt.date)
    assert result == dt.date(2023, 11, 16)


def test_get_last_flight_date_considers_empty_cached_chunks(monkeypatch, mock_config):
    """
    Verify incremental update safety: when the data frame is empty, falling back to 
    the raw chunk file boundaries ensures zero-flight windows still advance tracking positions.
    """
    monkeypatch.setattr(opensky, "load_or_none", lambda path: pd.DataFrame())
    monkeypatch.setattr(opensky, "_last_cached_chunk_date", lambda *a: dt.date(2026, 8, 10))
    monkeypatch.setattr(opensky.config, "route_flights_path", lambda *a: "flights.parquet")
    monkeypatch.setattr(opensky.config, "route_arrivals_dir", lambda *a: "cache")

    result = opensky.get_last_flight_date("EDDF", "LGTS")
    assert result == dt.date(2026, 8, 10)


def test_get_last_flight_date_resolves_maximum_between_both_sources(monkeypatch, mock_config):
    """
    Confirm comparison precedence: when both processed data frames and cache logs return dates, 
    the function must return the absolute maximum value to prevent redundant sync operations.
    """
    # 1,700,000,000 translates to 2023-11-14
    df = pd.DataFrame({"firstSeen": [1_700_000_000]})

    monkeypatch.setattr(opensky, "load_or_none", lambda path: df)
    # Return a date further into the future (2026) from the cache module
    monkeypatch.setattr(opensky, "_last_cached_chunk_date", lambda *a: dt.date(2026, 8, 15))
    monkeypatch.setattr(opensky.config, "route_flights_path", lambda *a: "flights.parquet")
    monkeypatch.setattr(opensky.config, "route_arrivals_dir", lambda *a: "cache")

    result = opensky.get_last_flight_date("EDDF", "LGTS")
    
    # 2026-08-15 should be picked over 2023-11-14
    assert result == dt.date(2026, 8, 15)


# ---------------------------------------------------------------------------
# build_route_dataset
# ---------------------------------------------------------------------------

def test_build_route_dataset_filters_origin_and_merges(monkeypatch, mock_config):
    """
    Verify orchestrator flow: ensure rows from non-matching departure origins are filtered out, 
    and matching newly-fetched rows are appended to the preexisting parquet dataset.
    """
    new_data = pd.DataFrame([
        {"icao24": "abc", "firstSeen": 100, "lastSeen": 200, "estDepartureAirport": "EDDF"},
        {"icao24": "wrong", "firstSeen": 300, "lastSeen": 400, "estDepartureAirport": "XXXX"}
    ])

    # Added 'route' and 'destination_airport' columns to mimic a real historical file output
    existing = pd.DataFrame([
        {
            "icao24": "existing", 
            "firstSeen": 50, 
            "lastSeen": 75, 
            "estDepartureAirport": "EDDF",
            "destination_airport": "LGTS",
            "route": "EDDF->LGTS"
        }
    ])

    monkeypatch.setattr(opensky, "fetch_arrivals_for_airport", lambda *args, **kwargs: new_data)
    monkeypatch.setattr(opensky, "load_or_none", lambda path: existing)
    monkeypatch.setattr(opensky.config, "route_arrivals_dir", lambda *a: "cache")
    monkeypatch.setattr(opensky.config, "route_flights_path", lambda *a: "flights.parquet")

    saved = {}
    monkeypatch.setattr(opensky, "save_parquet", lambda df, path: saved.update(df=df.copy(), path=path))
    monkeypatch.setattr(opensky, "_sanity_filter", lambda df: df)
    monkeypatch.setattr(opensky, "record_stage_stats", lambda *a, **k: None)

    result = opensky.build_route_dataset("EDDF", "LGTS", start_date="2026-08-01", end_date="2026-08-02")

    assert len(result) == 2
    assert set(result["icao24"]) == {"abc", "existing"}
    assert all(result["route"] == "EDDF->LGTS")
    assert saved["path"] == "flights.parquet"



def test_build_route_dataset_empty_fetch_does_not_overwrite_existing(monkeypatch, mock_config):
    """
    Ensure dataset safety: if a new data query returns absolutely zero rows, 
    the preexisting data storage on disk must remain un-mutated and completely untouched.
    """
    existing = pd.DataFrame([{"icao24": "existing", "firstSeen": 100, "lastSeen": 200}])

    monkeypatch.setattr(opensky, "fetch_arrivals_for_airport", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(opensky, "load_or_none", lambda path: existing)
    monkeypatch.setattr(opensky.config, "route_arrivals_dir", lambda *a: "cache")
    monkeypatch.setattr(opensky.config, "route_flights_path", lambda *a: "flights.parquet")
    
    # Intercept file writes to assert that zero writes occur during an empty fetch return
    monkeypatch.setattr(opensky, "save_parquet", lambda *a: pytest.fail("Preexisting processed dataset should not be overwritten"))

    result = opensky.build_route_dataset("EDDF", "LGTS", start_date="2026-08-01", end_date="2026-08-02")
    pd.testing.assert_frame_equal(result, existing)


def test_build_route_dataset_deduplicates_overlapping_rows(monkeypatch, mock_config):
    """
    Verify incremental update uniqueness: if a newly fetched record overlaps exactly 
    (same icao24/firstSeen/lastSeen) with a stored record, drop the duplicate copy.
    """
    # Overlapping flight 'abc' fetched again
    new_data = pd.DataFrame([
        {"icao24": "abc", "firstSeen": 100, "lastSeen": 200, "estDepartureAirport": "EDDF"}
    ])

    # 'abc' already resides in local disk history
    existing = pd.DataFrame([
        {"icao24": "abc", "firstSeen": 100, "lastSeen": 200, "estDepartureAirport": "EDDF"}
    ])

    monkeypatch.setattr(opensky, "fetch_arrivals_for_airport", lambda *args, **kwargs: new_data)
    monkeypatch.setattr(opensky, "load_or_none", lambda path: existing)
    monkeypatch.setattr(opensky.config, "route_arrivals_dir", lambda *a: "cache")
    monkeypatch.setattr(opensky.config, "route_flights_path", lambda *a: "flights.parquet")
    monkeypatch.setattr(opensky, "save_parquet", lambda *a, **k: None)
    monkeypatch.setattr(opensky, "_sanity_filter", lambda df: df)
    monkeypatch.setattr(opensky, "record_stage_stats", lambda *a, **k: None)

    result = opensky.build_route_dataset("EDDF", "LGTS", start_date="2026-08-01", end_date="2026-08-02")
    
    # Deduplication should leave only a single instance of the flight record
    assert len(result) == 1


def test_build_route_dataset_initializes_fresh_parquet_if_none_exists(monkeypatch, mock_config):
    """
    Verify historical initialization: if no target parquet file exists yet (fresh system run), 
    the engine must bypass data matching and construct a brand-new dataset safely.
    """
    new_data = pd.DataFrame([
        {"icao24": "fresh-flight", "firstSeen": 100, "lastSeen": 200, "estDepartureAirport": "EDDF"}
    ])

    monkeypatch.setattr(opensky, "fetch_arrivals_for_airport", lambda *args, **kwargs: new_data)
    # Simulate completely empty tracking history file structure
    monkeypatch.setattr(opensky, "load_or_none", lambda path: None)
    
    monkeypatch.setattr(opensky.config, "route_arrivals_dir", lambda *a: "cache")
    monkeypatch.setattr(opensky.config, "route_flights_path", lambda *a: "flights.parquet")
    
    saved_df = []
    monkeypatch.setattr(opensky, "save_parquet", lambda df, path: saved_df.append(df))
    monkeypatch.setattr(opensky, "_sanity_filter", lambda df: df)
    monkeypatch.setattr(opensky, "record_stage_stats", lambda *a, **k: None)

    result = opensky.build_route_dataset("EDDF", "LGTS", start_date="2026-08-01", end_date="2026-08-02")
    
    assert len(saved_df) == 1
    assert len(result) == 1
    assert result.iloc[0]["icao24"] == "fresh-flight"


