import datetime as dt

import pandas as pd
import pytest

from src.features import (
    WEATHER_FEATURES,
    DEPARTURE_WEATHER_COLUMNS,
    ARRIVAL_WEATHER_COLUMNS,
    FINAL_COLUMNS,
    REQUIRED_FLIGHT_COLUMNS,
    validate_required_columns,
    parse_opensky_timestamp,
    validate_timestamp_range,
    validate_no_duplicate_flights,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _unix(y, m, d, h=12, minute=0):

    """
    Convert a UTC date/time into a Unix timestamp in seconds.

    This helper is used throughout the tests to create realistic OpenSky-style
    timestamps. OpenSky normally provides firstSeen and lastSeen as Unix
    timestamps measured in seconds.
    """
    return int(
        dt.datetime(y, m, d, h, minute, tzinfo=dt.timezone.utc,).timestamp()
    )

def _base_flights_df():

    """
    Generate a valid mock dataframe containing every column required by
    REQUIRED_FLIGHT_COLUMNS.
    """
    return pd.DataFrame({
        "icao24": ["aaa111", "bbb222"],
        "firstSeen": [_unix(2024, 1, 10, 9), _unix(2024, 7, 10, 21)],
        "lastSeen": [_unix(2024, 1, 10, 11), _unix(2024, 7, 10, 23)],
        "estDepartureAirport": ["EDDF", "EDDF"],
        "estArrivalAirport": ["LGTS", "LGTS"],
        "callsign": ["AEE1421", "DLH08X"],
        "estDepartureAirportHorizDistance": [100.0, 200.0],
        "estDepartureAirportVertDistance": [10.0, 20.0],
        "estArrivalAirportHorizDistance": [150.0, 250.0],
        "estArrivalAirportVertDistance": [15.0, 25.0],
        "departureAirportCandidatesCount": [1, 2],
        "arrivalAirportCandidatesCount": [1, 2],
        "destination_airport": ["LGTS", "LGTS"],
        "route": ["EDDF->LGTS", "EDDF->LGTS"],
    })

# ---------------------------------------------------------------------------
# Constants / Schema Tests
# ---------------------------------------------------------------------------


def test_weather_features_contains_exact_15_variables():

    """
    Verify that features.py defines exactly the 15 weather variables
    expected by the project.

    Also verify that the departure and arrival weather column names are
    automatically constructed from WEATHER_FEATURES.
    """

    expected = [
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

    # Check that the weather feature list contains exactly the expected
    # variables and in exactly the expected order.
    assert WEATHER_FEATURES == expected
    assert len(WEATHER_FEATURES) == 15

    # Departure weather columns should be prefixed with "dep_".
    assert DEPARTURE_WEATHER_COLUMNS == [
        f"dep_{feature}"
        for feature in expected
    ]

    # Arrival weather columns should be prefixed with "arr_".
    assert ARRIVAL_WEATHER_COLUMNS == [
        f"arr_{feature}"
        for feature in expected
    ]


def test_final_columns_schema_definition_length():

    """
    Verify that FINAL_COLUMNS contains exactly 53 columns.

    This protects the final output schema from accidentally gaining or losing
    columns during future changes to features.py.
    """

    assert len(FINAL_COLUMNS) == 53


def test_final_columns_contains_expected_weather_columns():

    """
    Verify that FINAL_COLUMNS contains every departure and arrival weather
    column defined by the project schema.

    This ensures that the weather features generated during the merge are
    actually represented in the final dataframe schema.
    """

    # Every departure weather feature must exist in FINAL_COLUMNS.
    for column in DEPARTURE_WEATHER_COLUMNS:
        assert column in FINAL_COLUMNS

    # Every arrival weather feature must exist in FINAL_COLUMNS.
    for column in ARRIVAL_WEATHER_COLUMNS:
        assert column in FINAL_COLUMNS


def test_required_flight_columns_are_present_in_base_dataframe():

    """
    Verify that our mock flight dataframe contains every raw column required
    by features.py before feature construction can begin.

    This tests the test fixture itself and helps prevent future tests from
    accidentally using incomplete mock data.
    """

    df = _base_flights_df()

    missing = [
        column
        for column in REQUIRED_FLIGHT_COLUMNS
        if column not in df.columns
    ]

    # The mock dataframe should contain all required columns.
    assert missing == []


# ---------------------------------------------------------------------------
# validate_required_columns() Tests
# ---------------------------------------------------------------------------


def test_validate_required_columns_passes_on_valid_df():

    """
    Verify that validate_required_columns() does nothing when all required
    columns are present.

    A valid dataframe should pass without raising an exception.
    """

    df = _base_flights_df()

    # Should not raise an exception because every required column exists.
    validate_required_columns(
        df,
        REQUIRED_FLIGHT_COLUMNS,
        "FlightsDF",
    )


def test_validate_required_columns_raises_on_missing_columns():

    """
    Verify that validate_required_columns() raises ValueError when one or
    more required columns are missing.

    Here we deliberately remove icao24 and route from the dataframe.
    """

    df = _base_flights_df().drop(
        columns=[
            "route",
            "icao24",
        ]
    )

    # The function should detect the missing columns and raise ValueError.
    with pytest.raises(
        ValueError,
        match="FlightsDF is missing required columns",
    ):
        validate_required_columns(
            df,
            REQUIRED_FLIGHT_COLUMNS,
            "FlightsDF",
        )


def test_validate_required_columns_reports_all_missing_columns():

    """
    Verify that validate_required_columns() reports every missing column,
    rather than stopping after finding only the first missing column.

    This makes schema errors much easier to diagnose.
    """

    df = _base_flights_df().drop(
        columns=[
            "route",
            "icao24",
            "callsign",
        ]
    )

    with pytest.raises(
        ValueError,
        match="FlightsDF is missing required columns",
    ) as exc_info:
        validate_required_columns(
            df,
            REQUIRED_FLIGHT_COLUMNS,
            "FlightsDF",
        )

    error_message = str(exc_info.value)

    # All three missing columns should appear in the error message.
    assert "route" in error_message
    assert "icao24" in error_message
    assert "callsign" in error_message


# ---------------------------------------------------------------------------
# parse_opensky_timestamp() Tests
# ---------------------------------------------------------------------------


def test_parse_opensky_timestamp_handles_unix_integers():

    """
    Verify that Unix timestamps in seconds are correctly converted into
    timezone-aware UTC pandas timestamps.

    This is particularly important because calling pd.to_datetime() on
    Unix seconds without unit="s" can incorrectly interpret them as
    nanoseconds.
    """

    series = pd.Series([
        _unix(2024, 1, 10, 9),
        ])

    parsed = parse_opensky_timestamp(
        series,
        "firstSeen",
    )

    # Pandas 4-compatible check for a timezone-aware datetime dtype.
    assert isinstance(parsed.dtype, pd.DatetimeTZDtype,)

    # The resulting timezone must be UTC.
    assert str(parsed.dt.tz) == "UTC"

    # Verify the actual converted timestamp.
    assert parsed.iloc[0] == pd.Timestamp(
        "2024-01-10 09:00",
        tz="UTC",
    )


def test_parse_opensky_timestamp_handles_multiple_unix_values():

    """
    Verify that multiple Unix timestamps are converted correctly.

    This confirms that the function handles a whole Series rather than
    just a single timestamp.
    """

    series = pd.Series([
        _unix(2024, 1, 10, 9),
        _unix(2024, 1, 10, 11),
        _unix(2024, 7, 10, 21),
    ])

    parsed = parse_opensky_timestamp(
        series,
        "firstSeen",
    )

    # The number of rows must remain unchanged.
    assert len(parsed) == 3

    # All timestamps must use UTC.
    assert str(parsed.dt.tz) == "UTC"

    # Verify each converted timestamp.
    assert parsed.iloc[0] == pd.Timestamp(
        "2024-01-10 09:00",
        tz="UTC",
    )

    assert parsed.iloc[1] == pd.Timestamp(
        "2024-01-10 11:00",
        tz="UTC",
    )

    assert parsed.iloc[2] == pd.Timestamp(
        "2024-07-10 21:00",
        tz="UTC",
    )


def test_parse_opensky_timestamp_handles_existing_datetimes():

    """
    Verify that parse_opensky_timestamp() also accepts a Series that is
    already represented as timezone-aware pandas datetimes.

    The function should not try to interpret these values as Unix seconds.
    """

    series = pd.to_datetime(
        pd.Series([
            "2024-01-10 09:00:00",
        ]),
        utc=True,
    )

    parsed = parse_opensky_timestamp(
        series,
        "firstSeen",
    )

    # Confirm that the result remains timezone-aware.
    assert isinstance(
        parsed.dtype,
        pd.DatetimeTZDtype,
    )

    assert str(parsed.dt.tz) == "UTC"

    # The timestamp should remain unchanged.
    assert parsed.iloc[0] == pd.Timestamp(
        "2024-01-10 09:00",
        tz="UTC",
    )


def test_parse_opensky_timestamp_handles_unix_integers():
    """
    Verify that raw Unix integer timestamps from OpenSky are 
    correctly parsed into timezone-aware UTC datetimes.
    
    1704870000 corresponds exactly to 2024-01-10 07:00:00 UTC.
    """
    # Provide a raw integer like the one that OpenSky actually outputs
    series = pd.Series([1704870000])

    parsed = parse_opensky_timestamp(series)

    # The returned Series must always use the UTC timezone
    assert str(parsed.dt.tz) == "UTC"

    # Verify it matches the exact intended absolute moment
    assert parsed.iloc[0] == pd.Timestamp("2024-01-10 07:00:00", tz="UTC")


def test_parse_opensky_timestamp_raises_on_unparseable_strings():

    """
    Verify that completely invalid timestamp strings cause a clear
    ValueError instead of silently producing NaT values.

    This protects the pipeline from corrupted timestamp data.
    """

    series = pd.Series([
        "invalid_date_string",
    ])

    with pytest.raises(
        ValueError,
        match="could not be parsed as UTC timestamps",
    ):
        parse_opensky_timestamp(
            series,
            "firstSeen",
        )


def test_parse_opensky_timestamp_raises_when_one_value_is_invalid():

    """
    Verify that the parser detects partially invalid timestamp data.

    One valid timestamp and one invalid timestamp should result in a
    ValueError reporting that exactly one value could not be parsed.
    """

    series = pd.Series([
        "2024-01-10 09:00:00",
        "invalid_date", # the second series element is not a date - can't be parsed
    ])

    with pytest.raises(
        ValueError,
        match="contains 1 values",
    ):
        parse_opensky_timestamp(
            series,
            "firstSeen",
        )


# ---------------------------------------------------------------------------
# validate_timestamp_range() Tests
# ---------------------------------------------------------------------------


def test_validate_timestamp_range_passes_on_chronological_order():

    """
    Verify that a normal flight passes timestamp validation.

    A valid flight must satisfy:

        firstSeen < lastSeen
    """

    df = pd.DataFrame({
        "icao24": ["aaa111",],
        "firstSeen": [_unix(2024, 1, 10, 9),],
        "lastSeen": [_unix(2024, 1, 10, 11),],
        "route": ["A->B",],
    })

    # Valid because firstSeen occurs before lastSeen.
    validate_timestamp_range(df)


def test_validate_timestamp_range_passes_for_multiple_valid_flights():

    """
    Verify that multiple valid flights can be checked together.

    Every row has firstSeen earlier than lastSeen, so no exception should
    be raised.
    """

    df = pd.DataFrame({
        "icao24": ["aaa111", "bbb222",],
        "firstSeen": [_unix(2024, 1, 10, 9), _unix(2024, 1, 10, 15),],
        "lastSeen": [_unix(2024, 1, 10, 11), _unix(2024, 1, 10, 17),],
        "route": ["A->B", "B->C",],
    })

    # Both flights have positive durations.
    validate_timestamp_range(df)


def test_validate_timestamp_range_raises_on_zero_duration():

    """
    Verify that a flight with firstSeen == lastSeen is rejected.

    A zero-duration flight is invalid according to features.py.
    """

    df = pd.DataFrame({
        "icao24": ["aaa111",],
        "firstSeen": [_unix(2024, 1, 10, 10),],
        "lastSeen": [_unix(2024, 1, 10, 10),],
        "route": ["A->B",],
    })

    with pytest.raises(
        ValueError,
        match="Found 1 flights with lastSeen <= firstSeen",
    ):
        validate_timestamp_range(df)


def test_validate_timestamp_range_raises_on_negative_duration():

    """
    Verify that a flight where lastSeen occurs before firstSeen is rejected.

    This represents an impossible negative-duration flight observation.
    """

    df = pd.DataFrame({
        "icao24": ["bbb222",],
        "firstSeen": [_unix(2024, 1, 10, 11),],
        "lastSeen": [_unix(2024, 1, 10, 10),],
        "route": ["A->B",],
    })

    with pytest.raises(
        ValueError,
        match="Found 1 flights with lastSeen <= firstSeen",
    ):
        validate_timestamp_range(df)


def test_validate_timestamp_range_reports_multiple_invalid_flights():

    """
    Verify that the validator counts every invalid flight in the dataframe.

    In this test:
        - Flight aaa111 has zero duration.
        - Flight bbb222 has negative duration.
        - Flight ccc333 is valid.

    Therefore exactly two flights should be reported as invalid.
    """

    df = pd.DataFrame({
        "icao24": ["aaa111", "bbb222", "ccc333",],
        "firstSeen": [_unix(2024, 1, 10, 10), _unix(2024, 1, 10, 12), _unix(2024, 1, 10, 15),],
        "lastSeen": [_unix(2024, 1, 10, 10), _unix(2024, 1, 10, 11), _unix(2024, 1, 10, 17),],
        "route": ["A->B", "B->C", "C->D",],
    })

    with pytest.raises(
        ValueError,
        match="Found 2 flights with lastSeen <= firstSeen",
    ):
        validate_timestamp_range(df)


# ---------------------------------------------------------------------------
# validate_no_duplicate_flights() Tests
# ---------------------------------------------------------------------------


def test_validate_no_duplicate_flights_passes_for_unique_flights():

    """
    Verify that unique flight observations pass duplicate validation.

    The base dataframe contains two different flight observations, so no
    exception should be raised.
    """

    df = _base_flights_df()

    validate_no_duplicate_flights(df)


def test_validate_no_duplicate_flights_raises_for_duplicate_flight():

    """
    Verify that an exact duplicate flight observation is rejected.

    The duplicate definition in features.py uses:

        icao24
        firstSeen
        lastSeen
        estDepartureAirport
        estArrivalAirport

    Both rows have identical values for all five fields, so the second
    observation is considered a duplicate.
    """

    df = pd.DataFrame({
        "icao24": ["aaa111", "aaa111",],
        "firstSeen": [_unix(2024, 1, 10, 9), _unix(2024, 1, 10, 9),],
        "lastSeen": [_unix(2024, 1, 10, 11), _unix(2024, 1, 10, 11),],
        "estDepartureAirport": ["EDDF", "EDDF",],
        "estArrivalAirport": ["LGTS","LGTS",],
    })

    with pytest.raises(
        ValueError,
        match="duplicated flight observations",
    ):
        validate_no_duplicate_flights(df)


def test_validate_no_duplicate_flights_allows_same_icao24_for_different_flights():

    """
    Verify that the same aircraft is allowed to appear multiple times when
    the observations represent different flights.

    This is important because one aircraft can perform many flights during
    the dataset period.

    The two rows use the same icao24 but have different timestamps and
    airports, so they must NOT be considered duplicates.
    """

    df = pd.DataFrame({
        "icao24": ["aaa111", "aaa111",],
        "firstSeen": [_unix(2024, 1, 10, 9), _unix(2024, 1, 10, 15),],
        "lastSeen": [_unix(2024, 1, 10, 11), _unix(2024, 1, 10, 17),],
        "estDepartureAirport": ["EDDF", "LGTS",],
        "estArrivalAirport": ["LGTS", "EDDF",],
    })

    # Same aircraft, but clearly different flight observations.
    validate_no_duplicate_flights(df)