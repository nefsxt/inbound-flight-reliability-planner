import datetime as dt
import os
import pandas as pd
import pytest

from src.utils import (
    daterange_chunks, to_unix, cache_path, save_parquet, load_or_none,
    null_rate_report, record_stage_stats, load_pipeline_stats,
)

# ---------------------------------------------------------------------------
# daterange_chunks
# ---------------------------------------------------------------------------

def test_daterange_chunks_covers_full_range_inclusive():
    """
    Verify full interval coverage: ensure date-slicing maps every calendar day 
    to exactly one chunk without dropouts, gaps, or structural overlaps.
    """
    chunks = list(daterange_chunks("2024-01-01", "2024-01-05", chunk_days=2))
    assert chunks[0] == (dt.date(2024, 1, 1), dt.date(2024, 1, 2))
    assert chunks[-1][1] == dt.date(2024, 1, 5)
    
    covered = set()
    for start, end in chunks:
        d = start
        while d <= end:
            covered.add(d)
            d += dt.timedelta(days=1)
            
    expected = {dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(5)}
    assert covered == expected


def test_daterange_chunks_single_day():
    """
    Verify boundary ranges: check that passing identical start and end dates 
    outputs a single, unified one-day coordinate tuple.
    """
    chunks = list(daterange_chunks("2024-01-01", "2024-01-01", chunk_days=1))
    assert chunks == [(dt.date(2024, 1, 1), dt.date(2024, 1, 1))]


# ---------------------------------------------------------------------------
# to_unix
# ---------------------------------------------------------------------------

def test_to_unix_is_utc_midnight():
    """
    Confirm time zone precision: check that a date object converts accurately 
    into its absolute Unix epoch equivalent at UTC midnight.
    """
    ts = to_unix(dt.date(2024, 1, 1))
    expected = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp())
    assert ts == expected


# ---------------------------------------------------------------------------
# cache_path
# ---------------------------------------------------------------------------

def test_cache_path_sanitizes_key():
    """
    Verify string normalization: ensure file path outputs resolve cleanly 
    with proper local OS delimiters and sub-directory prefixes.
    """
    path = cache_path("data/raw", "arrivals", "LGTS_2024-01-01_2024-01-01")
    assert path == os.path.join("data/raw", "arrivals_LGTS_2024-01-01_2024-01-01.parquet")


# ---------------------------------------------------------------------------
# save_parquet & load_or_none
# ---------------------------------------------------------------------------

def test_save_and_load_parquet_roundtrip(tmp_path):
    """
    Validate binary serialization: check that saving data to a parquet file 
    and reading it back yields an identical, uncorrupted DataFrame schema.
    """
    df = pd.DataFrame({"a": [1, 2, 3]})
    path = os.path.join(tmp_path, "sub", "test.parquet")
    save_parquet(df, path)
    loaded = load_or_none(path)
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, df)


def test_load_or_none_missing_file(tmp_path):
    """
    Ensure safety defaults: confirm that attempting to open a non-existent 
    parquet path gracefully yields None instead of triggering a crashing OS error.
    """
    assert load_or_none(os.path.join(tmp_path, "nope.parquet")) is None


# ---------------------------------------------------------------------------
# null_rate_report
# ---------------------------------------------------------------------------

def test_null_rate_report_only_includes_columns_with_nulls():
    """
    Verify analytical filters: ensure that the null-rate calculator flags missing 
    attributes accurately while completely bypassing fully populated fields.
    """
    df = pd.DataFrame({"clean":[1, 2, 3], "dirty": [1, None, 3]})
    rates = null_rate_report(df)
    assert "clean" not in rates
    assert "dirty" in rates
    assert abs(rates["dirty"] - 33.3) < 0.5


# ---------------------------------------------------------------------------
# record_stage_stats & load_pipeline_stats
# ---------------------------------------------------------------------------

def test_record_and_load_pipeline_stats_roundtrip(tmp_path):
    """
    Verify metrics serialization: check that multi-stage logs write correctly to JSON, 
    append cleanly, and maintain chronological or sorting sequence parameters.
    """
    stats_path = os.path.join(tmp_path, "pipeline_stats.json")
    df = pd.DataFrame({"x":[1, 2], "y": [3, None]})
    
    record_stage_stats("1_unit_test_stage", df, extra={"k": "v"}, stats_path=stats_path)
    history = load_pipeline_stats(stats_path)
    assert len(history) == 1
    assert history[0]["stage"] == "1_unit_test_stage"
    assert history[0]["n_rows"] == 2
    assert history[0]["extra"] == {"k": "v"}

    record_stage_stats("2_second_stage", df, stats_path=stats_path)
    history = load_pipeline_stats(stats_path)
    assert len(history) == 2
    assert [h["stage"] for h in history] == ["1_unit_test_stage", "2_second_stage"]


def test_record_stage_stats_upserts_rather_than_appends(tmp_path):
    """
    Enforce pipeline tracking transparency: ensure re-running an identical pipeline 
    stage completely overwrites its previous run state instead of leaking duplicate records.
    """
    stats_path = os.path.join(tmp_path, "pipeline_stats.json")
    df_v1 = pd.DataFrame({"x": [1, 2, 3]})
    df_v2 = pd.DataFrame({"x": [1, 2, 3, 4, 5]})

    record_stage_stats("1_fetch", df_v1, stats_path=stats_path)
    record_stage_stats("1_fetch", df_v2, stats_path=stats_path)

    history = load_pipeline_stats(stats_path)
    assert len(history) == 1
    assert history[0]["n_rows"] == 5
