"""
Small shared helpers used across the fetch/feature scripts.
"""
import os
import csv
import io
import json
import time
import glob
import secrets
import datetime as dt
from pathlib import Path

import pandas as pd


def daterange_chunks(start_date: str, end_date: str, chunk_days: int):

    """
    Yield (chunk_start, chunk_end) datetime.date pairs covering
    [start_date, end_date] inclusive, in chunk_days-sized windows.
    Dates are 'YYYY-MM-DD' strings.
    """
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        yield cur, chunk_end
        cur = chunk_end + dt.timedelta(days=1)


def to_unix(d: dt.date, hour: int = 0, minute: int = 0) -> int:

    """UTC midnight (or given hour/min) of a date, as a unix timestamp."""

    dtobj = dt.datetime(d.year, d.month, d.day, hour, minute, tzinfo=dt.timezone.utc)
    return int(dtobj.timestamp())


def cache_path(raw_dir: str, prefix: str, key: str) -> str:

    """Sanitize a unique lookup key and build its target parquet file path."""

    safe_key = key.replace("/", "-").replace(":", "-")
    return os.path.join(raw_dir, f"{prefix}_{safe_key}.parquet")


def load_or_none(path: str):

    """Safely attempt to load a parquet file, returning None if any error occurs."""

    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


#def save_parquet(df: pd.DataFrame, path: str):
#    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
#    df.to_parquet(path, index=False)

# atomic writes version 
def save_parquet(df: pd.DataFrame, path: str):

    """
    Saves a pandas DataFrame to a Parquet file atomically on the local device.
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    random_suffix = secrets.token_hex(4)
    temp_path = target_path.with_suffix(f".tmp_{random_suffix}")

    try:
        with open(temp_path, "wb") as f:
            df.to_parquet(f, index=False)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, target_path)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def log_credit_usage(raw_dir: str, endpoint: str, span_days: float, note: str = ""):

    """
    Append a row to a running credit-usage log so you can eyeball roughly
    how much of your daily OpenSky budget you've spent. This is a rough
    self-tracking aid, not an authoritative credit count from OpenSky.
    """
    log_path = os.path.join(raw_dir, "opensky_credit_log.csv")

    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    is_new = not os.path.exists(log_path)

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp_utc", "endpoint", "span_days", "note"])
        writer.writerow([dt.datetime.utcnow().isoformat(), endpoint, span_days, note])


def polite_sleep(seconds: float):

    """Pause execution temporarily to respect API rate limits and avoid server spam."""

    time.sleep(seconds)


# ---------------------------------------------------------------------------
# Pipeline Observability
#
# Every fetch, feature, and training script appends a runtime snapshot to
# `data/processed/pipeline_stats.json` via `record_stage_stats()`.
#
# To view this data:
#   1. Run `python -m src.diagnostics` in your terminal for a clean text report.
#   2. Check the "Pipeline Explorer" tab in the Streamlit app interface.
# ---------------------------------------------------------------------------

PIPELINE_STATS_PATH = "data/processed/pipeline_stats.json"


def null_rate_report(df: pd.DataFrame) -> dict:

    """% missing per column, rounded, for columns that have any nulls."""

    if df is None or df.empty:
        return {}
    rates = (df.isna().mean() * 100).round(1)
    return {col: float(pct) for col, pct in rates.items() if pct > 0}


def record_stage_stats(stage: str, df: pd.DataFrame, extra: dict = None,
                        stats_path: str = PIPELINE_STATS_PATH):

    """
    Record a snapshot of {stage, timestamp, row/col counts, null rates,
    sample rows, ...extra} to a running JSON log, keyed by `stage` name.

    This is an UPSERT, not an append: if `stage` already has an entry (from
    an earlier run of the same script), that entry is replaced rather than
    piling up a duplicate. Re-running `python -m src.features` five times
    in a row should show the pipeline's CURRENT shape, not five stale
    snapshots mixed in with whatever the pipeline used to look like before
    a refactor. Entries are always written back sorted by stage name, so
    give stages a numeric prefix (e.g. "1_fetch_opensky", "2_fetch_weather")
    if you want the log to read in pipeline order.
    """
    Path(os.path.dirname(stats_path)).mkdir(parents=True, exist_ok=True)

    entry = {
        "stage": stage,
        "timestamp_utc": dt.datetime.utcnow().isoformat(),
        "n_rows": 0 if df is None else int(len(df)),
        "n_cols": 0 if df is None else int(len(df.columns)),
        "columns": [] if df is None else list(df.columns),
        "null_rates_pct": null_rate_report(df) if df is not None else {},
        "sample_rows": [] if df is None or df.empty else
            df.head(3).astype(str).to_dict(orient="records"),
    }
    if extra:
        entry["extra"] = extra

    history = []
    if os.path.exists(stats_path):
        try:
            with open(stats_path) as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            history = []

    history = [e for e in history if e.get("stage") != stage]
    history.append(entry)
    history.sort(key=lambda e: e.get("stage", ""))

    with open(stats_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    print(f"[STAGE] {stage}: {entry['n_rows']} rows, {entry['n_cols']} cols"
          + (f", nulls in {len(entry['null_rates_pct'])} col(s)" if entry['null_rates_pct'] else ""))


def load_pipeline_stats(stats_path: str = PIPELINE_STATS_PATH) -> list:

    """
    Load the historical execution metrics and row logs for all pipeline stages.
    
    Returns an empty list if no logging file has been initialized yet.
    """

    if not os.path.exists(stats_path):
        return []
    with open(stats_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Offline Data Inspection Helpers
#
# Used by `inspect_*()` functions across all `fetch_*.py` modules to profile
# cached data assets on disk (shapes, dtypes, nulls, and numeric summaries).
# 
# These tools never hit the network. They read directly from `data/raw/` 
# and produce explicit warnings if the expected files are missing.
# ---------------------------------------------------------------------------


def describe_dataframe(df: pd.DataFrame, label: str = "", n_sample: int = 5):

    """
    Print a full, human-readable inspection of a DataFrame: shape, dtypes
    (via df.info()), null rates per column, a sample of rows, and a numeric
    summary (describe()). Meant to be the one place you look to actually
    *see* data pulled from an external source, rather than trusting it blind.
    """
    header = label or "DataFrame"
    print(f"\n{'=' * 70}\n{header}\n{'=' * 70}")
    if df is None:
        print("  <None> -- nothing to show.")
        return
    if df.empty:
        print(f"  Empty DataFrame (0 rows). Columns: {list(df.columns)}")
        return

    print(f"shape: {df.shape[0]} rows x {df.shape[1]} cols")

    print("\n--- df.info() ---")
    buf = io.StringIO()
    df.info(buf=buf)
    print(buf.getvalue())

    nulls = null_rate_report(df)
    if nulls:
        print("--- columns with missing data (%) ---")
        for col, pct in sorted(nulls.items(), key=lambda kv: -kv[1]):
            print(f"  {col:<30} {pct:>5.1f}%")
    else:
        print("--- no missing data in any column ---")

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(f"\n--- sample rows (first {min(n_sample, len(df))} of {len(df)}) ---")
        print(df.head(n_sample))

        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            print("\n--- numeric summary (describe) ---")
            print(numeric.describe().T)


def inspect_cached_parquet(path: str, label: str = None, n_sample: int = 5):

    """
    Load and fully inspect a cached parquet file IF it exists on disk;
    otherwise say clearly that it hasn't been fetched yet instead of
    raising. Safe to call at any time (e.g. before deciding whether to
    spend API credits re-fetching something).

    Returns the loaded DataFrame, or None if the file doesn't exist / can't
    be read.
    """
    label = label or path
    if not os.path.exists(path):
        print(f"\n[NOT CACHED] {label}\n  -> {path} does not exist yet. "
              f"Run the corresponding fetch step to create it.")
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"\n[ERROR] Could not read {path}: {e}")
        return None
    describe_dataframe(df, label=f"{label}\n(file: {path})", n_sample=n_sample)
    return df


def inspect_cached_glob(raw_dir: str, pattern: str, label: str = "", n_sample: int = 5, combine: bool = True):

    """
    Inspect every cached parquet file matching `pattern` (glob syntax,
    relative to raw_dir) -- e.g. "arrivals_*.parquet". Prints how many files
    are cached, how many are empty (e.g. a chunk where the API legitimately
    returned no rows), and -- if `combine` -- concatenates the non-empty
    ones and runs describe_dataframe() on the combined result so you can
    see the whole picture at once.

    Returns the combined DataFrame (possibly empty) so callers can keep
    using the data programmatically too.
    """
    header = label or pattern
    files = sorted(glob.glob(os.path.join(raw_dir, pattern)))
    print(f"\n{'=' * 70}\n{header} -- cached files on disk\n{'=' * 70}")
    if not files:
        print(f"  No files matching '{pattern}' found in {raw_dir}. "
              f"Nothing has been fetched yet.")
        return pd.DataFrame()

    frames, empty_count, unreadable = [], 0, 0
    for f in files:
        try:
            d = pd.read_parquet(f)
        except Exception as e:
            print(f"  [ERROR] could not read {f}: {e}")
            unreadable += 1
            continue
        if d.empty:
            empty_count += 1
        else:
            frames.append(d)

    print(f"  {len(files)} file(s) matched, {empty_count} empty "
          f"(no rows returned for that pull), {unreadable} unreadable, "
          f"{len(frames)} with data.")

    if not combine or not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    describe_dataframe(combined, label=f"{header} -- COMBINED ({len(frames)} non-empty files)",
                        n_sample=n_sample)
    return combined
