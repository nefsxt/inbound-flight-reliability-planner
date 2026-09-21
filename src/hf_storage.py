"""
Synchronize local flight data, weather data, and model assets with Hugging Face.

Supports uploading and downloading repository assets, checking whether
configured routes require data updates, generating file manifests with
SHA-256 checksums and file sizes, and clearing local caches.

Usage:
    python sync_assets.py --upload
    python sync_assets.py --download
    python sync_assets.py --check
"""


from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path
import shutil

import config
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

LOCAL_DATA_DIR = Path("data")
LOCAL_MODELS_DIR = Path("models")


_ARRIVALS_CHUNK_RE = re.compile(
    r"^data/raw/(?P<route_key>[A-Z0-9_]+)/arrivals/arrivals_[A-Z0-9]+_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})\.parquet$"
)
_WEATHER_CHUNK_RE = re.compile(
    r"^data/raw/weather_hist_chunks/(?P<airport>[A-Z0-9]+)/[A-Z0-9]+_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})\.parquet$"
)


def _route_last_dates() -> dict:

    """Read each configured route's last stored date locally."""

    from src.fetch_opensky import get_last_flight_date
    from src.fetch_weather import get_last_weather_date

    routes = {}
    for origin, destination in config.ROUTES:
        key = config.route_key(origin, destination)
        last_flight = get_last_flight_date(origin, destination)
        last_weather = get_last_weather_date(origin, destination)

        routes[key] = {
            "last_flight_date": last_flight.isoformat() if last_flight else None,
            "last_weather_date": last_weather.isoformat() if last_weather else None,
        }
    return routes


def _generate_and_save_manifest() -> None:

    """Build and save manifest.json based on local data and model files."""

    #Ensure dir exists BEFORE writing manifest to avoid FileNotFoundError
    LOCAL_DATA_DIR.mkdir(exist_ok=True, parents=True)

    manifest = {
        "last_updated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", "local-run"),
        "routes": _route_last_dates(),
        "files": {},
    }

    target_dirs = [(LOCAL_DATA_DIR, "data"), (LOCAL_MODELS_DIR, "models")]

    for directory, prefix in target_dirs:
        if not directory.exists():
            continue

        for path in directory.rglob("*"):
            if path.is_file() and path.name != "manifest.json":
                repo_path = f"{prefix}/{path.relative_to(directory).as_posix()}"
                sha256_hash = hashlib.sha256()

                with open(path, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(byte_block)

                manifest["files"][repo_path] = {
                    "sha256": sha256_hash.hexdigest(),
                    "size_bytes": path.stat().st_size,
                }

    with open(LOCAL_DATA_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def upload_to_hf(include_models: bool = True) -> None:

    """Upload local data and optionally models to the HF dataset repo."""

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is missing.")

    _generate_and_save_manifest()
    api = HfApi(token=token)

    
    api.upload_folder(
        folder_path=str(LOCAL_DATA_DIR),
        path_in_repo="data",
        repo_id=config.HF_REPO_ID,
        repo_type=config.HF_REPO_TYPE,
        commit_message="Update project data",
    )

    if include_models and LOCAL_MODELS_DIR.exists():
        api.upload_folder(
            folder_path=str(LOCAL_MODELS_DIR),
            path_in_repo="models",
            repo_id=config.HF_REPO_ID,
            repo_type=config.HF_REPO_TYPE,
            commit_message="Update project models",
        )


def _list_remote_paths() -> list[str]:

    """Fetch all file paths currently in the HF dataset repo."""

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    try:
        return [
            entry.path
            for entry in api.list_repo_tree(
                repo_id=config.HF_REPO_ID,
                repo_type=config.HF_REPO_TYPE,
                recursive=True,
            )
        ]
    except RepositoryNotFoundError:
        return []
    except Exception:
        return []


def get_remote_route_dates() -> dict:
    """Discovers remote max dates efficiently by parsing the path list ONCE."""
    paths = _list_remote_paths()
    
    remote_flights = {}  # route_key -> max_date
    remote_weather = {}  # airport -> max_date

    # Parse full paths directly matching across the list in O(N) linear time
    for path in paths:
        # Standardize matching to forward slashes for cross-platform safety
        clean_path = path.replace("\\", "/")
        
        flight_match = _ARRIVALS_CHUNK_RE.match(clean_path)
        if flight_match:
            route_key = flight_match.group("route_key")
            end_date = dt.date.fromisoformat(flight_match.group("end"))
            remote_flights[route_key] = max(remote_flights.get(route_key, end_date), end_date)
            continue

        weather_match = _WEATHER_CHUNK_RE.match(clean_path)
        if weather_match:
            airport = weather_match.group("airport")
            end_date = dt.date.fromisoformat(weather_match.group("end"))
            remote_weather[airport] = max(remote_weather.get(airport, end_date), end_date)

    routes = {}
    for origin, destination in config.ROUTES:
        key = config.route_key(origin, destination)
        
        last_flight = remote_flights.get(key)
        
        w_orig = remote_weather.get(origin)
        w_dest = remote_weather.get(destination)
        
        if w_orig and w_dest:
            last_weather = max(w_orig, w_dest)
        else:
            last_weather = w_orig or w_dest

        routes[key] = {
            "last_flight_date": last_flight.isoformat() if last_flight else None,
            "last_weather_date": last_weather.isoformat() if last_weather else None,
        }

    return routes


def routes_need_update(as_of: str | None = None) -> bool:
    """True if data is older than yesterday. Uniform dt syntax used."""
    today = dt.date.fromisoformat(as_of) if as_of else dt.date.today()
    yesterday = today - dt.timedelta(days=1)

    remote_dates = get_remote_route_dates()

    for origin, destination in config.ROUTES:
        key = config.route_key(origin, destination)
        entry = remote_dates.get(key)

        if not entry:
            return True

        for field in ("last_flight_date", "last_weather_date"):
            value = entry.get(field)
            if not value or dt.date.fromisoformat(value) < yesterday:
                return True
    return False


def download_from_hf(local_dir: str = ".") -> str:
    """Download the current HF dataset snapshot safely."""
    token = os.environ.get("HF_TOKEN")
    
    # Catch unintended local nesting if a user mistakenly runs script with '--dir data'
    if local_dir in ("data", "models"):
        print(f"[WARNING]: Downloading directly into '{local_dir}' will likely create a nested '{local_dir}/{local_dir}/' path.")
        
    return snapshot_download(
        repo_id=config.HF_REPO_ID,
        repo_type=config.HF_REPO_TYPE,
        local_dir=local_dir,
        token=token,
    )


def download_inference_artifacts(local_dir: str = ".") -> str:
    """
    Download only the processed features and trained models
    required for inference.
    """
    token = os.environ.get("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Cannot download from Hugging Face."
        )

    return snapshot_download(
        repo_id=config.HF_REPO_ID,
        repo_type=config.HF_REPO_TYPE,
        local_dir=local_dir,
        token=token,
        allow_patterns=[
            "data/processed/features.parquet",
            "models/**",
        ],
    )

def clear_local_cache() -> None:
    """Wipes out local data/ and models/ directories to guarantee a fresh state."""
    for directory in (LOCAL_DATA_DIR, LOCAL_MODELS_DIR):
        if directory.exists():
            print(f"[CLEANUP] Removing existing local directory: {directory}")
            try:
                shutil.rmtree(directory)
            except PermissionError:
                # Common issue in Docker when containers run as root vs host users
                print(f"[WARNING] Permission denied deleting {directory}. Attempting folder contents purge instead.")
                for path in directory.glob("*"):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path)
                        
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sync assets with Hugging Face.")

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--upload", 
        action="store_true"
        )

    group.add_argument(
        "--download", 
        action="store_true"
        )

    group.add_argument(
        "--check", 
        action="store_true"
        )
    
    parser.add_argument(
        "--data-only", 
        action="store_true"
        )

    parser.add_argument(
        "--dir", 
        type=str, 
        default="." # Defaults to active repository root directory
        )  

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Purge local data/ and models/ folders before performing execution blocks.",
    )

    args = parser.parse_args()

    # Process cleanup execution before downloads or evaluations happen
    if args.clean:
        clear_local_cache()

    if args.upload:
        upload_to_hf(include_models=not args.data_only)
    elif args.download:
        download_from_hf(local_dir=args.dir)
    elif args.check:
        needed = routes_need_update()
        print(f"needs_update={'true' if needed else 'false'}")
