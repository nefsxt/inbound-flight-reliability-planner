"""
Tests for src.hf_storage.

Covers local cache cleanup, route date extraction, manifest generation,

Hugging Face uploads/downloads, remote path parsing, and update checks.

"""

import datetime as dt
import hashlib
import json
import os
import shutil

from unittest.mock import MagicMock

import pytest

import config
import src.hf_storage as hf_storage

from huggingface_hub.utils import RepositoryNotFoundError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_config(monkeypatch):

    """Provide deterministic configuration values for all tests."""

    monkeypatch.setattr(hf_storage.config, "HF_REPO_ID", "test-owner/test-repo")
    monkeypatch.setattr(hf_storage.config, "HF_REPO_TYPE", "dataset")
    monkeypatch.setattr(hf_storage.config, "ROUTES", [("EDDF", "LGTS")])
    monkeypatch.setattr(
        hf_storage.config,
        "route_key",
        lambda origin, destination: f"{origin}_{destination}",
    )


@pytest.fixture
def mock_env(monkeypatch):


    """Provide deterministic environment variables used by hf_storage."""

    monkeypatch.setenv("HF_TOKEN", "mock-token")
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")


@pytest.fixture
def isolated_workspace(tmp_path, monkeypatch):

    """Redirect local data/model directories to a temporary workspace."""

    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"

    monkeypatch.setattr(hf_storage, "LOCAL_DATA_DIR", data_dir)
    monkeypatch.setattr(hf_storage, "LOCAL_MODELS_DIR", models_dir)

    return data_dir, models_dir


# ---------------------------------------------------------------------------
# clear_local_cache
# ---------------------------------------------------------------------------


def test_clear_local_cache_removes_existing_directories(isolated_workspace):

    """Verify that existing data and model directories are removed and recreated."""

    data_dir, models_dir = isolated_workspace
    data_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    (data_dir / "stale_file.parquet").write_text("old data")
    (models_dir / "old_model.onnx").write_text("old model")

    hf_storage.clear_local_cache()

    assert data_dir.exists()
    assert data_dir.is_dir()
    assert list(data_dir.iterdir()) == []
    assert models_dir.exists()
    assert models_dir.is_dir()
    assert list(models_dir.iterdir()) == []


def test_clear_local_cache_creates_missing_directories(isolated_workspace):

    """Verify that missing data and model directories are created."""

    data_dir, models_dir = isolated_workspace

    assert not data_dir.exists()
    assert not models_dir.exists()

    hf_storage.clear_local_cache()

    assert data_dir.exists()
    assert data_dir.is_dir()
    assert models_dir.exists()
    assert models_dir.is_dir()


def test_clear_local_cache_handles_permission_error(isolated_workspace, monkeypatch):

    """Verify that a PermissionError triggers the fallback cleanup logic."""

    data_dir, models_dir = isolated_workspace
    data_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    (data_dir / "stale.parquet").write_text("data")
    (models_dir / "model.onnx").write_text("model")
    original_rmtree = shutil.rmtree

    def failing_rmtree(path):
        if path == data_dir:
            raise PermissionError("permission denied")
        original_rmtree(path)


    monkeypatch.setattr(hf_storage.shutil, "rmtree", failing_rmtree)

    hf_storage.clear_local_cache()

    assert data_dir.exists()
    assert list(data_dir.iterdir()) == []
    assert models_dir.exists()
    assert list(models_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# _route_last_dates
# ---------------------------------------------------------------------------


def test_route_last_dates_returns_iso_dates(monkeypatch):

    """Verify that flight and weather dates are returned as ISO strings."""

    monkeypatch.setattr(
        "src.fetch_opensky.get_last_flight_date",
        lambda origin, destination: dt.date(2026, 9, 1),
    )
    monkeypatch.setattr(
        "src.fetch_weather.get_last_weather_date",
        lambda origin, destination: dt.date(2026, 9, 5),
    )

    result = hf_storage._route_last_dates()

    assert result == {
        "EDDF_LGTS": {
            "last_flight_date": "2026-09-01",
            "last_weather_date": "2026-09-05",
        }
    }


def test_route_last_dates_handles_missing_dates(monkeypatch):

    """Verify that missing flight or weather dates are stored as None."""

    monkeypatch.setattr(
        "src.fetch_opensky.get_last_flight_date",
        lambda origin, destination: None,
    )
    monkeypatch.setattr(
        "src.fetch_weather.get_last_weather_date",
        lambda origin, destination: None,
    )

    result = hf_storage._route_last_dates()

    assert result == {
        "EDDF_LGTS": {
            "last_flight_date": None,
            "last_weather_date": None,
        }
    }


# ---------------------------------------------------------------------------
# _generate_and_save_manifest
# ---------------------------------------------------------------------------


def test_generate_and_save_manifest_creates_manifest(isolated_workspace, mock_env, monkeypatch):

    """Verify that the manifest contains run, route, and file information."""

    data_dir, _ = isolated_workspace

    monkeypatch.setattr(
        hf_storage,
        "_route_last_dates",
        lambda: {
            "EDDF_LGTS": {
                "last_flight_date": "2026-09-01",
                "last_weather_date": "2026-09-05",
            }
        },
    )

    raw_file = (
        data_dir
        / "raw"
        / "EDDF_LGTS"
        / "arrivals"
        / "arrivals_EDDF_2026-09-01_2026-09-07.parquet"
    )
    raw_file.parent.mkdir(parents=True)
    raw_file.write_bytes(b"test parquet data")

    hf_storage._generate_and_save_manifest()
    manifest_path = data_dir / "manifest.json"

    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())

    assert manifest["github_run_id"] == "12345"
    assert manifest["routes"]["EDDF_LGTS"] == {
        "last_flight_date": "2026-09-01",
        "last_weather_date": "2026-09-05",
    }

    repo_path = (
        "data/raw/EDDF_LGTS/arrivals/"
        "arrivals_EDDF_2026-09-01_2026-09-07.parquet"
    )

    assert repo_path in manifest["files"]


def test_generate_and_save_manifest_records_file_hash_and_size(isolated_workspace, mock_env, monkeypatch):

    """Verify that file size and SHA256 hash are recorded correctly."""

    data_dir, _ = isolated_workspace

    monkeypatch.setattr(hf_storage, "_route_last_dates", lambda: {})

    file_content = b"known test content"
    test_file = data_dir / "raw" / "test.parquet"
    test_file.parent.mkdir(parents=True)
    test_file.write_bytes(file_content)

    expected_hash = hashlib.sha256(file_content).hexdigest()

    hf_storage._generate_and_save_manifest()
    manifest = json.loads((data_dir / "manifest.json").read_text())

    file_info = manifest["files"]["data/raw/test.parquet"]

    assert file_info["size_bytes"] == len(file_content)
    assert file_info["sha256"] == expected_hash


def test_generate_and_save_manifest_excludes_manifest_file(isolated_workspace, mock_env, monkeypatch):

    """Verify that manifest.json is not included in its own file listing."""

    data_dir, _ = isolated_workspace

    monkeypatch.setattr(hf_storage, "_route_last_dates", lambda: {})
    data_dir.mkdir(parents=True)
    (data_dir / "manifest.json").write_text("old manifest")
    (data_dir / "data.parquet").write_text("data")

    hf_storage._generate_and_save_manifest()
    manifest = json.loads((data_dir / "manifest.json").read_text())

    assert "data/data.parquet" in manifest["files"]
    assert "data/manifest.json" not in manifest["files"]


def test_generate_and_save_manifest_includes_model_files(isolated_workspace, mock_env, monkeypatch):

    """Verify that files under models are recorded with the models prefix."""

    data_dir, models_dir = isolated_workspace

    monkeypatch.setattr(hf_storage, "_route_last_dates", lambda: {})

    model_file = models_dir / "classifier" / "model.onnx"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"model data")

    hf_storage._generate_and_save_manifest()
    manifest = json.loads((data_dir / "manifest.json").read_text())

    assert "models/classifier/model.onnx" in manifest["files"]


# ---------------------------------------------------------------------------
# upload_to_hf
# ---------------------------------------------------------------------------


def test_upload_to_hf_requires_hf_token(isolated_workspace, monkeypatch):

    """Verify that upload_to_hf raises ValueError when HF_TOKEN is missing."""

    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(
        ValueError,
        match="HF_TOKEN environment variable is missing.",
    ):
        hf_storage.upload_to_hf()


def test_upload_to_hf_uploads_data_and_models(isolated_workspace, mock_env, monkeypatch):

    """Verify that both data and models are uploaded when requested."""

    data_dir, models_dir = isolated_workspace
    data_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    mock_api = MagicMock()

    monkeypatch.setattr(
        hf_storage,
        "_generate_and_save_manifest",
        lambda: None,
    )
    monkeypatch.setattr(
        hf_storage,
        "HfApi",
        lambda token: mock_api,
    )

    hf_storage.upload_to_hf(include_models=True)

    assert mock_api.upload_folder.call_count == 2

    calls = mock_api.upload_folder.call_args_list

    assert calls[0].kwargs["repo_id"] == "test-owner/test-repo"
    assert calls[0].kwargs["repo_type"] == "dataset"
    assert calls[0].kwargs["folder_path"] == str(data_dir)
    assert calls[0].kwargs["path_in_repo"] == "data"
    assert calls[0].kwargs["commit_message"] == "Update project data"

    assert calls[1].kwargs["folder_path"] == str(models_dir)
    assert calls[1].kwargs["path_in_repo"] == "models"
    assert calls[1].kwargs["commit_message"] == "Update project models"


def test_upload_to_hf_skips_models_when_include_models_false(isolated_workspace, mock_env, monkeypatch):

    """Verify that only data is uploaded when include_models is False."""

    data_dir, models_dir = isolated_workspace
    data_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    mock_api = MagicMock()

    monkeypatch.setattr(
        hf_storage,
        "_generate_and_save_manifest",
        lambda: None,
    )
    monkeypatch.setattr(
        hf_storage,
        "HfApi",
        lambda token: mock_api,
    )

    hf_storage.upload_to_hf(include_models=False)

    assert mock_api.upload_folder.call_count == 1

    call = mock_api.upload_folder.call_args

    assert call.kwargs["folder_path"] == str(data_dir)
    assert call.kwargs["path_in_repo"] == "data"
    assert call.kwargs["commit_message"] == "Update project data"


def test_upload_to_hf_skips_missing_models_directory(isolated_workspace, mock_env, monkeypatch):
    
    """Verify that a missing models directory does not cause a model upload."""

    data_dir, _ = isolated_workspace
    data_dir.mkdir(parents=True)

    mock_api = MagicMock()

    monkeypatch.setattr(
        hf_storage,
        "_generate_and_save_manifest",
        lambda: None,
    )
    monkeypatch.setattr(
        hf_storage,
        "HfApi",
        lambda token: mock_api,
    )

    hf_storage.upload_to_hf(include_models=True)

    assert mock_api.upload_folder.call_count == 1
    assert mock_api.upload_folder.call_args.kwargs["path_in_repo"] == "data"


# ---------------------------------------------------------------------------
# _list_remote_paths
# ---------------------------------------------------------------------------


def test_list_remote_paths_returns_remote_paths(mock_env, monkeypatch):

    """Verify that repository entries are converted to their path strings."""

    mock_api = MagicMock()
    mock_api.list_repo_tree.return_value = [
        MagicMock(path="data/file1.parquet"),
        MagicMock(path="data/file2.parquet"),
        MagicMock(path="models/model.onnx"),
    ]

    monkeypatch.setattr(
        hf_storage,
        "HfApi",
        lambda token: mock_api,
    )

    result = hf_storage._list_remote_paths()

    assert result == [
        "data/file1.parquet",
        "data/file2.parquet",
        "models/model.onnx",
    ]

    mock_api.list_repo_tree.assert_called_once_with(
        repo_id="test-owner/test-repo",
        repo_type="dataset",
        recursive=True,
    )


def test_list_remote_paths_returns_empty_list_when_repo_missing(monkeypatch, mock_env):

    """Verify that a missing Hugging Face repository returns an empty list."""

    mock_api = MagicMock()
    response = MagicMock()
    response.status_code = 404
    response.reason = "Not Found"

    mock_api.list_repo_tree.side_effect = RepositoryNotFoundError(
        "repository not found",
        response=response,
    )

    monkeypatch.setattr(hf_storage, "HfApi", lambda token: mock_api)

    assert hf_storage._list_remote_paths() == []


def test_list_remote_paths_returns_empty_list_on_unexpected_error(mock_env, monkeypatch):

    """Verify that unexpected repository errors return an empty list."""

    mock_api = MagicMock()
    mock_api.list_repo_tree.side_effect = RuntimeError("network error")

    monkeypatch.setattr(
        hf_storage,
        "HfApi",
        lambda token: mock_api,
    )

    result = hf_storage._list_remote_paths()

    assert result == []


# ---------------------------------------------------------------------------
# get_remote_route_dates
# ---------------------------------------------------------------------------


def test_get_remote_route_dates_returns_latest_flight_and_weather_dates(monkeypatch):

    """Verify that the latest matching flight and weather dates are selected."""

    monkeypatch.setattr(
        hf_storage,
        "_list_remote_paths",
        lambda: [
            (
                "data/raw/EDDF_LGTS/arrivals/"
                "arrivals_EDDF_2026-09-01_2026-09-05.parquet"
            ),
            (
                "data/raw/EDDF_LGTS/arrivals/"
                "arrivals_EDDF_2026-09-01_2026-09-07.parquet"
            ),
            (
                "data/raw/weather_hist_chunks/EDDF/"
                "EDDF_2026-09-01_2026-09-06.parquet"
            ),
            (
                "data/raw/weather_hist_chunks/LGTS/"
                "LGTS_2026-09-01_2026-09-05.parquet"
            ),
        ],
    )

    result = hf_storage.get_remote_route_dates()

    assert result == {
        "EDDF_LGTS": {
            "last_flight_date": "2026-09-07",
            "last_weather_date": "2026-09-06",
        }
    }


def test_get_remote_route_dates_uses_destination_weather_when_origin_missing(monkeypatch,):

    """Verify that destination weather is used when origin weather is missing."""

    monkeypatch.setattr(
        hf_storage,
        "_list_remote_paths",
        lambda: [
            (
                "data/raw/EDDF_LGTS/arrivals/"
                "arrivals_EDDF_2026-09-01_2026-09-07.parquet"
            ),
            (
                "data/raw/weather_hist_chunks/LGTS/"
                "LGTS_2026-09-01_2026-09-06.parquet"
            ),
        ],
    )

    result = hf_storage.get_remote_route_dates()

    assert result["EDDF_LGTS"]["last_flight_date"] == "2026-09-07"
    assert result["EDDF_LGTS"]["last_weather_date"] == "2026-09-06"


def test_get_remote_route_dates_returns_none_when_no_matching_files(monkeypatch,):

    """Verify that missing remote data results in None dates."""

    monkeypatch.setattr(
        hf_storage,
        "_list_remote_paths",
        lambda: [
            "data/some_other_file.parquet",
            "models/model.onnx",
        ],
    )

    result = hf_storage.get_remote_route_dates()

    assert result == {
        "EDDF_LGTS": {
            "last_flight_date": None,
            "last_weather_date": None,
        }
    }


def test_get_remote_route_dates_normalizes_backslashes(monkeypatch):

    """Verify that Windows-style remote paths are parsed correctly."""

    monkeypatch.setattr(
        hf_storage,
        "_list_remote_paths",
        lambda: [
            (
                r"data\raw\EDDF_LGTS\arrivals"
                r"\arrivals_EDDF_2026-09-01_2026-09-07.parquet"
            ),
            (
                r"data\raw\weather_hist_chunks\EDDF"
                r"\EDDF_2026-09-01_2026-09-06.parquet"
            ),
        ],
    )

    result = hf_storage.get_remote_route_dates()

    assert result["EDDF_LGTS"] == {
        "last_flight_date": "2026-09-07",
        "last_weather_date": "2026-09-06",
    }


# ---------------------------------------------------------------------------
# routes_need_update
# ---------------------------------------------------------------------------


def test_routes_need_update_returns_false_when_data_is_current(monkeypatch):

    """Verify that no update is required when both datasets reach yesterday."""

    monkeypatch.setattr(
        hf_storage,
        "get_remote_route_dates",
        lambda: {
            "EDDF_LGTS": {
                "last_flight_date": "2026-09-05",
                "last_weather_date": "2026-09-05",
            }
        },
    )

    assert hf_storage.routes_need_update(as_of="2026-09-06") is False


def test_routes_need_update_returns_true_when_flight_data_is_old(monkeypatch):

    """Verify that an old flight date requires an update."""

    monkeypatch.setattr(
        hf_storage,
        "get_remote_route_dates",
        lambda: {
            "EDDF_LGTS": {
                "last_flight_date": "2026-09-04",
                "last_weather_date": "2026-09-05",
            }
        },
    )

    assert hf_storage.routes_need_update(as_of="2026-09-06") is True


def test_routes_need_update_returns_true_when_weather_data_is_old(monkeypatch):

    """Verify that an old weather date requires an update."""

    monkeypatch.setattr(
        hf_storage,
        "get_remote_route_dates",
        lambda: {
            "EDDF_LGTS": {
                "last_flight_date": "2026-09-05",
                "last_weather_date": "2026-09-04",
            }
        },
    )

    assert hf_storage.routes_need_update(as_of="2026-09-06") is True


def test_routes_need_update_returns_true_when_route_is_missing(monkeypatch):

    """Verify that a configured route missing remotely requires an update."""

    monkeypatch.setattr(
        hf_storage,
        "get_remote_route_dates",
        lambda: {},
    )

    assert hf_storage.routes_need_update(as_of="2026-09-06") is True


def test_routes_need_update_returns_true_when_flight_date_is_missing(monkeypatch,):

    """Verify that a missing flight date requires an update."""

    monkeypatch.setattr(
        hf_storage,
        "get_remote_route_dates",
        lambda: {
            "EDDF_LGTS": {
                "last_flight_date": None,
                "last_weather_date": "2026-09-05",
            }
        },
    )

    assert hf_storage.routes_need_update(as_of="2026-09-06") is True


def test_routes_need_update_returns_true_when_weather_date_is_missing(monkeypatch,):

    """Verify that a missing weather date requires an update."""

    monkeypatch.setattr(
        hf_storage,
        "get_remote_route_dates",
        lambda: {
            "EDDF_LGTS": {
                "last_flight_date": "2026-09-05",
                "last_weather_date": None,
            }
        },
    )

    assert hf_storage.routes_need_update(as_of="2026-09-06") is True


def test_routes_need_update_accepts_data_exactly_from_yesterday(monkeypatch):

    """Verify that data dated exactly yesterday is considered current."""

    monkeypatch.setattr(
        hf_storage,
        "get_remote_route_dates",
        lambda: {
            "EDDF_LGTS": {
                "last_flight_date": "2026-09-06",
                "last_weather_date": "2026-09-06",
            }
        },
    )

    assert hf_storage.routes_need_update(as_of="2026-09-07") is False


# ---------------------------------------------------------------------------
# download_from_hf
# ---------------------------------------------------------------------------


def test_download_from_hf_downloads_to_requested_directory(mock_env, tmp_path, monkeypatch):

    """Verify that snapshot_download receives the configured repository and directory."""

    local_dir = tmp_path / "download"

    mock_snapshot_download = MagicMock(return_value=str(local_dir))

    monkeypatch.setattr(
        hf_storage,
        "snapshot_download",
        mock_snapshot_download,
    )

    result = hf_storage.download_from_hf(local_dir=str(local_dir))

    assert result == str(local_dir)

    mock_snapshot_download.assert_called_once_with(
        repo_id="test-owner/test-repo",
        repo_type="dataset",
        local_dir=str(local_dir),
        token="mock-token",
    )


def test_download_from_hf_warns_when_downloading_to_data(mock_env, monkeypatch, capsys):

    """Verify that downloading directly into data prints the nested-path warning."""

    mock_snapshot_download = MagicMock(return_value="data")

    monkeypatch.setattr(
        hf_storage,
        "snapshot_download",
        mock_snapshot_download,
    )

    result = hf_storage.download_from_hf(local_dir="data")

    assert result == "data"

    captured = capsys.readouterr()

    assert (
        "[WARNING]: Downloading directly into 'data' "
        "will likely create a nested 'data/data/' path."
        in captured.out
    )


def test_download_from_hf_warns_when_downloading_to_models(mock_env, monkeypatch, capsys):

    """Verify that downloading directly into models prints the nested-path warning."""

    mock_snapshot_download = MagicMock(return_value="models")

    monkeypatch.setattr(
        hf_storage,
        "snapshot_download",
        mock_snapshot_download,
    )

    result = hf_storage.download_from_hf(local_dir="models")

    assert result == "models"

    captured = capsys.readouterr()

    assert (
        "[WARNING]: Downloading directly into 'models' "
        "will likely create a nested 'models/models/' path."
        in captured.out
    )


def test_download_from_hf_does_not_warn_for_custom_directory(mock_env, tmp_path, monkeypatch, capsys):

    """Verify that a custom download directory does not produce a warning."""

    local_dir = tmp_path / "hf_download"

    mock_snapshot_download = MagicMock(return_value=str(local_dir))

    monkeypatch.setattr(
        hf_storage,
        "snapshot_download",
        mock_snapshot_download,
    )

    hf_storage.download_from_hf(local_dir=str(local_dir))

    captured = capsys.readouterr()

    assert "[WARNING]" not in captured.out



# ---------------------------------------------------------------------------
# download_inference_artifacts
# ---------------------------------------------------------------------------


def test_download_inference_artifacts_uses_correct_patterns(tmp_path, monkeypatch,):
    """
    Verify that inference downloads are restricted to features.parquet
    and trained model files, excluding raw and intermediate data.
    """
    calls = {}

    monkeypatch.setenv("HF_TOKEN", "test-hf-token")

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(
        hf_storage,
        "snapshot_download",
        fake_snapshot_download,
    )

    result = hf_storage.download_inference_artifacts(
        local_dir=str(tmp_path)
    )

    assert calls["allow_patterns"] == [
        "data/processed/features.parquet",
        "models/**",
    ]
    assert calls["local_dir"] == str(tmp_path)
    assert result == str(tmp_path)


def test_download_inference_artifacts_uses_config(tmp_path, monkeypatch,):
    """
    Verify that inference downloads use the Hugging Face repository ID
    and repository type defined in the application configuration.
    """
    calls = {}

    monkeypatch.setenv("HF_TOKEN", "test-hf-token")

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(
        hf_storage,
        "snapshot_download",
        fake_snapshot_download,
    )

    hf_storage.download_inference_artifacts(
        local_dir=str(tmp_path)
    )

    assert calls["repo_id"] == config.HF_REPO_ID
    assert calls["repo_type"] == config.HF_REPO_TYPE


def test_download_inference_artifacts_passes_token(tmp_path,monkeypatch,):
    """
    Verify that HF_TOKEN is passed to snapshot_download so the private
    Hugging Face repository can be accessed.
    """
    token = "test-hf-token"
    calls = {}

    monkeypatch.setenv("HF_TOKEN", token)

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(
        hf_storage,
        "snapshot_download",
        fake_snapshot_download,
    )

    hf_storage.download_inference_artifacts(
        local_dir=str(tmp_path)
    )

    assert calls["token"] == token


def test_download_inference_artifacts_requires_token(monkeypatch,):
    """
    Verify that the inference download raises a RuntimeError when
    HF_TOKEN is not configured.
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        hf_storage.download_inference_artifacts()


def test_full_download_remains_unrestricted(tmp_path, monkeypatch,):
    """
    Verify that the original full repository download remains
    unrestricted so it can still be used for training and feature
    reconstruction.
    """
    calls = {}

    monkeypatch.setenv("HF_TOKEN", "test-hf-token")

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(
        hf_storage,
        "snapshot_download",
        fake_snapshot_download,
    )

    hf_storage.download_from_hf(
        local_dir=str(tmp_path)
    )

    assert "allow_patterns" not in calls