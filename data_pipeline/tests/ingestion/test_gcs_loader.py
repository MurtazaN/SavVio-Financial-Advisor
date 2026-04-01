"""
Tests for Data Ingestion — gcs_loader.py.

Covers GCSLoader initialisation, blob download/upload, CSV and JSONL loading
from Google Cloud Storage, and convenience functions for each data source.
"""
import os
import sys
import types
import json
import importlib.util
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path constants  (sys.path set up by conftest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------------
# Stub google.cloud and google.oauth2
# ---------------------------------------------------------------------------
def _stub_google():
    google = types.ModuleType("google")
    google_cloud = types.ModuleType("google.cloud")
    google_cloud_storage = types.ModuleType("google.cloud.storage")
    google_cloud_storage.Client = MagicMock()
    google_oauth2 = types.ModuleType("google.oauth2")
    google_oauth2_sa = types.ModuleType("google.oauth2.service_account")
    google_oauth2_sa.Credentials = MagicMock()

    google.cloud = google_cloud
    google.oauth2 = google_oauth2
    google_cloud.storage = google_cloud_storage
    google_oauth2.service_account = google_oauth2_sa

    sys.modules["google"] = google
    sys.modules["google.cloud"] = google_cloud
    sys.modules["google.cloud.storage"] = google_cloud_storage
    sys.modules["google.oauth2"] = google_oauth2
    sys.modules["google.oauth2.service_account"] = google_oauth2_sa

_stub_google()

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "src", "ingestion", "gcs_loader.py"),
        os.path.join(PROJECT_ROOT, "dags", "src", "gcs_loader.py"),
        os.path.join(PROJECT_ROOT, "src", "ingestion", "gcs_loader.py"),
        os.path.join(PROJECT_ROOT, "gcs_loader.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        spec = importlib.util.spec_from_file_location("gcs_loader", fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gcs_loader"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ImportError("Could not find gcs_loader.py. Searched:\n" + "\n".join(candidates))

M = _load()
GCSLoader = M.GCSLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_loader():
    """Return a GCSLoader with a fully mocked GCS client."""
    with patch("google.cloud.storage.Client") as mock_client_cls:
        loader = GCSLoader()
        loader.client = MagicMock()
    return loader


def _mock_blob(exists=True):
    blob = MagicMock()
    blob.exists.return_value = exists
    return blob


# =============================================================================
# 1) __init__
# =============================================================================

def test_init_with_no_credentials_uses_adc():
    """No credentials path → uses ADC (storage.Client called without creds)."""
    with patch("google.cloud.storage.Client") as mock_cls:
        GCSLoader()
        mock_cls.assert_called_once()


def test_init_with_valid_credentials_path(tmp_path):
    """Valid credentials path → uses service account credentials."""
    creds_file = tmp_path / "creds.json"
    creds_file.write_text("{}")
    with patch("google.cloud.storage.Client") as mock_cls, \
         patch("google.oauth2.service_account.Credentials.from_service_account_file") as mock_creds:
        mock_creds.return_value = MagicMock()
        GCSLoader(credentials_path=str(creds_file))
        mock_creds.assert_called_once_with(str(creds_file))


def test_init_nonexistent_credentials_falls_back_to_adc():
    """Non-existent credentials path → falls back to ADC silently."""
    with patch("google.cloud.storage.Client"):
        loader = GCSLoader(credentials_path="/nonexistent/path.json")
        assert loader.client is not None


# =============================================================================
# 2) download_blob
# =============================================================================

def test_download_blob_success(tmp_path):
    """Successful download returns destination path."""
    loader = _make_loader()
    dest = str(tmp_path / "file.csv")
    blob = _mock_blob(exists=True)
    blob.download_to_filename.side_effect = lambda p: open(p, "w").write("data")
    loader.client.bucket.return_value.blob.return_value = blob

    result = loader.download_blob("my-bucket", "raw/file.csv", dest)
    assert result == dest
    blob.download_to_filename.assert_called_once_with(dest)


def test_download_blob_not_found_raises(tmp_path):
    """Blob that doesn't exist → raises FileNotFoundError."""
    loader = _make_loader()
    blob = _mock_blob(exists=False)
    loader.client.bucket.return_value.blob.return_value = blob

    with pytest.raises(FileNotFoundError, match="not found in bucket"):
        loader.download_blob("bucket", "missing.csv", str(tmp_path / "out.csv"))


def test_download_blob_creates_parent_dirs(tmp_path):
    """download_blob creates parent directories if they don't exist."""
    loader = _make_loader()
    dest = str(tmp_path / "nested" / "dir" / "file.csv")
    blob = _mock_blob(exists=True)
    blob.download_to_filename.side_effect = lambda p: open(p, "w").write("x")
    loader.client.bucket.return_value.blob.return_value = blob

    loader.download_blob("bucket", "raw/file.csv", dest)
    assert os.path.exists(dest)


# =============================================================================
# 3) upload_blob
# =============================================================================

def test_upload_blob_success(tmp_path):
    """Successful upload returns correct GCS URI."""
    loader = _make_loader()
    src = tmp_path / "file.csv"
    src.write_text("a,b\n1,2")
    blob = MagicMock()
    loader.client.bucket.return_value.blob.return_value = blob

    uri = loader.upload_blob("my-bucket", str(src), "processed/file.csv")
    assert uri == "gs://my-bucket/processed/file.csv"
    blob.upload_from_filename.assert_called_once_with(str(src))


def test_upload_blob_missing_source_raises(tmp_path):
    """Missing source file → raises FileNotFoundError."""
    loader = _make_loader()
    with pytest.raises(FileNotFoundError, match="Source file not found"):
        loader.upload_blob("bucket", "/nonexistent/file.csv", "dest.csv")


def test_upload_blob_sets_content_type(tmp_path):
    """content_type is assigned to blob when provided."""
    loader = _make_loader()
    src = tmp_path / "file.csv"
    src.write_text("data")
    blob = MagicMock()
    loader.client.bucket.return_value.blob.return_value = blob

    loader.upload_blob("bucket", str(src), "dest.csv", content_type="text/csv")
    assert blob.content_type == "text/csv"


# =============================================================================
# 4) load_csv_from_gcs
# =============================================================================

def test_load_csv_from_gcs_returns_dataframe(tmp_path):
    """load_csv_from_gcs downloads file and returns DataFrame."""
    loader = _make_loader()
    dest = str(tmp_path / "fin.csv")
    csv_content = "income,expenses\n5000,2000\n4000,1500"

    def fake_download(bucket, blob_name, destination):
        with open(destination, "w") as f:
            f.write(csv_content)
        return destination

    loader.download_blob = MagicMock(side_effect=fake_download)
    df = loader.load_csv_from_gcs("bucket", "raw/fin.csv", dest)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["income", "expenses"]
    assert len(df) == 2


# =============================================================================
# 5) load_json_from_gcs
# =============================================================================

def test_load_json_from_gcs_returns_dataframe(tmp_path):
    """load_json_from_gcs downloads JSONL and returns DataFrame."""
    loader = _make_loader()
    dest = str(tmp_path / "reviews.jsonl")
    jsonl = '{"id": 1, "rating": 5}\n{"id": 2, "rating": 3}\n'

    def fake_download(bucket, blob_name, destination):
        with open(destination, "w") as f:
            f.write(jsonl)
        return destination

    loader.download_blob = MagicMock(side_effect=fake_download)
    df = loader.load_json_from_gcs("bucket", "raw/reviews.jsonl", dest)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "rating" in df.columns


# =============================================================================
# 6) upload_dataframe
# =============================================================================

def test_upload_dataframe_csv(tmp_path):
    """upload_dataframe uploads CSV and returns GCS URI."""
    loader = _make_loader()
    df = pd.DataFrame([{"a": 1, "b": 2}])
    loader.upload_blob = MagicMock(return_value="gs://bucket/out.csv")

    with patch("pathlib.Path.mkdir"), \
         patch.object(df, "to_csv"), \
         patch("pathlib.Path.unlink", lambda self: None):
        uri = loader.upload_dataframe(df, "bucket", "processed/out.csv", format="csv")

    assert uri == "gs://bucket/out.csv"
    loader.upload_blob.assert_called_once()


def test_upload_dataframe_json(tmp_path):
    """upload_dataframe uploads JSON and returns GCS URI."""
    loader = _make_loader()
    df = pd.DataFrame([{"id": 1}])
    loader.upload_blob = MagicMock(return_value="gs://bucket/out.json")

    with patch("pathlib.Path.mkdir"), \
         patch.object(df, "to_json"), \
         patch("pathlib.Path.unlink", lambda self: None):
        uri = loader.upload_dataframe(df, "bucket", "processed/out.json", format="json")

    assert uri == "gs://bucket/out.json"


def test_upload_dataframe_unsupported_format_raises():
    """Unsupported format → raises ValueError."""
    loader = _make_loader()
    df = pd.DataFrame([{"a": 1}])
    with pytest.raises(ValueError, match="Unsupported format"):
        loader.upload_dataframe(df, "bucket", "out.xyz", format="xyz")


# =============================================================================
# 7) list_blobs
# =============================================================================

def test_list_blobs_returns_names():
    """list_blobs returns list of blob names."""
    loader = _make_loader()
    blob_a, blob_b = MagicMock(), MagicMock()
    blob_a.name = "raw/a.csv"
    blob_b.name = "raw/b.csv"
    loader.client.bucket.return_value.list_blobs.return_value = iter([blob_a, blob_b])

    result = loader.list_blobs("bucket", prefix="raw/")
    assert isinstance(result, list)
    assert len(result) == 2
    assert "raw/a.csv" in result
    assert "raw/b.csv" in result


def test_list_blobs_with_no_prefix():
    """list_blobs works without prefix."""
    loader = _make_loader()
    loader.client.bucket.return_value.list_blobs.return_value = iter([])
    result = loader.list_blobs("bucket")
    assert result == []


# =============================================================================
# 8) Convenience functions
# =============================================================================

def test_load_financial_data_returns_dataframe():
    df = pd.DataFrame([{"income": 5000}])
    with patch.object(M.GCSLoader, "load_csv_from_gcs", return_value=df):
        result = M.load_financial_data("bucket", "raw/fin.csv")
    assert isinstance(result, pd.DataFrame)


def test_load_product_data_returns_dataframe():
    df = pd.DataFrame([{"product_id": "p1"}])
    with patch.object(M.GCSLoader, "load_json_from_gcs", return_value=df):
        result = M.load_product_data("bucket", "raw/prod.jsonl")
    assert isinstance(result, pd.DataFrame)


def test_load_review_data_returns_dataframe():
    df = pd.DataFrame([{"rating": 5}])
    with patch.object(M.GCSLoader, "load_json_from_gcs", return_value=df):
        result = M.load_review_data("bucket", "raw/rev.jsonl")
    assert isinstance(result, pd.DataFrame)