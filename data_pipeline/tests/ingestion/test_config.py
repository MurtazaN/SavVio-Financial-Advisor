"""
Tests for Data Ingestion — ingestion/config.py.

Covers default configuration values, environment-specific overrides,
GCS path construction, API endpoint configs, threshold settings, and
validation of the configuration module constants.
"""
import os
import sys
import types
import importlib
import importlib.util
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path constants  (sys.path set up by conftest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------------
# Stub dotenv so load_dotenv() is a no-op
# ---------------------------------------------------------------------------
if "dotenv" not in sys.modules:
    _dotenv = types.ModuleType("dotenv")
    _dotenv.load_dotenv = lambda *a, **kw: None
    sys.modules["dotenv"] = _dotenv

# ---------------------------------------------------------------------------
# Helper: reload config with a given set of env vars
# ---------------------------------------------------------------------------
def _load_config(env: dict = None):
    """
    Re-import config.py from file with the given environment variables.
    Isolates each test from side effects of previous imports.
    """
    env = env or {}
    # Remove existing module to force re-execution
    sys.modules.pop("config", None)

    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "src", "ingestion", "config.py"),
        os.path.join(PROJECT_ROOT, "dags", "src", "config.py"),
        os.path.join(PROJECT_ROOT, "src", "ingestion", "config.py"),
        os.path.join(PROJECT_ROOT, "config.py"),
    ]
    fpath = next((p for p in candidates if os.path.isfile(p)), None)
    if not fpath:
        raise ImportError("Could not find config.py. Searched:\n" + "\n".join(candidates))

    with patch.dict(os.environ, env, clear=False), \
         patch("dotenv.load_dotenv", lambda *a, **kw: None):
        spec = importlib.util.spec_from_file_location("config", fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["config"] = mod
        spec.loader.exec_module(mod)
    return mod


# Load once for tests that don't need custom env
C = _load_config({"ENVIRONMENT": "dev", "GCP_CREDENTIALS_PATH": ""})


# =============================================================================
# 1) GCS defaults
# =============================================================================

def test_gcs_bucket_name_default():
    # Don't set GCS_BUCKET_NAME at all → os.getenv uses default
    env = {k: v for k, v in os.environ.items() if k != "GCS_BUCKET_NAME"}
    env.update({"ENVIRONMENT": "dev", "GCP_CREDENTIALS_PATH": ""})
    with patch.dict(os.environ, env, clear=True):
        sys.modules.pop("config", None)
        candidates = [
            os.path.join(PROJECT_ROOT, "dags", "src", "ingestion", "config.py"),
            os.path.join(PROJECT_ROOT, "dags", "src", "config.py"),
        ]
        fpath = next((p for p in candidates if os.path.isfile(p)), None)
        spec = importlib.util.spec_from_file_location("config", fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["config"] = mod
        spec.loader.exec_module(mod)
    assert mod.GCS_BUCKET_NAME == "savvio-data-bucket"


def test_gcs_bucket_name_from_env():
    c = _load_config({"GCS_BUCKET_NAME": "my-bucket", "ENVIRONMENT": "dev", "GCP_CREDENTIALS_PATH": ""})
    assert c.GCS_BUCKET_NAME == "my-bucket"


def test_gcs_paths_defaults():
    c = _load_config({"ENVIRONMENT": "dev", "GCP_CREDENTIALS_PATH": ""})
    assert c.GCS_RAW_PATH == "raw/"
    assert c.GCS_PROCESSED_PATH == "processed/"
    assert c.GCS_FEATURES_PATH == "features/"
    assert c.GCS_VALIDATED_PATH == "validated/"


def test_blob_paths_defaults():
    c = _load_config({"ENVIRONMENT": "dev", "GCP_CREDENTIALS_PATH": ""})
    assert c.FINANCIAL_BLOB == "raw/financial_data.csv"
    assert c.PRODUCT_BLOB == "raw/product_data.jsonl"
    assert c.REVIEW_BLOB == "raw/review_data.jsonl"


# =============================================================================
# 2) API configuration
# =============================================================================

def test_api_timeout_default():
    c = _load_config({"ENVIRONMENT": "dev", "GCP_CREDENTIALS_PATH": ""})
    assert c.API_TIMEOUT == 30


def test_api_timeout_from_env():
    c = _load_config({"API_TIMEOUT": "60", "ENVIRONMENT": "dev", "GCP_CREDENTIALS_PATH": ""})
    assert c.API_TIMEOUT == 60


# =============================================================================
# 3) Environment & data source
# =============================================================================

def test_environment_default_is_dev():
    c = _load_config({"GCP_CREDENTIALS_PATH": ""})
    assert c.ENVIRONMENT == "dev"


def test_data_source_dev_is_gcs():
    c = _load_config({"ENVIRONMENT": "dev", "GCP_CREDENTIALS_PATH": ""})
    assert c.DATA_SOURCE == "gcs"


def test_data_source_prod_is_api():
    c = _load_config({"ENVIRONMENT": "prod", "GCP_CREDENTIALS_PATH": ""})
    assert c.DATA_SOURCE == "api"


def test_data_source_custom_env():
    c = _load_config({"ENVIRONMENT": "staging", "DATA_SOURCE": "local", "GCP_CREDENTIALS_PATH": ""})
    assert c.DATA_SOURCE == "local"


# =============================================================================
# 4) Pipeline thresholds
# =============================================================================

def test_max_missing_values_default():
    c = _load_config({"ENVIRONMENT": "dev", "GCP_CREDENTIALS_PATH": ""})
    assert c.MAX_MISSING_VALUES_PCT == pytest.approx(0.1)


def test_min_records_required_default():
    c = _load_config({"ENVIRONMENT": "dev", "GCP_CREDENTIALS_PATH": ""})
    assert c.MIN_RECORDS_REQUIRED == 100


def test_min_records_from_env():
    c = _load_config({"MIN_RECORDS_REQUIRED": "500", "ENVIRONMENT": "dev", "GCP_CREDENTIALS_PATH": ""})
    assert c.MIN_RECORDS_REQUIRED == 500


# =============================================================================
# 5) validate_config()
# =============================================================================

def test_validate_config_passes_dev_with_bucket():
    c = _load_config({
        "ENVIRONMENT": "dev",
        "GCS_BUCKET_NAME": "my-bucket",
        "GCP_CREDENTIALS_PATH": ""
    })
    c.validate_config()  # must not raise


def test_validate_config_passes_prod_with_api_creds():
    c = _load_config({
        "ENVIRONMENT": "prod",
        "API_BASE_URL": "https://api.example.com",
        "API_KEY": "secret",
        "GCP_CREDENTIALS_PATH": ""
    })
    c.validate_config()  # must not raise


def test_validate_config_fails_prod_without_api_key():
    c = _load_config({
        "ENVIRONMENT": "prod",
        "API_BASE_URL": "https://api.example.com",
        "GCP_CREDENTIALS_PATH": ""
    })
    # Remove API_KEY from the module scope
    c.API_KEY = None
    with pytest.raises(ValueError, match="API_KEY"):
        c.validate_config()



# =============================================================================
# 6) get_config_summary()
# =============================================================================

def test_get_config_summary_has_expected_keys():
    c = _load_config({"ENVIRONMENT": "dev", "GCP_CREDENTIALS_PATH": ""})
    summary = c.get_config_summary()
    assert isinstance(summary, dict)
    for key in ("environment", "data_source", "gcs_bucket", "api_base_url", "log_level", "data_dir"):
        assert key in summary


def test_get_config_summary_no_api_key():
    """API key must not appear in config summary."""
    c = _load_config({"ENVIRONMENT": "dev", "API_KEY": "topsecret", "GCP_CREDENTIALS_PATH": ""})
    summary = c.get_config_summary()
    assert "topsecret" not in str(summary)
    assert "api_key" not in summary


# =============================================================================
# 7) GCP credentials path validation
# =============================================================================

def test_gcp_credentials_path_missing_raises(tmp_path):
    bad_path = str(tmp_path / "nonexistent_creds.json")
    with pytest.raises(FileNotFoundError, match="GCP credentials file not found"):
        _load_config({
            "ENVIRONMENT": "dev",
            "GCP_CREDENTIALS_PATH": bad_path,
        })


def test_gcp_credentials_path_valid_does_not_raise(tmp_path):
    creds = tmp_path / "creds.json"
    creds.write_text("{}")
    c = _load_config({
        "ENVIRONMENT": "dev",
        "GCP_CREDENTIALS_PATH": str(creds),
    })
    assert c.GCP_CREDENTIALS_PATH == str(creds)