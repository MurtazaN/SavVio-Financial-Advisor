"""
Tests for Data Ingestion — run_ingestion.py orchestrator.

Covers GCS-vs-API routing logic, exception handling for each data source,
XCom push of record counts, and the Airflow task wrappers:
ingest_financial_task, ingest_product_task, ingest_review_task.
"""
import os
import sys
import types
import importlib.util
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path constants  (sys.path set up by conftest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------------
# Stub all dependencies before loading the module
# ---------------------------------------------------------------------------
def _make_stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules.setdefault(name, m)
    return m

# dotenv
_make_stub("dotenv", load_dotenv=lambda *a, **kw: None)

# google
for _n in ("google", "google.cloud", "google.cloud.storage",
           "google.oauth2", "google.oauth2.service_account"):
    _make_stub(_n, Client=MagicMock(), Credentials=MagicMock())

# requests / urllib3
_make_stub("requests", Session=MagicMock())
_make_stub("requests.exceptions", Timeout=Exception,
           HTTPError=Exception, RequestException=Exception)
_make_stub("requests.adapters", HTTPAdapter=MagicMock())
_make_stub("urllib3", util=MagicMock())
_make_stub("urllib3.util", retry=MagicMock())
_make_stub("urllib3.util.retry", Retry=MagicMock())

# ingestion.config (new import path)
_cfg = _make_stub("ingestion.config",
    ENVIRONMENT="dev",
    DATA_SOURCE="gcs",
    GCS_BUCKET_NAME="test-bucket",
    FINANCIAL_BLOB="raw/fin.csv",
    PRODUCT_BLOB="raw/prod.jsonl",
    REVIEW_BLOB="raw/rev.jsonl",
    FINANCIAL_RAW_PATH="/tmp/fin.csv",
    PRODUCT_RAW_PATH="/tmp/prod.jsonl",
    REVIEW_RAW_PATH="/tmp/rev.jsonl",
    GCP_CREDENTIALS_PATH=None,
    GCP_PROJECT_ID=None,
    API_BASE_URL="https://api.example.com/v1",
    API_KEY="key",
    API_TIMEOUT=30,
    FINANCIAL_API_ENDPOINT="https://api.example.com/v1/financial",
    PRODUCT_API_ENDPOINT="https://api.example.com/v1/products",
    REVIEW_API_ENDPOINT="https://api.example.com/v1/reviews",
    get_config_summary=lambda: {"environment": "dev"},
)

# ingestion sub-loaders (new paths)
_fake_fin  = pd.DataFrame([{"income": 5000}])
_fake_prod = pd.DataFrame([{"product_id": "p1"}])
_fake_rev  = pd.DataFrame([{"rating": 5}])

_gcs_stub = _make_stub("ingestion.gcs_loader",
    load_financial_data=MagicMock(return_value=_fake_fin),
    load_product_data=MagicMock(return_value=_fake_prod),
    load_review_data=MagicMock(return_value=_fake_rev),
)
_api_stub = _make_stub("ingestion.api_loader",
    load_financial_data=MagicMock(return_value=_fake_fin),
    load_product_data=MagicMock(return_value=_fake_prod),
    load_review_data=MagicMock(return_value=_fake_rev),
)
_ing = _make_stub("ingestion", config=_cfg,
                  gcs_loader=_gcs_stub, api_loader=_api_stub)

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "src", "ingestion", "run_ingestion.py"),
        os.path.join(PROJECT_ROOT, "dags", "src", "run_ingestion.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        spec = importlib.util.spec_from_file_location("run_ingestion", fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["run_ingestion"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ImportError("Could not find run_ingestion.py. Searched:\n" + "\n".join(candidates))

M = _load()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_DF_FIN  = pd.DataFrame([{"income": 5000}])
_DF_PROD = pd.DataFrame([{"product_id": "p1"}])
_DF_REV  = pd.DataFrame([{"rating": 5}])

def _make_context():
    return {"ti": MagicMock()}

def _reset_stubs():
    for stub in (_gcs_stub, _api_stub):
        stub.load_financial_data.reset_mock(return_value=True)
        stub.load_product_data.reset_mock(return_value=True)
        stub.load_review_data.reset_mock(return_value=True)
        stub.load_financial_data.return_value = _DF_FIN
        stub.load_product_data.return_value   = _DF_PROD
        stub.load_review_data.return_value    = _DF_REV
        stub.load_financial_data.side_effect  = None
        stub.load_product_data.side_effect    = None
        stub.load_review_data.side_effect     = None


# =============================================================================
# 1) run_ingestion — routing
# =============================================================================

def test_run_ingestion_routes_gcs():
    _reset_stubs()
    with patch.object(M, "DATA_SOURCE", "gcs"), \
         patch.object(M, "load_from_gcs", return_value=(_DF_FIN, _DF_PROD, _DF_REV)) as mock_gcs, \
         patch.object(M, "load_from_api") as mock_api:
        M.run_ingestion()
        mock_gcs.assert_called_once()
        mock_api.assert_not_called()

def test_run_ingestion_routes_api():
    _reset_stubs()
    with patch.object(M, "DATA_SOURCE", "api"), \
         patch.object(M, "load_from_api", return_value=(_DF_FIN, _DF_PROD, _DF_REV)) as mock_api, \
         patch.object(M, "load_from_gcs") as mock_gcs:
        M.run_ingestion()
        mock_api.assert_called_once()
        mock_gcs.assert_not_called()

def test_run_ingestion_raises_unknown_source():
    with patch.object(M, "DATA_SOURCE", "unknown"):
        with pytest.raises(ValueError, match="Unsupported data source"):
            M.run_ingestion()

def test_run_ingestion_returns_three_dataframes():
    with patch.object(M, "DATA_SOURCE", "gcs"), \
         patch.object(M, "load_from_gcs", return_value=(_DF_FIN, _DF_PROD, _DF_REV)):
        fin, prod, rev = M.run_ingestion()
        assert isinstance(fin, pd.DataFrame)
        assert isinstance(prod, pd.DataFrame)
        assert isinstance(rev, pd.DataFrame)

def test_run_ingestion_propagates_exception():
    with patch.object(M, "DATA_SOURCE", "gcs"), \
         patch.object(M, "load_from_gcs", side_effect=RuntimeError("down")):
        with pytest.raises(RuntimeError, match="down"):
            M.run_ingestion()


# =============================================================================
# 2) load_from_gcs
# =============================================================================

def test_load_from_gcs_returns_three_dataframes():
    _reset_stubs()
    fin, prod, rev = M.load_from_gcs()
    assert isinstance(fin, pd.DataFrame)
    assert isinstance(prod, pd.DataFrame)
    assert isinstance(rev, pd.DataFrame)

def test_load_from_gcs_raises_on_failure():
    _reset_stubs()
    _gcs_stub.load_financial_data.side_effect = Exception("gcs error")
    with pytest.raises(Exception, match="gcs error"):
        M.load_from_gcs()
    _gcs_stub.load_financial_data.side_effect = None


# =============================================================================
# 3) load_from_api
# =============================================================================

def test_load_from_api_returns_three_dataframes():
    _reset_stubs()
    fin, prod, rev = M.load_from_api()
    assert isinstance(fin, pd.DataFrame)
    assert isinstance(prod, pd.DataFrame)
    assert isinstance(rev, pd.DataFrame)

def test_load_from_api_raises_on_failure():
    _reset_stubs()
    _api_stub.load_financial_data.side_effect = Exception("api error")
    with pytest.raises(Exception, match="api error"):
        M.load_from_api()
    _api_stub.load_financial_data.side_effect = None


# =============================================================================
# 4) Airflow task functions
# =============================================================================

def test_ingest_financial_task_gcs():
    _reset_stubs()
    with patch.object(M, "DATA_SOURCE", "gcs"):
        result = M.ingest_financial_task(**_make_context())
    assert result["status"] == "success"
    assert result["records"] == len(_DF_FIN)

def test_ingest_financial_task_api():
    _reset_stubs()
    with patch.object(M, "DATA_SOURCE", "api"):
        result = M.ingest_financial_task(**_make_context())
    assert result["status"] == "success"

def test_ingest_product_task_gcs():
    _reset_stubs()
    with patch.object(M, "DATA_SOURCE", "gcs"):
        result = M.ingest_product_task(**_make_context())
    assert result["status"] == "success"
    assert result["records"] == len(_DF_PROD)

def test_ingest_product_task_api():
    _reset_stubs()
    with patch.object(M, "DATA_SOURCE", "api"):
        result = M.ingest_product_task(**_make_context())
    assert result["status"] == "success"

def test_ingest_review_task_gcs():
    _reset_stubs()
    with patch.object(M, "DATA_SOURCE", "gcs"):
        result = M.ingest_review_task(**_make_context())
    assert result["status"] == "success"

def test_ingest_review_task_api():
    _reset_stubs()
    with patch.object(M, "DATA_SOURCE", "api"):
        result = M.ingest_review_task(**_make_context())
    assert result["status"] == "success"

def test_ingest_financial_task_pushes_xcom():
    _reset_stubs()
    ctx = _make_context()
    with patch.object(M, "DATA_SOURCE", "gcs"):
        M.ingest_financial_task(**ctx)
    ctx["ti"].xcom_push.assert_called_once_with(
        key="financial_path", value=M.FINANCIAL_RAW_PATH
    )

def test_ingest_product_task_pushes_xcom():
    _reset_stubs()
    ctx = _make_context()
    with patch.object(M, "DATA_SOURCE", "gcs"):
        M.ingest_product_task(**ctx)
    ctx["ti"].xcom_push.assert_called_once_with(
        key="product_path", value=M.PRODUCT_RAW_PATH
    )

def test_ingest_review_task_pushes_xcom():
    _reset_stubs()
    ctx = _make_context()
    with patch.object(M, "DATA_SOURCE", "gcs"):
        M.ingest_review_task(**ctx)
    ctx["ti"].xcom_push.assert_called_once_with(
        key="review_path", value=M.REVIEW_RAW_PATH
    )