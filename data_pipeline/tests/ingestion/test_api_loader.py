"""
Tests for Data Ingestion — api_loader.py.

Covers APILoader initialisation, request making, pagination, file saving,
and the convenience functions for loading financial, product, and review data
from external API endpoints.
"""
import os
import sys
import json
import types
import importlib.util
from unittest.mock import MagicMock, patch, mock_open

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub requests if not installed
# ---------------------------------------------------------------------------
if "requests" not in sys.modules:
    _req = types.ModuleType("requests")

    class _Session:
        def __init__(self): self.headers = MagicMock()
        def mount(self, *a): pass
        def request(self, *a, **kw): pass

    class _Timeout(Exception): pass
    class _HTTPError(Exception): pass
    class _RequestException(Exception): pass

    _exceptions = types.ModuleType("requests.exceptions")
    _exceptions.Timeout = _Timeout
    _exceptions.HTTPError = _HTTPError
    _exceptions.RequestException = _RequestException

    _adapters = types.ModuleType("requests.adapters")
    _adapters.HTTPAdapter = MagicMock()

    _urllib3 = types.ModuleType("urllib3")
    _urllib3_util = types.ModuleType("urllib3.util")
    _urllib3_util_retry = types.ModuleType("urllib3.util.retry")
    _urllib3_util_retry.Retry = MagicMock()

    _req.Session = _Session
    _req.exceptions = _exceptions
    _req.adapters = _adapters
    sys.modules["requests"] = _req
    sys.modules["requests.exceptions"] = _exceptions
    sys.modules["requests.adapters"] = _adapters
    sys.modules["urllib3"] = _urllib3
    sys.modules["urllib3.util"] = _urllib3_util
    sys.modules["urllib3.util.retry"] = _urllib3_util_retry

import requests

# ---------------------------------------------------------------------------
# Path constants  (sys.path set up by conftest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "src", "ingestion", "api_loader.py"),
        os.path.join(PROJECT_ROOT, "dags", "src", "api_loader.py"),
        os.path.join(PROJECT_ROOT, "src", "ingestion", "api_loader.py"),
        os.path.join(PROJECT_ROOT, "api_loader.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        spec = importlib.util.spec_from_file_location("api_loader", fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["api_loader"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ImportError("Could not find api_loader.py. Searched:\n" + "\n".join(candidates))

M = _load()
APILoader = M.APILoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mock_response(json_data, status_code=200):
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = json.dumps(json_data)
    resp.raise_for_status = MagicMock()
    return resp


def _make_loader(**kwargs):
    """Instantiate APILoader with requests.Session patched out."""
    with patch("requests.Session"):
        return APILoader(base_url="https://api.example.com", **kwargs)


# =============================================================================
# 1) __init__
# =============================================================================

def test_init_sets_base_url():
    loader = _make_loader()
    assert loader.base_url == "https://api.example.com"


def test_init_strips_trailing_slash():
    with patch("requests.Session"):
        loader = APILoader(base_url="https://api.example.com/")
    assert loader.base_url == "https://api.example.com"


def test_init_sets_api_key_header():
    loader = _make_loader(api_key="secret123")
    loader.session.headers.update.assert_called()
    # Collect all calls to headers.update and check Authorization was set
    all_calls = [str(c) for c in loader.session.headers.update.call_args_list]
    assert any("Bearer secret123" in c for c in all_calls)


def test_init_no_api_key_no_auth_header():
    loader = _make_loader(api_key=None)
    all_calls = [str(c) for c in loader.session.headers.update.call_args_list]
    assert not any("Bearer" in c for c in all_calls)


# =============================================================================
# 2) _make_request
# =============================================================================

def test_make_request_get_success():
    """Successful GET returns parsed JSON."""
    loader = _make_loader()
    loader.session.request.return_value = _mock_response({"data": [{"id": 1}]})

    result = loader._make_request("/test")
    assert result == {"data": [{"id": 1}]}
    loader.session.request.assert_called_once()
    args, kwargs = loader.session.request.call_args
    assert kwargs["method"] == "GET" or args[0] == "GET"


def test_make_request_builds_correct_url():
    loader = _make_loader()
    loader.session.request.return_value = _mock_response({})
    loader._make_request("/financial")
    _, kwargs = loader.session.request.call_args
    assert kwargs.get("url", "") == "https://api.example.com/financial" or \
           "https://api.example.com/financial" in str(loader.session.request.call_args)


def test_make_request_passes_params():
    loader = _make_loader()
    loader.session.request.return_value = _mock_response([])
    loader._make_request("/ep", params={"page": 1})
    _, kwargs = loader.session.request.call_args
    assert kwargs.get("params") == {"page": 1}


def test_make_request_raises_on_timeout():
    import requests
    loader = _make_loader()
    loader.session.request.side_effect = requests.exceptions.Timeout
    with pytest.raises(requests.exceptions.Timeout):
        loader._make_request("/ep")


def test_make_request_raises_on_http_error():
    import requests
    loader = _make_loader()
    resp = _mock_response({}, status_code=500)
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
    loader.session.request.return_value = resp
    with pytest.raises(requests.exceptions.HTTPError):
        loader._make_request("/ep")


# =============================================================================
# 3) fetch_with_pagination
# =============================================================================

def test_fetch_with_pagination_single_page():
    """Single page response (has_more=False) returns all records."""
    loader = _make_loader()
    loader._make_request = MagicMock(return_value={
        "data": [{"id": 1}, {"id": 2}],
        "has_more": False
    })
    with patch("time.sleep"):
        result = loader.fetch_with_pagination("/ep", page_size=100)
    assert result == [{"id": 1}, {"id": 2}]


def test_fetch_with_pagination_multiple_pages():
    """Fetches all pages until has_more is False."""
    loader = _make_loader()
    # page 2 returns fewer records than page_size=10 → loop stops naturally
    responses = [
        {"data": [{"id": i} for i in range(10)], "has_more": True},
        {"data": [{"id": i} for i in range(10, 13)], "has_more": False},
    ]
    loader._make_request = MagicMock(side_effect=responses)
    with patch("time.sleep"):
        result = loader.fetch_with_pagination("/ep", page_size=10)
    assert len(result) == 13


def test_fetch_with_pagination_list_response():
    """API returning a plain list is handled correctly."""
    loader = _make_loader()
    loader._make_request = MagicMock(return_value=[{"id": 1}])
    with patch("time.sleep"):
        result = loader.fetch_with_pagination("/ep", page_size=100)
    assert result == [{"id": 1}]


def test_fetch_with_pagination_respects_max_pages():
    """max_pages stops fetching after N pages."""
    loader = _make_loader()
    loader._make_request = MagicMock(return_value={"data": [{"id": 1}], "has_more": True})
    with patch("time.sleep"):
        result = loader.fetch_with_pagination("/ep", page_size=1, max_pages=2)
    assert loader._make_request.call_count == 2


def test_fetch_with_pagination_empty_response_stops():
    """Empty records on first page → returns empty list."""
    loader = _make_loader()
    loader._make_request = MagicMock(return_value={"data": [], "has_more": False})
    with patch("time.sleep"):
        result = loader.fetch_with_pagination("/ep")
    assert result == []


# =============================================================================
# 4) save_to_file
# =============================================================================

def test_save_to_file_json(tmp_path):
    loader = _make_loader()
    data = [{"id": 1, "name": "test"}]
    fpath = str(tmp_path / "out.json")
    result = loader.save_to_file(data, fpath, format="json")
    assert result == fpath
    with open(fpath) as f:
        loaded = json.load(f)
    assert loaded == data


def test_save_to_file_csv(tmp_path):
    loader = _make_loader()
    data = [{"id": 1, "name": "test"}, {"id": 2, "name": "foo"}]
    fpath = str(tmp_path / "out.csv")
    loader.save_to_file(data, fpath, format="csv")
    df = pd.read_csv(fpath)
    assert list(df.columns) == ["id", "name"]
    assert len(df) == 2


def test_save_to_file_unsupported_format_raises(tmp_path):
    loader = _make_loader()
    with pytest.raises(ValueError, match="Unsupported format"):
        loader.save_to_file([{}], str(tmp_path / "out.xyz"), format="xyz")


def test_save_to_file_creates_parent_dirs(tmp_path):
    loader = _make_loader()
    fpath = str(tmp_path / "nested" / "dir" / "out.json")
    loader.save_to_file([{"a": 1}], fpath, format="json")
    assert os.path.exists(fpath)


# =============================================================================
# 5) fetch_and_save
# =============================================================================

def test_fetch_and_save_returns_dataframe(tmp_path):
    loader = _make_loader()
    data = [{"id": 1, "val": "x"}, {"id": 2, "val": "y"}]
    loader.fetch_with_pagination = MagicMock(return_value=data)
    fpath = str(tmp_path / "out.json")
    df = loader.fetch_and_save("/ep", fpath, format="json", use_pagination=True)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_fetch_and_save_no_pagination(tmp_path):
    loader = _make_loader()
    loader._make_request = MagicMock(return_value=[{"id": 1}])
    fpath = str(tmp_path / "out.json")
    df = loader.fetch_and_save("/ep", fpath, format="json", use_pagination=False)
    assert len(df) == 1


# =============================================================================
# 6) Convenience functions
# =============================================================================

def test_load_financial_data_returns_dataframe(tmp_path):
    data = [{"income": 5000, "expenses": 2000}]
    with patch.object(M.APILoader, "fetch_and_save", return_value=pd.DataFrame(data)):
        df = M.load_financial_data(
            api_base_url="https://api.example.com",
            destination_path=str(tmp_path / "fin.csv")
        )
    assert isinstance(df, pd.DataFrame)
    assert "income" in df.columns


def test_load_product_data_returns_dataframe(tmp_path):
    data = [{"product_id": "p1", "name": "Widget"}]
    with patch.object(M.APILoader, "fetch_and_save", return_value=pd.DataFrame(data)):
        df = M.load_product_data(
            api_base_url="https://api.example.com",
            destination_path=str(tmp_path / "prod.json")
        )
    assert isinstance(df, pd.DataFrame)


def test_load_review_data_returns_dataframe(tmp_path):
    data = [{"product_id": "p1", "rating": 5}]
    with patch.object(M.APILoader, "fetch_and_save", return_value=pd.DataFrame(data)):
        df = M.load_review_data(
            api_base_url="https://api.example.com",
            destination_path=str(tmp_path / "rev.json")
        )
    assert isinstance(df, pd.DataFrame)