"""
Tests for Feature Engineering — run_features.py orchestrator.

Covers the main() CLI (skip flags, exception handling) and the Airflow task
wrappers: feature_financial_task, feature_review_task.
"""
import os
import sys
import types
import importlib.util
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path constants  (sys.path set up by conftest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------------
# Stub dependencies before loading module
# ---------------------------------------------------------------------------
_utils = sys.modules.get("utils", types.ModuleType("utils"))
_utils.setup_logging = lambda *a, **kw: None
sys.modules["utils"] = _utils

for _name, _attrs in {
    "financial_features": {"run_financial_features": MagicMock()},
    "product_review_features": {"run_review_features": MagicMock()},
}.items():
    m = types.ModuleType(_name)
    for k, v in _attrs.items():
        setattr(m, k, v)
    sys.modules[_name] = m

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "src", "features", "run_features.py"),
        os.path.join(PROJECT_ROOT, "dags", "src", "run_features.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        spec = importlib.util.spec_from_file_location("run_features", fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["run_features"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ImportError("Could not find run_features.py. Searched:\n" + "\n".join(candidates))

M = _load()

_fin = sys.modules["financial_features"]
_rev = sys.modules["product_review_features"]

def _reset():
    _fin.run_financial_features.reset_mock()
    _rev.run_review_features.reset_mock()


# =============================================================================
# 1) main()
# =============================================================================

def test_main_calls_both_by_default():
    _reset()
    with patch("sys.argv", ["run_features.py"]):
        M.main()
    _fin.run_financial_features.assert_called_once()
    _rev.run_review_features.assert_called_once()

def test_main_skip_financial():
    _reset()
    with patch("sys.argv", ["run_features.py", "--skip-financial"]):
        M.main()
    _fin.run_financial_features.assert_not_called()
    _rev.run_review_features.assert_called_once()

def test_main_skip_reviews():
    _reset()
    with patch("sys.argv", ["run_features.py", "--skip-reviews"]):
        M.main()
    _fin.run_financial_features.assert_called_once()
    _rev.run_review_features.assert_not_called()

def test_main_skip_both():
    _reset()
    with patch("sys.argv", ["run_features.py", "--skip-financial", "--skip-reviews"]):
        M.main()
    _fin.run_financial_features.assert_not_called()
    _rev.run_review_features.assert_not_called()

def test_main_financial_exception_does_not_crash():
    _reset()
    _fin.run_financial_features.side_effect = Exception("boom")
    with patch("sys.argv", ["run_features.py"]):
        M.main()  # must not raise
    _rev.run_review_features.assert_called_once()
    _fin.run_financial_features.side_effect = None

def test_main_review_exception_does_not_crash():
    _reset()
    _rev.run_review_features.side_effect = Exception("boom")
    with patch("sys.argv", ["run_features.py"]):
        M.main()  # must not raise
    _fin.run_financial_features.assert_called_once()
    _rev.run_review_features.side_effect = None


# =============================================================================
# 2) feature_financial_task
# =============================================================================

def test_feature_financial_task_calls_run():
    _reset()
    M.feature_financial_task()
    _fin.run_financial_features.assert_called_once()

def test_feature_financial_task_passes_correct_paths():
    _reset()
    M.feature_financial_task()
    _, kwargs = _fin.run_financial_features.call_args
    assert kwargs["input_path"].endswith("financial_preprocessed.csv")
    assert kwargs["output_path"].endswith("financial_featured.csv")

def test_feature_financial_task_accepts_context():
    _reset()
    M.feature_financial_task(ti=MagicMock())
    _fin.run_financial_features.assert_called_once()

def test_feature_financial_task_propagates_exception():
    _reset()
    _fin.run_financial_features.side_effect = RuntimeError("oops")
    with pytest.raises(RuntimeError, match="oops"):
        M.feature_financial_task()
    _fin.run_financial_features.side_effect = None


# =============================================================================
# 3) feature_review_task
# =============================================================================

def test_feature_review_task_calls_run():
    _reset()
    M.feature_review_task()
    _rev.run_review_features.assert_called_once()

def test_feature_review_task_passes_correct_paths():
    _reset()
    M.feature_review_task()
    _, kwargs = _rev.run_review_features.call_args
    assert kwargs["reviews_path"].endswith("review_preprocessed.jsonl")
    assert kwargs["products_path"].endswith("product_preprocessed.jsonl")
    assert kwargs["product_output_path"].endswith("product_featured.jsonl")
    assert kwargs["review_output_path"].endswith("review_featured.jsonl")

def test_feature_review_task_accepts_context():
    _reset()
    M.feature_review_task(ti=MagicMock())
    _rev.run_review_features.assert_called_once()

def test_feature_review_task_propagates_exception():
    _reset()
    _rev.run_review_features.side_effect = RuntimeError("oops")
    with pytest.raises(RuntimeError, match="oops"):
        M.feature_review_task()
    _rev.run_review_features.side_effect = None