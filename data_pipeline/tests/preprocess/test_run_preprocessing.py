"""
Tests for Data Preprocessing — run_preprocessing.py orchestrator.

Covers main() CLI (exception handling, execution order) and the Airflow task
wrappers: preprocess_financial_task, preprocess_product_task,
preprocess_review_task.
"""
import os
import sys
import types
import importlib.util
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Path constants  (sys.path set up by conftest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------------
# Stub preprocess sub-modules BEFORE loading run_preprocessing
# ---------------------------------------------------------------------------
def _stub_preprocess():
    for name in ("preprocess", "preprocess.financial", "preprocess.product", "preprocess.review"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    sys.modules["preprocess.financial"].main = MagicMock()
    sys.modules["preprocess.product"].main  = MagicMock()
    sys.modules["preprocess.review"].main   = MagicMock()

    # Make attributes accessible via the parent module
    sys.modules["preprocess"].financial = sys.modules["preprocess.financial"]
    sys.modules["preprocess"].product   = sys.modules["preprocess.product"]
    sys.modules["preprocess"].review    = sys.modules["preprocess.review"]

_stub_preprocess()

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "src", "preprocess", "run_preprocessing.py"),
        os.path.join(PROJECT_ROOT, "src", "preprocess", "run_preprocessing.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        spec = importlib.util.spec_from_file_location("run_preprocessing", fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["run_preprocessing"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ImportError("Could not find run_preprocessing.py. Searched:\n" + "\n".join(candidates))

M = _load()

# Shortcuts to the stubbed main() functions
_fin  = sys.modules["preprocess.financial"]
_prod = sys.modules["preprocess.product"]
_rev  = sys.modules["preprocess.review"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _reset():
    """Reset all main() mocks before each test."""
    _fin.main.reset_mock()
    _prod.main.reset_mock()
    _rev.main.reset_mock()


# =============================================================================
# 1) run_pipeline
# =============================================================================

def test_run_pipeline_calls_all_three_mains():
    _reset()
    M.run_pipeline()
    _fin.main.assert_called_once()
    _prod.main.assert_called_once()
    _rev.main.assert_called_once()


def test_run_pipeline_calls_in_order():
    _reset()
    call_order = []
    _fin.main.side_effect  = lambda: call_order.append("financial")
    _prod.main.side_effect = lambda: call_order.append("product")
    _rev.main.side_effect  = lambda: call_order.append("review")

    M.run_pipeline()
    assert call_order == ["financial", "product", "review"]

    # restore
    _fin.main.side_effect  = None
    _prod.main.side_effect = None
    _rev.main.side_effect  = None


@pytest.mark.parametrize("failing_module", ["financial", "product", "review"])
def test_run_pipeline_exits_on_stage_failure(failing_module):
    _reset()
    mod = {"financial": _fin, "product": _prod, "review": _rev}[failing_module]
    mod.main.side_effect = Exception(f"{failing_module} boom")
    with pytest.raises(SystemExit):
        M.run_pipeline()
    mod.main.side_effect = None


# =============================================================================
# 2) Airflow task wrappers (parametrized)
# =============================================================================

@pytest.mark.parametrize("task_fn,mock_mod", [
    ("preprocess_financial_task", "financial"),
    ("preprocess_product_task", "product"),
    ("preprocess_review_task", "review"),
])
class TestPreprocessTasks:
    def test_task_calls_main_with_context(self, task_fn, mock_mod):
        _reset()
        mod = {"financial": _fin, "product": _prod, "review": _rev}[mock_mod]
        getattr(M, task_fn)(ti=MagicMock())
        mod.main.assert_called_once()

    def test_task_propagates_exception(self, task_fn, mock_mod):
        _reset()
        mod = {"financial": _fin, "product": _prod, "review": _rev}[mock_mod]
        mod.main.side_effect = RuntimeError("oops")
        with pytest.raises(RuntimeError, match="oops"):
            getattr(M, task_fn)()
        mod.main.side_effect = None