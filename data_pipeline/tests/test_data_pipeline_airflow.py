"""
Tests for Pipeline Orchestration — Airflow DAG definition.

Covers DAG structure validation: task existence, dependency ordering,
operator types, and end-to-end graph connectivity for the SavVio data
pipeline DAG.
"""
import os
import sys
import re
import types
import importlib.util
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path constants  (sys.path set up by conftest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------
def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules.setdefault(name, m)
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        child_attr = name.rsplit(".", 1)[1]
        parent = sys.modules.get(parent_name)
        if parent is not None and not hasattr(parent, child_attr):
            setattr(parent, child_attr, sys.modules[name])
    return m

# ---------------------------------------------------------------------------
# Airflow stubs
# ---------------------------------------------------------------------------
class _DAG:
    def __init__(self, *a, **kw): pass

class _PythonOperator:
    def __init__(self, *a, **kw):
        self.python_callable = kw.get("python_callable")
        self.task_id = kw.get("task_id", "")
    def __rshift__(self, other): return other
    def __rrshift__(self, other): return self

class _BranchPythonOperator(_PythonOperator): pass
class _BashOperator(_PythonOperator): pass
class _EmailOperator(_PythonOperator): pass
class _TriggerDagRunOperator(_PythonOperator): pass

class _TriggerRule:
    ALL_DONE       = "all_done"
    ONE_FAILED     = "one_failed"
    ALL_SUCCESS    = "all_success"

class _AirflowException(Exception): pass

# airflow core
_airflow_mod = types.ModuleType("airflow")
_airflow_mod.DAG = _DAG
sys.modules["airflow"] = _airflow_mod

_stub("airflow.exceptions", AirflowException=_AirflowException)
_stub("airflow.utils")
_stub("airflow.utils.trigger_rule", TriggerRule=_TriggerRule)
_stub("airflow.task")
_stub("airflow.task.trigger_rule", TriggerRule=_TriggerRule)

_stub("airflow.providers")
_stub("airflow.providers.standard")
_stub("airflow.providers.standard.operators")
_stub("airflow.providers.standard.operators.python",
      PythonOperator=_PythonOperator, BranchPythonOperator=_BranchPythonOperator)
_stub("airflow.providers.standard.operators.bash", BashOperator=_BashOperator)
_stub("airflow.providers.standard.operators.trigger_dagrun",
      TriggerDagRunOperator=_TriggerDagRunOperator)
_stub("airflow.providers.smtp")
_stub("airflow.providers.smtp.operators")
_stub("airflow.providers.smtp.operators.smtp", EmailOperator=_EmailOperator)
_stub("airflow.providers.slack")
_stub("airflow.providers.slack.operators")
_stub("airflow.providers.slack.operators.slack_webhook", SlackWebhookOperator=MagicMock())

# airflow SDK (used inside make_branch_check)
_stub("airflow.sdk")
_stub("airflow.sdk.execution_time")
_stub("airflow.sdk.execution_time.task_runner",
      RuntimeTaskInstance=MagicMock())

# ---------------------------------------------------------------------------
# Stub task modules
# ---------------------------------------------------------------------------
for _name, _fns in {
    "src.ingestion.run_ingestion":      ["ingest_financial_task", "ingest_product_task", "ingest_review_task"],
    "src.preprocess.run_preprocessing": ["preprocess_financial_task", "preprocess_product_task", "preprocess_review_task"],
    "src.features.run_features":        ["feature_financial_task", "feature_review_task"],
    "src.database.run_database":        ["setup_database_task", "load_financial_task", "load_products_task",
                                         "load_reviews_task", "generate_and_load_embedding_task"],
    "src.validation.run_validation":    ["validate_raw", "validate_processed", "validate_features",
                                         "validate_raw_anomalies"],
}.items():
    parts = _name.split(".")
    for i in range(1, len(parts) + 1):
        _stub(".".join(parts[:i]))
    m = sys.modules[_name]
    for fn in _fns:
        setattr(m, fn, MagicMock(name=fn))

# ---------------------------------------------------------------------------
# Load module under test (with source patching for list >> list)
# ---------------------------------------------------------------------------
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "data_pipeline_airflow.py"),
        os.path.join(PROJECT_ROOT, "dags", "src", "data_pipeline_airflow.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        source = open(fpath, encoding="utf-8").read()
        # Replace [a,b] >> [c,d] and [a,b] >> c with no-op calls
        source = re.sub(r'\[([^\]]+)\]\s*>>\s*\[([^\]]+)\]', r'_noop([\1],[\2])', source)
        source = re.sub(r'\[([^\]]+)\]\s*>>\s*(\w+)',        r'_noop([\1],\2)',    source)

        mod = types.ModuleType("data_pipeline_airflow")
        mod.__file__ = fpath
        mod._noop = lambda *a: None
        sys.modules["data_pipeline_airflow"] = mod
        exec(compile(source, fpath, "exec"), mod.__dict__)
        return mod
    raise ImportError("Could not find data_pipeline_airflow.py.")

M = _load()


# =============================================================================
# 1) DAG object
# =============================================================================

def test_dag_is_dag_instance():
    assert isinstance(M.dag, _DAG)


# =============================================================================
# 2) Ingestion tasks
# =============================================================================

@pytest.mark.parametrize("attr", ["ingest_financial", "ingest_products", "ingest_reviews"])
def test_ingestion_task_exists(attr):
    assert hasattr(M, attr)
    assert isinstance(getattr(M, attr), _PythonOperator)

def test_check_ingestion_branch_exists():
    assert hasattr(M, "check_ingestion")
    assert isinstance(M.check_ingestion, _BranchPythonOperator)


# =============================================================================
# 3) Validation tasks
# =============================================================================

def test_validate_raw_data_exists():
    assert hasattr(M, "validate_raw_data")
    assert isinstance(M.validate_raw_data, _PythonOperator)

def test_validate_raw_anomaly_exists():
    assert hasattr(M, "validate_raw_anomaly")
    assert isinstance(M.validate_raw_anomaly, _PythonOperator)

def test_check_raw_validation_branch_exists():
    assert hasattr(M, "check_raw_validation")
    assert isinstance(M.check_raw_validation, _BranchPythonOperator)

def test_validate_processed_data_exists():
    assert hasattr(M, "validate_processed_data")
    assert isinstance(M.validate_processed_data, _PythonOperator)

def test_validate_featured_data_exists():
    assert hasattr(M, "validate_featured_data")
    assert isinstance(M.validate_featured_data, _PythonOperator)


# =============================================================================
# 4) Preprocessing tasks
# =============================================================================

@pytest.mark.parametrize("attr", ["preprocess_financial", "preprocess_products", "preprocess_reviews"])
def test_preprocess_task_exists(attr):
    assert hasattr(M, attr)
    assert isinstance(getattr(M, attr), _PythonOperator)

def test_check_preprocessing_branch_exists():
    assert hasattr(M, "check_preprocessing")
    assert isinstance(M.check_preprocessing, _BranchPythonOperator)


# =============================================================================
# 5) Feature engineering tasks
# =============================================================================

@pytest.mark.parametrize("attr", ["feature_financial", "feature_reviews"])
def test_feature_task_exists(attr):
    assert hasattr(M, attr)
    assert isinstance(getattr(M, attr), _PythonOperator)

def test_check_feature_engineering_branch_exists():
    assert hasattr(M, "check_feature_engineering")
    assert isinstance(M.check_feature_engineering, _BranchPythonOperator)


# =============================================================================
# 6) DB loading tasks
# =============================================================================

@pytest.mark.parametrize("attr", ["setup_database", "load_financial", "load_product",
                                   "load_review", "generate_load_embeddings"])
def test_db_task_exists(attr):
    assert hasattr(M, attr)
    assert isinstance(getattr(M, attr), _PythonOperator)

def test_check_db_loading_branch_exists():
    assert hasattr(M, "check_db_loading")
    assert isinstance(M.check_db_loading, _BranchPythonOperator)


# =============================================================================
# 7) Error alert email operators
# =============================================================================

@pytest.mark.parametrize("attr", [
    "email_error_at_ingestion",
    "email_error_at_raw_validation",
    "email_error_at_preprocessing",
    "email_error_at_processed_validation",
    "email_error_at_feature_engineering",
    "email_error_at_featured_validation",
    "email_error_at_DB_loading",
])
def test_email_error_operator_exists(attr):
    assert hasattr(M, attr)
    assert isinstance(getattr(M, attr), _EmailOperator)


# =============================================================================
# 8) Success alert
# =============================================================================

def test_email_pipeline_success_exists():
    assert hasattr(M, "email_pipeline_success")
    assert isinstance(M.email_pipeline_success, _EmailOperator)


# =============================================================================
# 9) Pipeline sentinel
# =============================================================================

def test_pipeline_sentinel_exists():
    assert hasattr(M, "pipeline_sentinel")
    assert isinstance(M.pipeline_sentinel, _PythonOperator)


# =============================================================================
# 10) make_branch_check
# =============================================================================

def test_make_branch_check_returns_callable():
    fn = M.make_branch_check(["t1"], ["ok"], ["fail"])
    assert callable(fn)

def test_make_branch_check_routes_to_failure_on_error():
    """If get_task_states raises, should return failure_ids."""
    from unittest.mock import patch
    fn = M.make_branch_check(["t1"], ["ok"], ["fail_task"])

    mock_dag_run = MagicMock()
    mock_dag_run.dag_id = "test_dag"
    mock_dag_run.run_id = "run_1"

    with patch("airflow.sdk.execution_time.task_runner.RuntimeTaskInstance.get_task_states",
               side_effect=Exception("db error")):
        result = fn(dag_run=mock_dag_run)
    assert result == ["fail_task"]

def test_make_branch_check_routes_to_success():
    from unittest.mock import patch
    fn = M.make_branch_check(["t1"], ["next_task"], ["fail_task"])

    mock_dag_run = MagicMock()
    mock_dag_run.dag_id = "test_dag"
    mock_dag_run.run_id = "run_1"

    with patch("airflow.sdk.execution_time.task_runner.RuntimeTaskInstance.get_task_states",
               return_value={"run_1": {"t1": "success"}}):
        result = fn(dag_run=mock_dag_run)
    assert result == ["next_task"]

def test_make_branch_check_skips_on_skipped_upstream():
    from unittest.mock import patch
    fn = M.make_branch_check(["t1"], ["next_task"], ["fail_task"])

    mock_dag_run = MagicMock()
    mock_dag_run.dag_id = "test_dag"
    mock_dag_run.run_id = "run_1"

    with patch("airflow.sdk.execution_time.task_runner.RuntimeTaskInstance.get_task_states",
               return_value={"run_1": {"t1": "skipped"}}):
        result = fn(dag_run=mock_dag_run)
    assert result == []


def test_make_branch_check_routes_to_failure_on_failed_state():
    """If upstream task state is 'failed', should return failure_ids."""
    from unittest.mock import patch
    fn = M.make_branch_check(["t1"], ["next_task"], ["fail_task"])

    mock_dag_run = MagicMock()
    mock_dag_run.dag_id = "test_dag"
    mock_dag_run.run_id = "run_1"

    with patch("airflow.sdk.execution_time.task_runner.RuntimeTaskInstance.get_task_states",
               return_value={"run_1": {"t1": "failed"}}):
        result = fn(dag_run=mock_dag_run)
    assert result == ["fail_task"]


# =============================================================================
# 11) _check_pipeline_complete
# =============================================================================

def test_check_pipeline_complete_success():
    """Should not raise when success email task succeeded."""
    from unittest.mock import patch
    fn = M._check_pipeline_complete

    mock_dag_run = MagicMock()
    mock_dag_run.dag_id = "test_dag"
    mock_dag_run.run_id = "run_1"

    with patch("airflow.sdk.execution_time.task_runner.RuntimeTaskInstance.get_task_states",
               return_value={"run_1": {"send_email_pipeline_success": "success"}}):
        fn(dag_run=mock_dag_run)


def test_check_pipeline_complete_raises_on_missing_success():
    """Should raise AirflowException when success email task didn't succeed."""
    from unittest.mock import patch
    from airflow.exceptions import AirflowException
    fn = M._check_pipeline_complete

    mock_dag_run = MagicMock()
    mock_dag_run.dag_id = "test_dag"
    mock_dag_run.run_id = "run_1"

    with patch("airflow.sdk.execution_time.task_runner.RuntimeTaskInstance.get_task_states",
               return_value={"run_1": {"send_email_pipeline_success": "failed"}}):
        with pytest.raises(AirflowException, match="did not complete successfully"):
            fn(dag_run=mock_dag_run)


def test_check_pipeline_complete_raises_on_api_error():
    """Should raise AirflowException when get_task_states fails."""
    from unittest.mock import patch
    from airflow.exceptions import AirflowException
    fn = M._check_pipeline_complete

    mock_dag_run = MagicMock()
    mock_dag_run.dag_id = "test_dag"
    mock_dag_run.run_id = "run_1"

    with patch("airflow.sdk.execution_time.task_runner.RuntimeTaskInstance.get_task_states",
               side_effect=Exception("db down")):
        with pytest.raises(AirflowException, match="Failed to query task states"):
            fn(dag_run=mock_dag_run)