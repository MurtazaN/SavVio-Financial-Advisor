"""
Tests for Validation Orchestration — run_validation.py.
"""
import os
import sys
import types
import importlib.util
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------------
# Stub validation_config
# ---------------------------------------------------------------------------
class Severity(Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"

@dataclass
class CheckResult:
    check_name:   str
    passed:       bool
    severity:     Severity
    dataset:      str
    stage:        str
    details:      str = ""
    metric_value: float = 0

@dataclass
class ValidationReport:
    stage: str
    results: list = field(default_factory=list)
    passed: bool = True
    has_warnings: bool = False

    def add(self, r):
        self.results.append(r)
        if not r.passed and r.severity == Severity.CRITICAL:
            self.passed = False
        if not r.passed and r.severity == Severity.WARNING:
            self.has_warnings = True

    def print_summary(self): pass
    def save(self): pass

    @property
    def summary(self):
        failed_critical = sum(1 for r in self.results if not r.passed and r.severity == Severity.CRITICAL)
        failed_warning  = sum(1 for r in self.results if not r.passed and r.severity == Severity.WARNING)
        failed_info     = sum(1 for r in self.results if not r.passed and r.severity == Severity.INFO)
        if failed_critical > 0:
            action = "HALT"
        elif failed_warning > 0:
            action = "ALERT"
        else:
            action = "CONTINUE"
        return {
            "stage": self.stage,
            "pipeline_action": action,
            "failed_critical": failed_critical,
            "failed_warning": failed_warning,
            "failed_info": failed_info,
            "timestamp": "2026-01-01T00:00:00",
        }

# ---------------------------------------------------------------------------
# Stub all modules before loading run_validation
# ---------------------------------------------------------------------------
_vc = types.ModuleType("savviocore.validation.validation_config")
_vc.Severity         = Severity
_vc.CheckResult      = CheckResult
_vc.ValidationReport = ValidationReport
_vc.load_thresholds  = lambda path=None: {}
sys.modules["savviocore.validation.validation_config"] = _vc

_run_raw_mock              = MagicMock()
_run_processed_mock        = MagicMock()
_run_feature_mock          = MagicMock()
_run_anomaly_mock          = MagicMock()
_run_raw_anomaly_mock      = MagicMock()

for _mod_name in (
    "validate", "validate.raw_validator", "validate.processed_validator",
    "src.validation.validate", "src.validation.validate.raw_validator",
    "src.validation.validate.processed_validator",
    "anomaly", "anomaly.anomaly_validator",
    "src.validation.anomaly", "src.validation.anomaly.anomaly_validator",
    "savviocore", "savviocore.validation", "savviocore.validation.feature_validator",
    "src", "src.validation", "src.utils",
):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

sys.modules["src.utils"].setup_logging = lambda *a, **kw: None

# Set mock functions on all possible import paths
for _mod_name in ("validate.raw_validator", "src.validation.validate.raw_validator"):
    sys.modules[_mod_name].run_raw_validation = _run_raw_mock

for _mod_name in ("validate.processed_validator", "src.validation.validate.processed_validator"):
    sys.modules[_mod_name].run_processed_validation = _run_processed_mock

for _mod_name in ("savviocore.validation.feature_validator",):
    sys.modules[_mod_name].run_feature_validation = _run_feature_mock

for _mod_name in ("anomaly.anomaly_validator", "src.validation.anomaly.anomaly_validator"):
    sys.modules[_mod_name].run_anomaly_validation     = _run_anomaly_mock
    sys.modules[_mod_name].run_raw_anomaly_validation = _run_raw_anomaly_mock

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "src", "validation", "run_validation.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        spec = importlib.util.spec_from_file_location("run_validation", fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["run_validation"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ImportError("Could not find run_validation.py. Searched:\n" + "\n".join(candidates))

M = _load()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _report(stage="raw", action="CONTINUE"):
    r = ValidationReport(stage=stage)
    if action == "HALT":
        r.results.append(CheckResult("c1", False, Severity.CRITICAL, "ds", stage))
        r.passed = False
    elif action == "ALERT":
        r.results.append(CheckResult("w1", False, Severity.WARNING, "ds", stage))
        r.has_warnings = True
    return r

def _reset():
    _run_raw_mock.reset_mock()
    _run_processed_mock.reset_mock()
    _run_feature_mock.reset_mock()
    _run_anomaly_mock.reset_mock()
    _run_raw_anomaly_mock.reset_mock()


# =============================================================================
# 1) _handle_report
# =============================================================================

def test_handle_report_continue_does_not_raise():
    r = _report(action="CONTINUE")
    M._handle_report(r)

def test_handle_report_halt_raises_runtime_error_with_stage():
    r = _report(stage="processed", action="HALT")
    with pytest.raises(RuntimeError, match="VALIDATION FAILED") as exc_info:
        M._handle_report(r)
    assert "processed" in str(exc_info.value)

def test_handle_report_alert_does_not_raise():
    r = _report(action="ALERT")
    M._handle_report(r)


# =============================================================================
# 2) _send_alert
# =============================================================================

def test_send_alert_called_on_alert(monkeypatch):
    r = _report(action="ALERT")
    mock_alert = MagicMock()
    monkeypatch.setattr(M, "_send_alert", mock_alert)
    M._handle_report(r)
    mock_alert.assert_called_once_with(r)


# =============================================================================
# 3) Airflow-compatible callables
# =============================================================================

def test_validate_raw_returns_summary():
    _reset()
    _run_raw_mock.return_value = _report("raw", "CONTINUE")
    result = M.validate_raw()
    assert isinstance(result, dict)
    assert "pipeline_action" in result

def test_validate_raw_halts_on_critical():
    _reset()
    _run_raw_mock.return_value = _report("raw", "HALT")
    with pytest.raises(RuntimeError):
        M.validate_raw()

def test_validate_processed_returns_summary():
    _reset()
    _run_processed_mock.return_value = _report("processed", "CONTINUE")
    result = M.validate_processed()
    assert "pipeline_action" in result

def test_validate_processed_halts_on_critical():
    _reset()
    _run_processed_mock.return_value = _report("processed", "HALT")
    with pytest.raises(RuntimeError):
        M.validate_processed()

def test_validate_features_returns_summary():
    _reset()
    _run_feature_mock.return_value = _report("features", "CONTINUE")
    result = M.validate_features()
    assert "pipeline_action" in result

def test_validate_features_halts_on_critical():
    _reset()
    _run_feature_mock.return_value = _report("features", "HALT")
    with pytest.raises(RuntimeError):
        M.validate_features()

def test_validate_raw_anomalies_never_halts():
    _reset()
    _run_raw_anomaly_mock.return_value = _report("raw_anomaly", "CONTINUE")
    result = M.validate_raw_anomalies()
    assert result["pipeline_action"] == "CONTINUE"

def test_validate_anomalies_returns_summary():
    _reset()
    _run_anomaly_mock.return_value = _report("anomaly", "CONTINUE")
    result = M.validate_anomalies()
    assert "pipeline_action" in result

def test_validate_anomalies_halts_on_critical():
    _reset()
    _run_anomaly_mock.return_value = _report("anomaly", "HALT")
    with pytest.raises(RuntimeError):
        M.validate_anomalies()


# =============================================================================
# 4) STAGE_MAP completeness
# =============================================================================

def test_stage_map_covers_all_pipeline_stages():
    expected = {"raw", "raw_anomalies", "processed", "features", "anomalies"}
    assert expected == set(M.STAGE_MAP.keys())
    pipeline_stages = {s for s, _ in M.VALIDATION_PIPELINE}
    assert pipeline_stages == set(M.STAGE_MAP.keys())