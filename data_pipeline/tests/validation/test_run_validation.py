"""
Tests for Validation Orchestration — run_validation.py.

Covers the validation pipeline orchestrator: _handle_report (continue/halt/alert),
_send_alert, the stage-specific task wrappers (validate_raw, validate_processed,
validate_features, validate_anomalies), and the STAGE_MAP routing table.
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
# Path constants  (sys.path set up by conftest.py)
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

_vc = types.ModuleType("savviocore.validation.validation_config")
_vc.Severity         = Severity
_vc.CheckResult      = CheckResult
_vc.ValidationReport = ValidationReport
_vc.load_thresholds  = lambda path=None: {}
sys.modules["savviocore.validation.validation_config"] = _vc

# Stub sub-validator modules
for _mod in ("validate.raw_validator", "validate.processed_validator",
             "savviocore.validation.feature_validator", "anomaly.anomaly_validator",
             "validate", "anomaly", "savviocore", "savviocore.validation"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

sys.modules["validate.raw_validator"].run_raw_validation          = MagicMock()
sys.modules["validate.processed_validator"].run_processed_validation = MagicMock()
sys.modules["savviocore.validation.feature_validator"].run_feature_validation  = MagicMock()
sys.modules["anomaly.anomaly_validator"].run_anomaly_validation      = MagicMock()
sys.modules["anomaly.anomaly_validator"].run_raw_anomaly_validation  = MagicMock()

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


# =============================================================================
# 1) _handle_report
# =============================================================================

def test_handle_report_continue_does_not_raise():
    r = _report(action="CONTINUE")
    M._handle_report(r)  # must not raise

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
    sys.modules["validate.raw_validator"].run_raw_validation.return_value = _report("raw", "CONTINUE")
    result = M.validate_raw()
    assert isinstance(result, dict)
    assert "pipeline_action" in result

def test_validate_raw_halts_on_critical():
    sys.modules["validate.raw_validator"].run_raw_validation.return_value = _report("raw", "HALT")
    with pytest.raises(RuntimeError):
        M.validate_raw()

def test_validate_processed_returns_summary():
    sys.modules["validate.processed_validator"].run_processed_validation.return_value = _report("processed", "CONTINUE")
    result = M.validate_processed()
    assert "pipeline_action" in result

def test_validate_processed_halts_on_critical():
    sys.modules["validate.processed_validator"].run_processed_validation.return_value = _report("processed", "HALT")
    with pytest.raises(RuntimeError):
        M.validate_processed()

def test_validate_features_returns_summary():
    sys.modules["savviocore.validation.feature_validator"].run_feature_validation.return_value = _report("features", "CONTINUE")
    result = M.validate_features()
    assert "pipeline_action" in result

def test_validate_features_halts_on_critical():
    sys.modules["savviocore.validation.feature_validator"].run_feature_validation.return_value = _report("features", "HALT")
    with pytest.raises(RuntimeError):
        M.validate_features()

def test_validate_raw_anomalies_never_halts():
    """Tier-1 raw anomaly scan is INFO-only and must never halt."""
    sys.modules["anomaly.anomaly_validator"].run_raw_anomaly_validation.return_value = _report("raw_anomaly", "CONTINUE")
    result = M.validate_raw_anomalies()
    assert result["pipeline_action"] == "CONTINUE"

def test_validate_anomalies_returns_summary():
    sys.modules["anomaly.anomaly_validator"].run_anomaly_validation.return_value = _report("anomaly", "CONTINUE")
    result = M.validate_anomalies()
    assert "pipeline_action" in result

def test_validate_anomalies_halts_on_critical():
    sys.modules["anomaly.anomaly_validator"].run_anomaly_validation.return_value = _report("anomaly", "HALT")
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