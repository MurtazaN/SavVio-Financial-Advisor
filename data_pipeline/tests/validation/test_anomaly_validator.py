"""
Tests for Anomaly Detection & Alerts — anomaly/anomaly_validator.py.

Covers outlier detection using IQR and z-score methods, distribution checks,
and threshold-based alerting for financial, product, and review data anomalies.
"""
import os
import sys
import types
import importlib.util
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import MagicMock, patch

import pandas as pd
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

    def add(self, r: CheckResult):
        self.results.append(r)
        if not r.passed and r.severity == Severity.CRITICAL:
            self.passed = False
        if not r.passed and r.severity == Severity.WARNING:
            self.has_warnings = True

    def print_summary(self): pass
    def save(self): pass

_vc = types.ModuleType("savviocore.validation.validation_config")
_vc.Severity        = Severity
_vc.CheckResult     = CheckResult
_vc.ValidationReport = ValidationReport
sys.modules["savviocore.validation.validation_config"] = _vc

# ---------------------------------------------------------------------------
# Stub anomaly.detectors
# ---------------------------------------------------------------------------
class _FakeDetector:
    def __init__(self, df): self._df = df
    def check_iqr(self, col, multiplier=1.5): return []
    def check_z_score(self, col, threshold=3.0): return []
    def check_rule(self, col, rule_fn, name): return []

_ad = types.ModuleType("anomaly")
_ad_det = types.ModuleType("anomaly.detectors")
_ad_det.AnomalyDetector = _FakeDetector
_ad.detectors = _ad_det
sys.modules.setdefault("anomaly", _ad)
sys.modules["anomaly.detectors"] = _ad_det

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "src", "validation", "anomaly", "anomaly_validator.py"),
        os.path.join(PROJECT_ROOT, "dags", "src", "validation", "anomaly_validator.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        spec = importlib.util.spec_from_file_location("anomaly_validator", fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["anomaly_validator"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ImportError("Could not find anomaly_validator.py. Searched:\n" + "\n".join(candidates))

M = _load()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _csv(tmp_path, **cols):
    """Write a minimal financial CSV and return its path."""
    n = max(len(v) for v in cols.values()) if cols else 1
    default = {
        "monthly_income":   [5000] * n,
        "monthly_expenses": [2000] * n,
        "savings_balance":  [30000] * n,
    }
    default.update(cols)
    p = tmp_path / "fin.csv"
    pd.DataFrame(default).to_csv(p, index=False)
    return str(p)


# =============================================================================
# 1) _report_outliers
# =============================================================================

def test_report_outliers_no_outliers_passes():
    r = M._report_outliers("monthly_income", "iqr", [], "ds")
    assert r.passed is True
    assert r.metric_value == 0

def test_report_outliers_with_outliers_fails():
    r = M._report_outliers("monthly_income", "iqr", [1, 2, 3], "ds")
    assert r.passed is False
    assert r.metric_value == 3

def test_report_outliers_check_name_format():
    r = M._report_outliers("col_x", "zscore", [], "ds")
    assert "col_x" in r.check_name
    assert "zscore" in r.check_name

def test_report_outliers_severity_propagated():
    r = M._report_outliers("col", "iqr", [], "ds", severity=Severity.CRITICAL)
    assert r.severity == Severity.CRITICAL


# =============================================================================
# 2) _quarantine_records
# =============================================================================

def test_quarantine_records_creates_file(tmp_path):
    df = pd.DataFrame([{"monthly_income": 99999}])
    qdir = str(tmp_path / "quarantine")
    M._quarantine_records(df, [0], "test_dataset", quarantine_dir=qdir)
    files = list((tmp_path / "quarantine").glob("*.json"))
    assert len(files) == 1

def test_quarantine_records_empty_indices_skips(tmp_path):
    df = pd.DataFrame([{"monthly_income": 5000}])
    qdir = str(tmp_path / "quarantine")
    M._quarantine_records(df, [], "test_dataset", quarantine_dir=qdir)
    assert not (tmp_path / "quarantine").exists()

def test_quarantine_records_file_contains_correct_rows(tmp_path):
    df = pd.DataFrame([{"income": 100}, {"income": 200}, {"income": 999}])
    qdir = str(tmp_path / "q")
    M._quarantine_records(df, [2], "test", quarantine_dir=qdir)
    import json
    f = list((tmp_path / "q").glob("*.json"))[0]
    rows = [json.loads(l) for l in f.read_text().strip().split("\n")]
    assert rows[0]["income"] == 999


# =============================================================================
# 3) _raw_financial_anomalies
# =============================================================================

def test_raw_financial_anomalies_file_not_found():
    results = M._raw_financial_anomalies("/nonexistent/path.csv")
    assert len(results) == 1
    assert results[0].passed is True   # Tier 1 never halts
    assert results[0].stage == "raw_anomaly"

def test_raw_financial_anomalies_returns_results(tmp_path):
    path = _csv(tmp_path, monthly_income_usd=[5000]*5,
                savings_usd=[10000]*5, monthly_expenses_usd=[2000]*5)
    results = M._raw_financial_anomalies(path)
    assert len(results) >= 3
    assert all(r.stage == "raw_anomaly" for r in results)

def test_raw_financial_anomalies_all_info_severity(tmp_path):
    path = _csv(tmp_path, monthly_income_usd=[5000]*3,
                savings_usd=[10000]*3, monthly_expenses_usd=[2000]*3)
    results = M._raw_financial_anomalies(path)
    assert all(r.severity == Severity.INFO for r in results)

def test_raw_financial_anomalies_with_outliers_still_passes(tmp_path):
    """Tier 1 results are INFO-only; even outliers should not cause passed=False."""
    path = _csv(tmp_path, monthly_income_usd=[5000]*3,
                savings_usd=[10000]*3, monthly_expenses_usd=[2000]*3)

    class _DetectorWithOutliers(_FakeDetector):
        def check_iqr(self, col, **kw): return [0, 1]
        def check_z_score(self, col, **kw): return [0]

    with patch.object(M, "AnomalyDetector", _DetectorWithOutliers):
        results = M._raw_financial_anomalies(path)
    # Tier 1: all INFO, pipeline never halts
    assert all(r.severity == Severity.INFO for r in results)


# =============================================================================
# 4) _featured_financial_anomalies
# =============================================================================

def test_featured_financial_anomalies_file_not_found():
    results = M._featured_financial_anomalies("/nonexistent/path.csv")
    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].severity == Severity.CRITICAL

def test_featured_financial_anomalies_returns_results(tmp_path):
    p = tmp_path / "featured.csv"
    pd.DataFrame([{
        "monthly_income": 5000, "monthly_expenses": 2000,
        "savings_balance": 30000, "discretionary_income": 2500,
        "debt_to_income_ratio": 0.1,
    }]).to_csv(p, index=False)
    results = M._featured_financial_anomalies(str(p))
    assert len(results) >= 1
    assert all(r.stage == "anomaly" for r in results)

def test_featured_financial_anomalies_expenses_2x_income_flagged(tmp_path):
    """Row where expenses > 2x income should be detected as outlier."""
    p = tmp_path / "featured.csv"
    pd.DataFrame([
        {"monthly_income": 1000, "monthly_expenses": 5000, "savings_balance": 0},
        {"monthly_income": 5000, "monthly_expenses": 2000, "savings_balance": 10000},
    ]).to_csv(p, index=False)
    results = M._featured_financial_anomalies(str(p))
    ratio_result = next((r for r in results if "expenses_2x" in r.check_name), None)
    assert ratio_result is not None
    assert ratio_result.passed is False
    assert ratio_result.metric_value == 1


# =============================================================================
# 5) run_raw_anomaly_validation
# =============================================================================

def test_run_raw_anomaly_validation_returns_report(tmp_path):
    path = _csv(tmp_path, monthly_income_usd=[5000]*3,
                savings_usd=[10000]*3, monthly_expenses_usd=[2000]*3)
    report = M.run_raw_anomaly_validation(financial_path=path)
    assert isinstance(report, ValidationReport)
    assert report.stage == "raw_anomaly"

def test_run_raw_anomaly_validation_missing_file_still_returns_report():
    report = M.run_raw_anomaly_validation(financial_path="/nonexistent.csv")
    assert isinstance(report, ValidationReport)
    assert report.passed is True  # Tier 1 never halts


# =============================================================================
# 6) run_anomaly_validation
# =============================================================================

def test_run_anomaly_validation_returns_report(tmp_path):
    p = tmp_path / "featured.csv"
    pd.DataFrame([{"monthly_income": 5000, "monthly_expenses": 2000,
                   "savings_balance": 30000}]).to_csv(p, index=False)
    report = M.run_anomaly_validation(financial_path=str(p))
    assert isinstance(report, ValidationReport)
    assert report.stage == "anomaly"

def test_run_anomaly_validation_missing_file_fails():
    report = M.run_anomaly_validation(financial_path="/nonexistent.csv")
    assert report.passed is False

def test_run_anomaly_validation_clean_data_passes(tmp_path):
    p = tmp_path / "featured.csv"
    pd.DataFrame([{"monthly_income": 5000, "monthly_expenses": 2000,
                   "savings_balance": 30000}]).to_csv(p, index=False)
    report = M.run_anomaly_validation(financial_path=str(p))
    assert report.passed is True