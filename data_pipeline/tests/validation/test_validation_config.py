"""
Tests for shared validation infrastructure — validation_config.py.

Covers Severity enum ordering, CheckResult construction and tagging,
ValidationReport (passed/warnings/summary/save), and threshold loading
(defaults, file-based overrides, merge logic).
"""
import os
import sys
import json
import importlib.util

import pytest

# ---------------------------------------------------------------------------
# Path constants  (sys.path set up by conftest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
import importlib.util
import savviocore

def _load():
    fpath = os.path.join(savviocore.__path__[0], "validation", "validation_config.py")
    spec = importlib.util.spec_from_file_location("real_validation_config", fpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

M = _load()
Severity        = M.Severity
CheckResult     = M.CheckResult
ValidationReport = M.ValidationReport
load_thresholds = M.load_thresholds


# =============================================================================
# 1) Severity
# =============================================================================

def test_severity_ordering():
    assert Severity.INFO < Severity.WARNING < Severity.CRITICAL


# =============================================================================
# 2) CheckResult
# =============================================================================

@pytest.mark.parametrize("severity,label", [
    (Severity.INFO, "INFO"),
    (Severity.WARNING, "WARNING"),
    (Severity.CRITICAL, "CRITICAL"),
])
def test_check_result_tag_contains_severity(severity, label):
    r = CheckResult("test", False, severity, "ds", "raw")
    assert label in r.tag


# =============================================================================
# 3) ValidationReport.add / passed / has_warnings
# =============================================================================

def test_report_passed_with_no_results():
    r = ValidationReport(stage="raw")
    assert r.passed is True

def test_report_passed_with_only_passing_checks():
    r = ValidationReport(stage="raw")
    r.add(CheckResult("c1", True, Severity.CRITICAL, "ds", "raw"))
    assert r.passed is True

def test_report_fails_on_critical():
    r = ValidationReport(stage="raw")
    r.add(CheckResult("c1", False, Severity.CRITICAL, "ds", "raw"))
    assert r.passed is False

def test_report_passes_despite_warning():
    r = ValidationReport(stage="raw")
    r.add(CheckResult("w1", False, Severity.WARNING, "ds", "raw"))
    assert r.passed is True

def test_report_has_warnings_true():
    r = ValidationReport(stage="raw")
    r.add(CheckResult("w1", False, Severity.WARNING, "ds", "raw"))
    assert r.has_warnings is True

def test_report_has_warnings_false_when_only_info():
    r = ValidationReport(stage="raw")
    r.add(CheckResult("i1", False, Severity.INFO, "ds", "raw"))
    assert r.has_warnings is False

def test_report_has_warnings_false_when_all_pass():
    r = ValidationReport(stage="raw")
    r.add(CheckResult("c1", True, Severity.WARNING, "ds", "raw"))
    assert r.has_warnings is False


# =============================================================================
# 4) ValidationReport.summary
# =============================================================================

def test_summary_continue():
    r = ValidationReport(stage="raw")
    r.add(CheckResult("c1", True, Severity.INFO, "ds", "raw"))
    assert r.summary["pipeline_action"] == "CONTINUE"

def test_summary_alert():
    r = ValidationReport(stage="raw")
    r.add(CheckResult("w1", False, Severity.WARNING, "ds", "raw"))
    assert r.summary["pipeline_action"] == "ALERT"

def test_summary_halt():
    r = ValidationReport(stage="raw")
    r.add(CheckResult("c1", False, Severity.CRITICAL, "ds", "raw"))
    assert r.summary["pipeline_action"] == "HALT"

def test_summary_halt_takes_priority_over_alert():
    r = ValidationReport(stage="raw")
    r.add(CheckResult("w1", False, Severity.WARNING, "ds", "raw"))
    r.add(CheckResult("c1", False, Severity.CRITICAL, "ds", "raw"))
    assert r.summary["pipeline_action"] == "HALT"

def test_summary_counts():
    r = ValidationReport(stage="raw")
    r.add(CheckResult("p1", True,  Severity.INFO,     "ds", "raw"))
    r.add(CheckResult("i1", False, Severity.INFO,     "ds", "raw"))
    r.add(CheckResult("w1", False, Severity.WARNING,  "ds", "raw"))
    r.add(CheckResult("c1", False, Severity.CRITICAL, "ds", "raw"))
    s = r.summary
    assert s["total_checks"]    == 4
    assert s["passed"]          == 1
    assert s["failed_info"]     == 1
    assert s["failed_warning"]  == 1
    assert s["failed_critical"] == 1

def test_summary_stage():
    r = ValidationReport(stage="features")
    assert r.summary["stage"] == "features"

def test_summary_timestamp_present():
    r = ValidationReport(stage="raw")
    assert "timestamp" in r.summary


# =============================================================================
# 5) ValidationReport.save
# =============================================================================

def test_save_creates_file(tmp_path):
    r = ValidationReport(stage="raw")
    r.add(CheckResult("c1", True, Severity.INFO, "ds", "raw"))
    path = r.save(log_dir=str(tmp_path / "logs"))
    assert os.path.exists(path)

def test_save_file_contains_valid_json(tmp_path):
    r = ValidationReport(stage="processed")
    r.add(CheckResult("c1", True, Severity.WARNING, "ds", "processed"))
    path = r.save(log_dir=str(tmp_path / "logs"))
    with open(path) as f:
        data = json.load(f)
    assert "summary" in data
    assert "checks" in data

def test_save_summary_matches_report(tmp_path):
    r = ValidationReport(stage="features")
    r.add(CheckResult("c1", False, Severity.CRITICAL, "ds", "features"))
    path = r.save(log_dir=str(tmp_path))
    with open(path) as f:
        data = json.load(f)
    assert data["summary"]["pipeline_action"] == "HALT"

def test_save_checks_contain_all_results(tmp_path):
    r = ValidationReport(stage="raw")
    r.add(CheckResult("c1", True,  Severity.INFO,    "ds", "raw"))
    r.add(CheckResult("c2", False, Severity.WARNING, "ds", "raw"))
    path = r.save(log_dir=str(tmp_path))
    with open(path) as f:
        data = json.load(f)
    assert len(data["checks"]) == 2


# =============================================================================
# 6) load_thresholds
# =============================================================================

def test_load_thresholds_returns_defaults_when_no_file():
    t = load_thresholds(None)
    assert isinstance(t, dict)
    assert "null_pct_critical" in t
    assert "min_records_critical" in t

def test_load_thresholds_from_file(tmp_path):
    cfg = tmp_path / "thresholds.json"
    cfg.write_text(json.dumps({"null_pct_critical": 0.99}))
    t = load_thresholds(str(cfg))
    assert t["null_pct_critical"] == pytest.approx(0.99)

def test_load_thresholds_merges_with_defaults(tmp_path):
    cfg = tmp_path / "thresholds.json"
    cfg.write_text(json.dumps({"null_pct_critical": 0.99}))
    t = load_thresholds(str(cfg))
    assert "min_records_critical" in t  # default still present

def test_load_thresholds_nonexistent_file_returns_defaults():
    t = load_thresholds("/nonexistent/path.json")
    assert t == M.DEFAULT_THRESHOLDS