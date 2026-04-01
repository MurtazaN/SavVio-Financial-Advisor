"""
Tests for Anomaly Detection & Alerts — anomaly/detectors.py.

Covers the pluggable anomaly detection functions: IQR-based outlier detection,
z-score outlier detection, and isolation-forest-based anomaly scoring.
"""
import os
import sys
import importlib.util

import pandas as pd
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path constants  (sys.path set up by conftest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "src", "validation", "anomaly", "detectors.py"),
        os.path.join(PROJECT_ROOT, "dags", "src", "validation", "detectors.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        spec = importlib.util.spec_from_file_location("detectors", fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["detectors"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ImportError("Could not find detectors.py. Searched:\n" + "\n".join(candidates))

M = _load()
AnomalyDetector = M.AnomalyDetector

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _det(data: dict) -> AnomalyDetector:
    return AnomalyDetector(pd.DataFrame(data))


# =============================================================================
# 1) check_z_score
# =============================================================================

def test_zscore_no_outliers():
    det = _det({"val": [10, 11, 10, 12, 10, 11]})
    assert det.check_z_score("val", threshold=3.0) == []

def test_zscore_detects_extreme_outlier():
    det = _det({"val": [10, 10, 10, 10, 10, 1000]})
    outliers = det.check_z_score("val", threshold=2.0)
    assert len(outliers) >= 1

def test_zscore_missing_column_returns_empty():
    det = _det({"val": [1, 2, 3]})
    assert det.check_z_score("nonexistent") == []

def test_zscore_single_row_returns_empty():
    det = _det({"val": [42]})
    assert det.check_z_score("val") == []

def test_zscore_constant_column_returns_empty():
    """All same values → std=0, no z-score possible."""
    det = _det({"val": [5, 5, 5, 5, 5]})
    assert det.check_z_score("val") == []

def test_zscore_ignores_nan():
    """NaN rows are dropped before z-score; valid outliers still detected."""
    det = _det({"val": [10, 10, None, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 99999]})
    outliers = det.check_z_score("val", threshold=2.0)
    assert len(outliers) >= 1

def test_zscore_returns_list_of_ints():
    det = _det({"val": [10, 10, 10, 10, 1000]})
    outliers = det.check_z_score("val", threshold=1.0)
    assert isinstance(outliers, list)

def test_zscore_threshold_affects_result():
    det = _det({"val": [10, 10, 10, 10, 50]})
    strict = det.check_z_score("val", threshold=1.0)
    lenient = det.check_z_score("val", threshold=5.0)
    assert len(strict) >= len(lenient)


# =============================================================================
# 2) check_iqr
# =============================================================================

def test_iqr_no_outliers():
    det = _det({"val": [10, 11, 10, 12, 10, 11]})
    assert det.check_iqr("val", multiplier=1.5) == []

def test_iqr_detects_outlier():
    det = _det({"val": [10, 10, 10, 10, 10, 1000]})
    outliers = det.check_iqr("val", multiplier=1.5)
    assert len(outliers) >= 1

def test_iqr_missing_column_returns_empty():
    det = _det({"val": [1, 2, 3]})
    assert det.check_iqr("nonexistent") == []

def test_iqr_single_row_returns_empty():
    det = _det({"val": [42]})
    assert det.check_iqr("val") == []

def test_iqr_ignores_nan():
    det = _det({"val": [10, 10, 10, None, 1000]})
    outliers = det.check_iqr("val", multiplier=1.5)
    assert len(outliers) >= 1

def test_iqr_multiplier_affects_result():
    det = _det({"val": [10, 10, 10, 10, 40]})
    strict = det.check_iqr("val", multiplier=0.5)
    lenient = det.check_iqr("val", multiplier=10.0)
    assert len(strict) >= len(lenient)

def test_iqr_returns_list():
    det = _det({"val": [1, 2, 3, 4, 100]})
    assert isinstance(det.check_iqr("val"), list)

def test_iqr_both_tails_detected():
    """Very low and very high values should both be flagged."""
    det = _det({"val": [-1000, 10, 10, 10, 10, 10, 1000]})
    outliers = det.check_iqr("val", multiplier=1.5)
    assert len(outliers) == 2


# =============================================================================
# 3) check_rule
# =============================================================================

def test_check_rule_all_pass():
    det = _det({"income": [100, 200, 300]})
    result = det.check_rule("income", lambda x: x > 0, "positive")
    assert result == []

def test_check_rule_flags_violations():
    det = _det({"income": [100, -50, 200, 0]})
    result = det.check_rule("income", lambda x: x > 0, "positive")
    assert len(result) == 2

def test_check_rule_missing_column_returns_empty():
    det = _det({"income": [100, 200]})
    result = det.check_rule("nonexistent", lambda x: x > 0, "positive")
    assert result == []

def test_check_rule_returns_correct_indices():
    det = _det({"val": [10, -1, 20, -2]})
    result = det.check_rule("val", lambda x: x > 0, "positive")
    assert set(result) == {1, 3}

def test_check_rule_all_fail():
    det = _det({"val": [-1, -2, -3]})
    result = det.check_rule("val", lambda x: x > 0, "positive")
    assert len(result) == 3

def test_check_rule_custom_lambda():
    """Rule: value must be between 0 and 1 (ratio)."""
    det = _det({"ratio": [0.1, 0.5, 1.5, -0.1]})
    result = det.check_rule("ratio", lambda x: 0 <= x <= 1, "valid_ratio")
    assert len(result) == 2