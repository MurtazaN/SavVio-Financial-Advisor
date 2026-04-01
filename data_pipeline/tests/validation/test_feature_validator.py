"""
Tests for Feature Validation — validate/feature_validator.py.

Covers feature-level checks: ratio range validation (0–1 for ratios,
≥0 for emergency_fund_months), rating_variance non-negative, NaN detection
in computed features, and cross-dataset consistency between product features
and review ratings.
"""
import os
import sys
import types
import importlib.util
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import MagicMock, patch

import numpy as np
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

    def add(self, r):
        self.results.append(r)
        if not r.passed and r.severity == Severity.CRITICAL:
            self.passed = False
        if not r.passed and r.severity == Severity.WARNING:
            self.has_warnings = True

    def print_summary(self): pass
    def save(self): pass

_vc = types.ModuleType("savviocore.validation.validation_config")
_vc.Severity         = Severity
_vc.CheckResult      = CheckResult
_vc.ValidationReport = ValidationReport
_vc.load_thresholds  = lambda path=None: {}
sys.modules["savviocore.validation.validation_config"] = _vc

# ---------------------------------------------------------------------------
# Stub great_expectations
# ---------------------------------------------------------------------------
class _FakePandasDataset(pd.DataFrame):
    """Minimal PandasDataset that wraps a real DataFrame and fakes GE methods."""

    def expect_column_to_exist(self, col):
        return {"success": col in self.columns, "result": {}}

    def expect_column_values_to_be_between(self, col, min_value=None, max_value=None, mostly=1.0):
        if col not in self.columns:
            return {"success": False, "result": {}}
        s = self[col].dropna()
        if min_value is not None:
            bad = (s < min_value).sum()
        else:
            bad = 0
        if max_value is not None:
            bad += (s > max_value).sum()
        pct_ok = 1 - bad / len(s) if len(s) else 1.0
        return {"success": pct_ok >= mostly,
                "result": {"unexpected_percent": round((1 - pct_ok) * 100, 2)}}

    def expect_column_values_to_not_be_null(self, col):
        if col not in self.columns:
            return {"success": False, "result": {}}
        null_count = self[col].isna().sum()
        return {"success": null_count == 0, "result": {"unexpected_percent": null_count}}


def _fake_from_pandas(df):
    obj = _FakePandasDataset(df)
    return obj

_gx = types.ModuleType("great_expectations")
_gx.from_pandas = _fake_from_pandas
_gx_ds = types.ModuleType("great_expectations.dataset")
_gx_ds.PandasDataset = _FakePandasDataset
_gx_ds_pd = types.ModuleType("great_expectations.dataset.pandas_dataset")
_gx_ds_pd.PandasDataset = _FakePandasDataset

sys.modules["great_expectations"] = _gx
sys.modules["great_expectations.dataset"] = _gx_ds
sys.modules["great_expectations.dataset.pandas_dataset"] = _gx_ds_pd

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
import importlib.util
import savviocore

def _load():
    fpath = os.path.join(savviocore.__path__[0], "validation", "feature_validator.py")
    spec = importlib.util.spec_from_file_location("real_feature_validator", fpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

M = _load()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gdf(**cols):
    n = max(len(v) for v in cols.values()) if cols else 1
    return _FakePandasDataset(pd.DataFrame({k: v for k, v in cols.items()}))

def _fin_gdf(**overrides):
    base = {
        "discretionary_income":         [2500.0] * 5,
        "debt_to_income_ratio":          [0.1]   * 5,
        "saving_to_income_ratio":        [0.2]   * 5,
        "monthly_expense_burden_ratio":  [0.5]   * 5,
        "emergency_fund_months":         [6.0]   * 5,
    }
    base.update(overrides)
    return _FakePandasDataset(pd.DataFrame(base))


# =============================================================================
# 1) _no_nan_inf
# =============================================================================

def test_no_nan_inf_clean_column():
    gdf = _gdf(val=[1.0, 2.0, 3.0])
    results = M._no_nan_inf(gdf, "val", "ds")
    assert all(r.passed for r in results)

def test_no_nan_inf_with_nan():
    gdf = _gdf(val=[1.0, float("nan"), 3.0])
    results = M._no_nan_inf(gdf, "val", "ds")
    nan_r = next(r for r in results if "nan" in r.check_name)
    assert not nan_r.passed

def test_no_nan_inf_with_inf():
    gdf = _gdf(val=[1.0, float("inf"), 3.0])
    results = M._no_nan_inf(gdf, "val", "ds")
    inf_r = next(r for r in results if "inf" in r.check_name)
    assert not inf_r.passed

def test_no_nan_inf_missing_column_returns_empty():
    gdf = _gdf(val=[1.0, 2.0])
    results = M._no_nan_inf(gdf, "nonexistent", "ds")
    assert results == []

def test_no_nan_inf_returns_two_results():
    gdf = _gdf(val=[1.0, 2.0, 3.0])
    results = M._no_nan_inf(gdf, "val", "ds")
    assert len(results) == 2


# =============================================================================
# 2) validate_financial_features
# =============================================================================

def test_financial_features_all_exist_pass():
    gdf = _fin_gdf()
    results = M.validate_financial_features(gdf, {})
    exist_results = [r for r in results if "feat_exists" in r.check_name]
    assert all(r.passed for r in exist_results)

def test_financial_features_missing_col_fails():
    gdf = _FakePandasDataset(pd.DataFrame({
        "discretionary_income": [100.0],
        # missing other required columns
    }))
    results = M.validate_financial_features(gdf, {})
    exist_fails = [r for r in results if "feat_exists" in r.check_name and not r.passed]
    assert len(exist_fails) >= 1

def test_financial_features_no_nan_checks_present():
    gdf = _fin_gdf()
    results = M.validate_financial_features(gdf, {})
    nan_results = [r for r in results if "no_nan" in r.check_name]
    assert len(nan_results) >= 1

def test_financial_features_dti_range_check():
    gdf = _fin_gdf(debt_to_income_ratio=[0.1] * 5)
    results = M.validate_financial_features(gdf, {})
    dti_r = next((r for r in results if "dti_range" in r.check_name), None)
    assert dti_r is not None
    assert dti_r.passed

def test_financial_features_all_negative_discretionary_warns():
    gdf = _fin_gdf(discretionary_income=[-100.0] * 5)
    results = M.validate_financial_features(gdf, {})
    neg_r = next((r for r in results if "all_negative" in r.check_name), None)
    assert neg_r is not None
    assert not neg_r.passed


# =============================================================================
# 3) validate_review_features
# =============================================================================

def test_review_features_rating_variance_exists():
    gdf = _gdf(rating_variance=[0.5, 1.0, 0.0])
    results = M.validate_review_features(gdf, {})
    exist_r = next(r for r in results if "feat_exists_rating_variance" in r.check_name)
    assert exist_r.passed

def test_review_features_missing_rating_variance_fails():
    gdf = _gdf(other_col=[1, 2, 3])
    results = M.validate_review_features(gdf, {})
    exist_r = next(r for r in results if "feat_exists_rating_variance" in r.check_name)
    assert not exist_r.passed

def test_review_features_negative_variance_fails():
    gdf = _gdf(rating_variance=[-0.1, 0.5, 1.0])
    results = M.validate_review_features(gdf, {})
    range_r = next((r for r in results if "non_negative" in r.check_name), None)
    assert range_r is not None
    assert not range_r.passed

def test_review_features_no_nan_check_present():
    gdf = _gdf(rating_variance=[0.0, 1.0, 0.5])
    results = M.validate_review_features(gdf, {})
    nan_results = [r for r in results if "no_nan" in r.check_name]
    assert len(nan_results) >= 1


# =============================================================================
# 4) validate_affordability_features
# =============================================================================

def test_affordability_features_exist_check():
    gdf = _gdf(
        price_to_income_ratio=[0.1, 0.2],
        affordability_score=[100.0, 200.0],
        residual_utility_score=[0.5, 0.8],
    )
    results = M.validate_affordability_features(gdf, {})
    exist_results = [r for r in results if "feat_exists" in r.check_name]
    assert all(r.passed for r in exist_results)

def test_affordability_rus_all_zero_warns():
    gdf = _gdf(
        price_to_income_ratio=[0.1],
        affordability_score=[100.0],
        residual_utility_score=[0.0],
    )
    results = M.validate_affordability_features(gdf, {})
    rus_r = next((r for r in results if "rus_not_all_zero" in r.check_name), None)
    assert rus_r is not None
    assert not rus_r.passed


# =============================================================================
# 5) validate_formula_spot_checks
# =============================================================================

def test_formula_discretionary_income_correct():
    gdf = _FakePandasDataset(pd.DataFrame({
        "discretionary_income":  [3000.0, 2000.0],
        "income_usd":            [5000.0, 4000.0],
        "total_fixed_expenses":  [2000.0, 2000.0],
    }))
    results = M.validate_formula_spot_checks(gdf)
    r = next((x for x in results if "discretionary" in x.check_name), None)
    assert r is not None
    assert r.passed

def test_formula_discretionary_income_wrong():
    gdf = _FakePandasDataset(pd.DataFrame({
        "discretionary_income":  [9999.0, 9999.0],  # wrong
        "income_usd":            [5000.0, 4000.0],
        "total_fixed_expenses":  [2000.0, 2000.0],
    }))
    results = M.validate_formula_spot_checks(gdf)
    r = next((x for x in results if "discretionary" in x.check_name), None)
    assert r is not None
    assert not r.passed

def test_formula_no_required_cols_returns_empty():
    gdf = _FakePandasDataset(pd.DataFrame({"other": [1, 2, 3]}))
    results = M.validate_formula_spot_checks(gdf)
    assert results == []


# =============================================================================
# 6) run_feature_validation (integration)
# =============================================================================

def test_run_feature_validation_missing_files_fails():
    report = M.run_feature_validation(
        financial_path="/nonexistent/fin.csv",
        reviews_path="/nonexistent/rev.csv",
        threshold_config=None,
    )
    assert isinstance(report, ValidationReport)
    assert not report.passed

def test_run_feature_validation_valid_files_passes(tmp_path):
    fin = tmp_path / "fin.csv"
    rev = tmp_path / "rev.csv"
    pd.DataFrame({
        "discretionary_income":          [2500.0],
        "debt_to_income_ratio":          [0.1],
        "saving_to_income_ratio":        [0.2],
        "monthly_expense_burden_ratio":  [0.5],
        "emergency_fund_months":         [6.0],
    }).to_csv(fin, index=False)
    pd.DataFrame({"rating_variance": [0.5]}).to_csv(rev, index=False)

    report = M.run_feature_validation(
        financial_path=str(fin),
        reviews_path=str(rev),
        threshold_config=None,
    )
    assert isinstance(report, ValidationReport)
    assert report.stage == "features"
    assert report.passed
