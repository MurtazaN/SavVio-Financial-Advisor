"""
Tests for Raw Data Validation — validate/raw_validator.py.

Covers required column checks, range validation, duplicate detection, and
cross-referential checks (review ASINs vs product ASINs) for raw financial,
product, and review data.
"""
import os
import sys
import json
import types
import importlib.util
from dataclasses import dataclass, field
from enum import Enum

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
_vc.load_thresholds  = lambda path=None: {
    "min_records_critical": 1,
    "min_records_warning":  1,
    "null_pct_critical":    0.5,
    "null_pct_warning":     0.1,
    "dup_pct_critical":     0.5,
    "dup_pct_warning":      0.1,
}
sys.modules["savviocore.validation.validation_config"] = _vc

# ---------------------------------------------------------------------------
# Stub great_expectations
# ---------------------------------------------------------------------------
class _FakePandasDataset(pd.DataFrame):
    def expect_column_to_exist(self, col):
        return {"success": col in self.columns, "result": {}}

    def expect_column_values_to_be_between(self, col, min_value=None, max_value=None, mostly=1.0):
        if col not in self.columns:
            return {"success": False, "result": {}}
        s = self[col].dropna()
        bad = 0
        if min_value is not None: bad += (s < min_value).sum()
        if max_value is not None: bad += (s > max_value).sum()
        pct_ok = 1 - bad / len(s) if len(s) else 1.0
        return {"success": pct_ok >= mostly, "result": {"unexpected_percent": round((1-pct_ok)*100,2)}}

    def expect_column_values_to_not_be_null(self, col, mostly=1.0):
        if col not in self.columns:
            return {"success": False, "result": {}}
        null_count = self[col].isna().sum()
        pct_ok = 1 - null_count / len(self) if len(self) else 1.0
        return {"success": pct_ok >= mostly, "result": {"unexpected_percent": null_count}}

    def expect_column_value_lengths_to_be_between(self, col, min_value=None, mostly=1.0):
        if col not in self.columns:
            return {"success": False, "result": {}}
        s = self[col].dropna().astype(str)
        bad = (s.str.len() < (min_value or 0)).sum()
        pct_ok = 1 - bad / len(s) if len(s) else 1.0
        return {"success": pct_ok >= mostly, "result": {}}

    def expect_column_values_to_be_in_set(self, col, values, mostly=1.0):
        if col not in self.columns:
            return {"success": False, "result": {}}
        bad = (~self[col].isin(values)).sum()
        pct_ok = 1 - bad / len(self) if len(self) else 1.0
        return {"success": pct_ok >= mostly, "result": {"unexpected_percent": bad}}

    def expect_column_values_to_be_in_type_list(self, col, type_list):
        if col not in self.columns:
            return {"success": False, "result": {}}
        return {"success": True, "result": {}}

def _fake_from_pandas(df): return _FakePandasDataset(df)

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
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "src", "validation", "validate", "raw_validator.py"),
        os.path.join(PROJECT_ROOT, "dags", "src", "validation", "raw_validator.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        spec = importlib.util.spec_from_file_location("raw_validator", fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["raw_validator"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ImportError("Could not find raw_validator.py. Searched:\n" + "\n".join(candidates))

M = _load()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_THRESHOLDS = {
    "min_records_critical": 1, "min_records_warning": 1,
    "null_pct_critical": 0.5, "null_pct_warning": 0.1,
    "dup_pct_critical": 0.5,  "dup_pct_warning": 0.1,
}

def _write_csv(path, df): df.to_csv(path, index=False)

def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def _fin_csv(tmp_path, **overrides):
    base = {"user_id": ["U1","U2"], "monthly_income_usd": [5000.0,4000.0],
            "monthly_expenses_usd": [2000.0,1500.0], "savings_usd": [30000.0,20000.0]}
    base.update(overrides)
    p = tmp_path / "fin.csv"
    _write_csv(p, pd.DataFrame(base))
    return str(p)

def _prod_jsonl(tmp_path, rows=None):
    p = tmp_path / "prod.jsonl"
    rows = rows or [{"parent_asin":"PA1","title":"Widget","price":29.99},
                    {"parent_asin":"PA2","title":"Gadget","price":49.99}]
    _write_jsonl(p, rows)
    return str(p)

def _rev_jsonl(tmp_path, rows=None):
    p = tmp_path / "rev.jsonl"
    rows = rows or [{"rating":4.0,"asin":"PA1","user_id":"U1"},
                    {"rating":5.0,"asin":"PA2","user_id":"U2"}]
    _write_jsonl(p, rows)
    return str(p)


# =============================================================================
# 1) validate_financial_raw
# =============================================================================

def test_fin_raw_required_cols_pass(tmp_path):
    path = _fin_csv(tmp_path)
    results = M.validate_financial_raw(path, _THRESHOLDS)
    col_r = [r for r in results if "fin_col_exists_" in r.check_name and
             r.check_name.split("fin_col_exists_")[1] in M.FINANCIAL_REQUIRED_COLS]
    assert all(r.passed for r in col_r)

def test_fin_raw_missing_required_col_fails(tmp_path):
    p = tmp_path / "fin.csv"
    _write_csv(p, pd.DataFrame({"user_id": ["U1"]}))
    results = M.validate_financial_raw(str(p), _THRESHOLDS)
    fails = [r for r in results if "fin_col_exists_monthly_income_usd" in r.check_name and not r.passed]
    assert len(fails) >= 1

def test_fin_raw_negative_income_warns(tmp_path):
    path = _fin_csv(tmp_path, monthly_income_usd=[-100.0, 4000.0])
    results = M.validate_financial_raw(path, _THRESHOLDS)
    r = next((x for x in results if "income_non_negative" in x.check_name), None)
    assert r is not None and not r.passed

def test_fin_raw_duplicate_user_ids_flagged(tmp_path):
    path = _fin_csv(tmp_path, user_id=["U1", "U1"])
    results = M.validate_financial_raw(path, _THRESHOLDS)
    r = next((x for x in results if "duplicate_user_ids" in x.check_name), None)
    assert r is not None and not r.passed

def test_fin_raw_no_duplicates_pass(tmp_path):
    path = _fin_csv(tmp_path)
    results = M.validate_financial_raw(path, _THRESHOLDS)
    r = next((x for x in results if "duplicate_user_ids" in x.check_name), None)
    assert r is not None and r.passed

def test_fin_raw_row_count_check(tmp_path):
    path = _fin_csv(tmp_path)
    thresh = {**_THRESHOLDS, "min_records_critical": 100}
    results = M.validate_financial_raw(path, thresh)
    r = next((x for x in results if "row_count_critical" in x.check_name), None)
    assert r is not None and not r.passed

def test_fin_raw_credit_score_range(tmp_path):
    path = _fin_csv(tmp_path, credit_score=[750, 800])
    results = M.validate_financial_raw(path, _THRESHOLDS)
    r = next((x for x in results if "credit_score_range" in x.check_name), None)
    assert r is not None and r.passed


# =============================================================================
# 2) validate_products_raw
# =============================================================================

def test_prod_raw_required_cols_pass(tmp_path):
    path = _prod_jsonl(tmp_path)
    results = M.validate_products_raw(path, _THRESHOLDS)
    col_r = [r for r in results if "prod_col_exists_" in r.check_name and
             r.check_name.split("prod_col_exists_")[1] in M.PRODUCT_REQUIRED_COLS]
    assert all(r.passed for r in col_r)

def test_prod_raw_missing_required_col_fails(tmp_path):
    path = _prod_jsonl(tmp_path, rows=[{"parent_asin": "PA1"}])
    results = M.validate_products_raw(path, _THRESHOLDS)
    fails = [r for r in results if "prod_col_exists_price" in r.check_name and not r.passed]
    assert len(fails) >= 1

def test_prod_raw_price_range_pass(tmp_path):
    path = _prod_jsonl(tmp_path)
    results = M.validate_products_raw(path, _THRESHOLDS)
    r = next((x for x in results if "price_positive" in x.check_name), None)
    assert r is not None and r.passed

def test_prod_raw_duplicate_asins_flagged(tmp_path):
    path = _prod_jsonl(tmp_path, rows=[
        {"parent_asin":"PA1","title":"A","price":10},
        {"parent_asin":"PA1","title":"B","price":20},
    ])
    results = M.validate_products_raw(path, _THRESHOLDS)
    r = next((x for x in results if "duplicate_asins" in x.check_name), None)
    assert r is not None and not r.passed

def test_prod_raw_avg_rating_range(tmp_path):
    path = _prod_jsonl(tmp_path, rows=[
        {"parent_asin":"PA1","title":"A","price":10,"average_rating":4.5},
    ])
    results = M.validate_products_raw(path, _THRESHOLDS)
    r = next((x for x in results if "avg_rating_range" in x.check_name), None)
    assert r is not None and r.passed


# =============================================================================
# 3) validate_reviews_raw
# =============================================================================

def test_rev_raw_required_cols_pass(tmp_path):
    path = _rev_jsonl(tmp_path)
    results = M.validate_reviews_raw(path, _THRESHOLDS)
    col_r = [r for r in results if "rev_col_exists_" in r.check_name and
             r.check_name.split("rev_col_exists_")[1] in M.REVIEW_REQUIRED_COLS]
    assert all(r.passed for r in col_r)

def test_rev_raw_rating_range_pass(tmp_path):
    path = _rev_jsonl(tmp_path)
    results = M.validate_reviews_raw(path, _THRESHOLDS)
    r = next((x for x in results if "rating_range" in x.check_name), None)
    assert r is not None and r.passed

def test_rev_raw_invalid_rating_fails(tmp_path):
    path = _rev_jsonl(tmp_path, rows=[
        {"rating": 6.0, "asin": "A1", "user_id": "U1"},
    ])
    results = M.validate_reviews_raw(path, _THRESHOLDS)
    r = next((x for x in results if "rating_range" in x.check_name), None)
    assert r is not None and not r.passed

def test_rev_raw_duplicate_user_asin_flagged(tmp_path):
    path = _rev_jsonl(tmp_path, rows=[
        {"rating":4.0,"asin":"A1","user_id":"U1"},
        {"rating":5.0,"asin":"A1","user_id":"U1"},  # duplicate
    ])
    results = M.validate_reviews_raw(path, _THRESHOLDS)
    r = next((x for x in results if "duplicate_user_asin" in x.check_name), None)
    assert r is not None and not r.passed


# =============================================================================
# 4) validate_cross_references
# =============================================================================

def test_cross_ref_all_asins_match(tmp_path):
    prod = _prod_jsonl(tmp_path)
    rev  = _rev_jsonl(tmp_path)  # ASINs PA1, PA2 match products
    results = M.validate_cross_references(prod, rev)
    assert len(results) == 1
    assert results[0].passed

def test_cross_ref_orphan_asins_flagged(tmp_path):
    prod = _prod_jsonl(tmp_path, rows=[{"parent_asin":"PA1","title":"A","price":10}])
    rev  = _rev_jsonl(tmp_path, rows=[
        {"rating":4.0,"asin":"PA1","user_id":"U1"},
        {"rating":5.0,"asin":"ORPHAN","user_id":"U2"},  # not in products
    ])
    results = M.validate_cross_references(prod, rev)
    assert len(results) == 1
    assert not results[0].passed


# =============================================================================
# 5) run_raw_validation (integration)
# =============================================================================

def test_run_raw_validation_returns_report(tmp_path):
    fin  = _fin_csv(tmp_path)
    prod = _prod_jsonl(tmp_path)
    rev  = _rev_jsonl(tmp_path)
    report = M.run_raw_validation(fin, prod, rev, threshold_config=None)
    assert isinstance(report, ValidationReport)
    assert report.stage == "raw"
    assert len(report.results) > 0
    assert report.passed