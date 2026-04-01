"""
Tests for Processed Data Validation — validate/processed_validator.py.

Covers post-preprocessing checks: required columns, null detection, value ranges,
duplicate detection, record loss thresholds, and rating validation for processed
financial, product, and review data.
"""
import os
import sys
import json
import types
import importlib.util
from dataclasses import dataclass, field
from enum import Enum

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
        return {"success": pct_ok >= mostly,
                "result": {"unexpected_percent": round((1 - pct_ok) * 100, 2)}}

    def expect_column_values_to_not_be_null(self, col):
        if col not in self.columns:
            return {"success": False, "result": {}}
        null_count = self[col].isna().sum()
        return {"success": null_count == 0, "result": {"unexpected_percent": null_count}}

    def expect_column_value_lengths_to_be_between(self, col, min_value=None, mostly=1.0):
        if col not in self.columns:
            return {"success": False, "result": {}}
        s = self[col].dropna().astype(str)
        bad = (s.str.len() < (min_value or 0)).sum() if min_value else 0
        pct_ok = 1 - bad / len(s) if len(s) else 1.0
        return {"success": pct_ok >= mostly, "result": {}}

    def expect_column_values_to_be_in_set(self, col, values):
        if col not in self.columns:
            return {"success": False, "result": {}}
        bad = (~self[col].isin(values)).sum()
        return {"success": bad == 0, "result": {"unexpected_percent": bad}}

def _fake_from_pandas(df):
    return _FakePandasDataset(df)

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
        os.path.join(PROJECT_ROOT, "dags", "src", "validation", "validate", "processed_validator.py"),
        os.path.join(PROJECT_ROOT, "dags", "src", "validation", "processed_validator.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        spec = importlib.util.spec_from_file_location("processed_validator", fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["processed_validator"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ImportError("Could not find processed_validator.py. Searched:\n" + "\n".join(candidates))

M = _load()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_csv(path, df): df.to_csv(path, index=False)
def _write_jsonl(path, df):
    with open(path, "w") as f:
        for r in df.to_dict(orient="records"):
            f.write(json.dumps(r) + "\n")

def _fin_df(**overrides):
    base = {"user_id": ["U1", "U2"], "monthly_income": [5000.0, 4000.0],
            "monthly_expenses": [2000.0, 1500.0], "savings_balance": [30000.0, 20000.0]}
    base.update(overrides)
    return pd.DataFrame(base)

def _prod_df(**overrides):
    base = {"product_id": ["P1", "P2"], "product_name": ["Widget", "Gadget"],
            "price": [29.99, 49.99]}
    base.update(overrides)
    return pd.DataFrame(base)

def _rev_df(**overrides):
    base = {"user_id": ["U1", "U2"], "asin": ["A1", "A2"], "rating": [4.0, 5.0]}
    base.update(overrides)
    return pd.DataFrame(base)


# =============================================================================
# 1) validate_financial_processed
# =============================================================================

def test_fin_processed_required_cols_pass(tmp_path):
    fin = tmp_path / "fin.csv"
    raw = tmp_path / "raw.csv"
    _write_csv(fin, _fin_df())
    _write_csv(raw, _fin_df())
    results = M.validate_financial_processed(str(fin), str(raw), {})
    col_results = [r for r in results if "fin_proc_col_" in r.check_name]
    assert all(r.passed for r in col_results)

def test_fin_processed_missing_col_fails(tmp_path):
    fin = tmp_path / "fin.csv"
    raw = tmp_path / "raw.csv"
    _write_csv(fin, pd.DataFrame({"user_id": ["U1"]}))
    _write_csv(raw, _fin_df())
    results = M.validate_financial_processed(str(fin), str(raw), {})
    col_fails = [r for r in results if "fin_proc_col_" in r.check_name and not r.passed]
    assert len(col_fails) >= 1

def test_fin_processed_no_nulls_check(tmp_path):
    fin = tmp_path / "fin.csv"
    raw = tmp_path / "raw.csv"
    _write_csv(fin, _fin_df())
    _write_csv(raw, _fin_df())
    results = M.validate_financial_processed(str(fin), str(raw), {})
    null_checks = [r for r in results if "no_nulls" in r.check_name]
    assert len(null_checks) >= 1
    assert all(r.passed for r in null_checks)

def test_fin_processed_negative_income_warns(tmp_path):
    fin = tmp_path / "fin.csv"
    raw = tmp_path / "raw.csv"
    _write_csv(fin, _fin_df(monthly_income=[-100.0, 4000.0]))
    _write_csv(raw, _fin_df())
    results = M.validate_financial_processed(str(fin), str(raw), {})
    neg_r = next((r for r in results if "monthly_income_non_negative" in r.check_name), None)
    assert neg_r is not None
    assert not neg_r.passed

def test_fin_processed_no_duplicates_pass(tmp_path):
    fin = tmp_path / "fin.csv"
    raw = tmp_path / "raw.csv"
    _write_csv(fin, _fin_df())
    _write_csv(raw, _fin_df())
    results = M.validate_financial_processed(str(fin), str(raw), {})
    dup_r = next((r for r in results if "no_duplicates" in r.check_name), None)
    assert dup_r is not None
    assert dup_r.passed

def test_fin_processed_duplicates_flagged(tmp_path):
    fin = tmp_path / "fin.csv"
    raw = tmp_path / "raw.csv"
    df = _fin_df(user_id=["U1", "U1"])  # duplicate
    _write_csv(fin, df)
    _write_csv(raw, _fin_df())
    results = M.validate_financial_processed(str(fin), str(raw), {})
    dup_r = next((r for r in results if "no_duplicates" in r.check_name), None)
    assert dup_r is not None
    assert not dup_r.passed

def test_fin_processed_record_loss_check(tmp_path):
    fin = tmp_path / "fin.csv"
    raw = tmp_path / "raw.csv"
    # processed = 2 rows, raw = 20 rows → 90% loss → should fail
    raw_df = pd.DataFrame({
        "user_id": [f"U{i}" for i in range(20)],
        "monthly_income": [5000.0] * 20,
        "monthly_expenses": [2000.0] * 20,
        "savings_balance": [30000.0] * 20,
    })
    _write_csv(fin, _fin_df())
    _write_csv(raw, raw_df)
    results = M.validate_financial_processed(str(fin), str(raw), {})
    loss_r = next((r for r in results if "record_loss" in r.check_name), None)
    assert loss_r is not None
    assert not loss_r.passed


# =============================================================================
# 2) validate_products_processed
# =============================================================================

def test_prod_processed_required_cols_pass(tmp_path):
    prod = tmp_path / "prod.jsonl"
    raw  = tmp_path / "raw.jsonl"
    _write_jsonl(prod, _prod_df())
    _write_jsonl(raw, _prod_df())
    results = M.validate_products_processed(str(prod), str(raw), {})
    col_results = [r for r in results if "prod_proc_col_" in r.check_name]
    assert all(r.passed for r in col_results)

def test_prod_processed_price_range_pass(tmp_path):
    prod = tmp_path / "prod.jsonl"
    raw  = tmp_path / "raw.jsonl"
    _write_jsonl(prod, _prod_df())
    _write_jsonl(raw, _prod_df())
    results = M.validate_products_processed(str(prod), str(raw), {})
    price_r = next((r for r in results if "price_range" in r.check_name), None)
    assert price_r is not None
    assert price_r.passed

def test_prod_processed_no_duplicates_pass(tmp_path):
    prod = tmp_path / "prod.jsonl"
    raw  = tmp_path / "raw.jsonl"
    _write_jsonl(prod, _prod_df())
    _write_jsonl(raw, _prod_df())
    results = M.validate_products_processed(str(prod), str(raw), {})
    dup_r = next((r for r in results if "no_duplicates" in r.check_name), None)
    assert dup_r is not None
    assert dup_r.passed

def test_prod_processed_duplicate_ids_flagged(tmp_path):
    prod = tmp_path / "prod.jsonl"
    raw  = tmp_path / "raw.jsonl"
    df = _prod_df(product_id=["P1", "P1"])
    _write_jsonl(prod, df)
    _write_jsonl(raw, _prod_df())
    results = M.validate_products_processed(str(prod), str(raw), {})
    dup_r = next((r for r in results if "no_duplicates" in r.check_name), None)
    assert not dup_r.passed


# =============================================================================
# 3) validate_reviews_processed
# =============================================================================

def test_rev_processed_required_cols_pass(tmp_path):
    rev = tmp_path / "rev.jsonl"
    raw = tmp_path / "raw.jsonl"
    _write_jsonl(rev, _rev_df())
    _write_jsonl(raw, _rev_df())
    results = M.validate_reviews_processed(str(rev), str(raw), {})
    col_results = [r for r in results if "rev_proc_col_" in r.check_name]
    assert all(r.passed for r in col_results)

def test_rev_processed_rating_range_pass(tmp_path):
    rev = tmp_path / "rev.jsonl"
    raw = tmp_path / "raw.jsonl"
    _write_jsonl(rev, _rev_df())
    _write_jsonl(raw, _rev_df())
    results = M.validate_reviews_processed(str(rev), str(raw), {})
    rating_r = next((r for r in results if "rating_range" in r.check_name), None)
    assert rating_r is not None
    assert rating_r.passed

def test_rev_processed_invalid_rating_fails(tmp_path):
    rev = tmp_path / "rev.jsonl"
    raw = tmp_path / "raw.jsonl"
    _write_jsonl(rev, _rev_df(rating=[6.0, 0.0]))  # out of 1–5
    _write_jsonl(raw, _rev_df())
    results = M.validate_reviews_processed(str(rev), str(raw), {})
    rating_r = next((r for r in results if "rating_range" in r.check_name), None)
    assert not rating_r.passed

def test_rev_processed_sentiment_valid(tmp_path):
    rev = tmp_path / "rev.jsonl"
    raw = tmp_path / "raw.jsonl"
    df = _rev_df()
    df["sentiment"] = ["positive", "negative"]
    _write_jsonl(rev, df)
    _write_jsonl(raw, _rev_df())
    results = M.validate_reviews_processed(str(rev), str(raw), {})
    sent_r = next((r for r in results if "sentiment_valid" in r.check_name), None)
    assert sent_r is not None
    assert sent_r.passed

def test_rev_processed_no_duplicates_pass(tmp_path):
    rev = tmp_path / "rev.jsonl"
    raw = tmp_path / "raw.jsonl"
    _write_jsonl(rev, _rev_df())
    _write_jsonl(raw, _rev_df())
    results = M.validate_reviews_processed(str(rev), str(raw), {})
    dup_r = next((r for r in results if "no_duplicates" in r.check_name), None)
    assert dup_r is not None
    assert dup_r.passed


# =============================================================================
# 4) run_processed_validation (integration)
# =============================================================================

def test_run_processed_validation_returns_report(tmp_path):
    fin  = tmp_path / "fin.csv";   _write_csv(fin, _fin_df())
    prod = tmp_path / "prod.jsonl"; _write_jsonl(prod, _prod_df())
    rev  = tmp_path / "rev.jsonl";  _write_jsonl(rev, _rev_df())

    report = M.run_processed_validation(
        financial_path=str(fin), products_path=str(prod), reviews_path=str(rev),
        raw_financial=str(fin), raw_products=str(prod), raw_reviews=str(rev),
        threshold_config=None,
    )
    assert isinstance(report, ValidationReport)
    assert report.stage == "processed"
    assert len(report.results) > 0
