"""
Tests for Data Preprocessing — preprocess/financial.py.

Covers deterministic financial data cleaning: column renaming, has_loan
coercion, deduplication, missing value handling, range validation, and
output CSV creation from raw financial CSV input.
"""
import os
import sys
import types
import importlib.util

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path constants  (sys.path set up by conftest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------------
# Stub preprocess.utils
# ---------------------------------------------------------------------------
def _stub_utils():
    utils = types.ModuleType("preprocess.utils")
    utils.ensure_output_dir = lambda path: os.makedirs(os.path.dirname(path), exist_ok=True)
    utils.get_raw_path      = lambda name: os.path.join("data", "raw", name)
    utils.get_processed_path = lambda name, **kw: os.path.join("data", "processed", name)
    utils.setup_logging     = lambda *a, **kw: None
    sys.modules["preprocess.utils"] = utils
    if "preprocess" not in sys.modules:
        sys.modules["preprocess"] = types.ModuleType("preprocess")
    sys.modules["preprocess"].utils = utils

_stub_utils()

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "src", "preprocess", "financial.py"),
        os.path.join(PROJECT_ROOT, "src", "preprocess", "financial.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        spec = importlib.util.spec_from_file_location(
            "preprocess.financial", fpath, submodule_search_locations=[]
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "preprocess"
        sys.modules["preprocess.financial"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ImportError("Could not find preprocess/financial.py. Searched:\n" + "\n".join(candidates))

M = _load()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _base_row(**overrides):
    row = {
        "user_id": "U1",
        "monthly_income_usd": 5000.0,
        "monthly_expenses_usd": 2000.0,
        "savings_usd": 30000.0,
        "has_loan": 1,
        "loan_amount_usd": 10000.0,
        "monthly_emi_usd": 500.0,
        "loan_interest_rate_pct": 5.0,
        "loan_term_months": 24,
        "credit_score": 700,
        "employment_status": "employed",
        "region": "US",
    }
    row.update(overrides)
    return row

def _df(*rows):
    return pd.DataFrame(rows if rows else [_base_row()])

def _write_csv(path, df):
    df.to_csv(path, index=False)


# =============================================================================
# 1) _validate_required_columns
# =============================================================================

def test_validate_required_columns_passes():
    df = _df()
    M._validate_required_columns(df, ["user_id", "monthly_income_usd"])  # no exception

def test_validate_required_columns_raises_on_missing():
    df = pd.DataFrame([{"user_id": "U1"}])
    with pytest.raises(ValueError, match="missing required columns"):
        M._validate_required_columns(df, ["user_id", "monthly_income_usd"])


# =============================================================================
# 2) _to_binary_has_loan
# =============================================================================

@pytest.mark.parametrize("val,expected", [
    (1,       1), (0,       0),
    ("true",  1), ("false", 0),
    ("yes",   1), ("no",    0),
    ("y",     1), ("n",     0),
    ("loan",  1), ("none",  0),
    (None,    0), (float("nan"), 0),
    ("",      0),
])
def test_to_binary_has_loan(val, expected):
    assert M._to_binary_has_loan(val) == expected

def test_to_binary_has_loan_unknown_token_returns_1():
    assert M._to_binary_has_loan("unknown_value") == 1


# =============================================================================
# 3) preprocess_financial_data — column operations
# =============================================================================

def test_preprocess_renames_columns(tmp_path):
    p = tmp_path / "fin.csv"
    out = tmp_path / "out.csv"
    _write_csv(p, _df())
    df = M.preprocess_financial_data(str(p), str(out))
    assert "monthly_income" in df.columns
    assert "monthly_income_usd" not in df.columns
    assert "savings_balance" in df.columns
    assert "monthly_emi" in df.columns

def test_preprocess_keeps_required_columns(tmp_path):
    p = tmp_path / "fin.csv"
    out = tmp_path / "out.csv"
    _write_csv(p, _df())
    df = M.preprocess_financial_data(str(p), str(out))
    for col in ["user_id", "monthly_income", "monthly_expenses",
                "savings_balance", "credit_score"]:
        assert col in df.columns

def test_preprocess_drops_demographic_columns(tmp_path):
    p = tmp_path / "fin.csv"
    out = tmp_path / "out.csv"
    row = _base_row()
    row["age"] = 30
    row["gender"] = "F"
    _write_csv(p, pd.DataFrame([row]))
    df = M.preprocess_financial_data(str(p), str(out))
    assert "age" not in df.columns
    assert "gender" not in df.columns


# =============================================================================
# 4) preprocess_financial_data — type coercions
# =============================================================================

def test_preprocess_has_loan_is_binary(tmp_path):
    p = tmp_path / "fin.csv"
    out = tmp_path / "out.csv"
    _write_csv(p, _df(_base_row(has_loan="yes")))
    df = M.preprocess_financial_data(str(p), str(out))
    assert df.loc[0, "has_loan"] == 1

def test_preprocess_credit_score_is_int(tmp_path):
    p = tmp_path / "fin.csv"
    out = tmp_path / "out.csv"
    _write_csv(p, _df())
    df = M.preprocess_financial_data(str(p), str(out))
    assert df["credit_score"].dtype in [int, np.int64, np.int32]

def test_preprocess_monetary_cols_are_float(tmp_path):
    p = tmp_path / "fin.csv"
    out = tmp_path / "out.csv"
    _write_csv(p, _df())
    df = M.preprocess_financial_data(str(p), str(out))
    assert df["monthly_income"].dtype == float


# =============================================================================
# 5) preprocess_financial_data — deduplication
# =============================================================================

def test_preprocess_removes_duplicate_user_ids(tmp_path):
    p = tmp_path / "fin.csv"
    out = tmp_path / "out.csv"
    r1 = _base_row(user_id="U1", monthly_income_usd=5000.0)
    r2 = _base_row(user_id="U1", monthly_income_usd=6000.0)  # duplicate
    _write_csv(p, pd.DataFrame([r1, r2]))
    df = M.preprocess_financial_data(str(p), str(out))
    assert len(df) == 1
    assert df.loc[df.index[0], "monthly_income"] == 5000.0  # keeps first

def test_preprocess_keeps_unique_users(tmp_path):
    p = tmp_path / "fin.csv"
    out = tmp_path / "out.csv"
    _write_csv(p, pd.DataFrame([_base_row(user_id="U1"), _base_row(user_id="U2")]))
    df = M.preprocess_financial_data(str(p), str(out))
    assert len(df) == 2


# =============================================================================
# 6) preprocess_financial_data — missing value handling
# =============================================================================

def test_preprocess_drops_missing_critical_columns(tmp_path):
    p = tmp_path / "fin.csv"
    out = tmp_path / "out.csv"
    r1 = _base_row(user_id="U1")
    r2 = _base_row(user_id="U2", monthly_income_usd=None)  # missing critical
    _write_csv(p, pd.DataFrame([r1, r2]))
    df = M.preprocess_financial_data(str(p), str(out))
    assert len(df) == 1
    assert df.iloc[0]["user_id"] == "U1"

def test_preprocess_fills_loan_cols_zero_for_no_loan(tmp_path):
    p = tmp_path / "fin.csv"
    out = tmp_path / "out.csv"
    row = _base_row(has_loan=0, loan_amount_usd=None, monthly_emi_usd=None)
    _write_csv(p, pd.DataFrame([row]))
    df = M.preprocess_financial_data(str(p), str(out))
    assert df.iloc[0]["loan_amount"] == pytest.approx(0.0)
    assert df.iloc[0]["monthly_emi"] == pytest.approx(0.0)


# =============================================================================
# 7) preprocess_financial_data — range validation
# =============================================================================

def test_preprocess_drops_negative_income(tmp_path):
    p = tmp_path / "fin.csv"
    out = tmp_path / "out.csv"
    _write_csv(p, pd.DataFrame([
        _base_row(user_id="U1", monthly_income_usd=-100.0),
        _base_row(user_id="U2"),
    ]))
    df = M.preprocess_financial_data(str(p), str(out))
    assert "U1" not in df["user_id"].values

def test_preprocess_drops_invalid_credit_score(tmp_path):
    p = tmp_path / "fin.csv"
    out = tmp_path / "out.csv"
    _write_csv(p, pd.DataFrame([
        _base_row(user_id="U1", credit_score=100),  # below 300
        _base_row(user_id="U2"),
    ]))
    df = M.preprocess_financial_data(str(p), str(out))
    assert "U1" not in df["user_id"].values

def test_preprocess_keeps_valid_credit_score_boundaries(tmp_path):
    p = tmp_path / "fin.csv"
    out = tmp_path / "out.csv"
    _write_csv(p, pd.DataFrame([
        _base_row(user_id="U1", credit_score=300),
        _base_row(user_id="U2", credit_score=850),
    ]))
    df = M.preprocess_financial_data(str(p), str(out))
    assert len(df) == 2


# =============================================================================
# 8) preprocess_financial_data — output file
# =============================================================================

def test_preprocess_creates_output_csv(tmp_path):
    p = tmp_path / "fin.csv"
    out = tmp_path / "out.csv"
    _write_csv(p, _df())
    M.preprocess_financial_data(str(p), str(out))
    assert out.exists()
    df_out = pd.read_csv(out)
    assert len(df_out) == 1

def test_preprocess_missing_required_input_col_raises(tmp_path):
    p = tmp_path / "fin.csv"
    out = tmp_path / "out.csv"
    _write_csv(p, pd.DataFrame([{"user_id": "U1"}]))
    with pytest.raises(ValueError, match="missing required columns"):
        M.preprocess_financial_data(str(p), str(out))