"""
Tests for Feature Engineering — financial_features.py.

Covers: calculate_liquid_savings (SCF-based derivation),
discretionary_income, debt_to_income_ratio, saving_to_income_ratio (STIR),
monthly_expense_burden_ratio, emergency_fund_months (EFM).
Also tests run_financial_features pipeline including zero-division handling,
inf replacement, and output CSV creation.
"""
import os
import sys
import importlib.util
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path constants  (sys.path and utils stub set up by conftest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "src", "features", "financial_features.py"),
        os.path.join(PROJECT_ROOT, "dags", "src", "financial_features.py"),
        os.path.join(PROJECT_ROOT, "src", "features", "financial_features.py"),
        os.path.join(PROJECT_ROOT, "financial_features.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        try:
            spec = importlib.util.spec_from_file_location("financial_features", fpath)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["financial_features"] = mod
            spec.loader.exec_module(mod)
            return mod
        except Exception as e:
            raise ImportError(f"Found {fpath} but failed to load: {e}")
    raise ImportError(f"Could not find financial_features.py. Searched:\n" + "\n".join(candidates))

M = _load()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _base_df(**overrides):
    """Return a minimal valid DataFrame row as a dict, with optional overrides."""
    row = {
        "monthly_income": 5000.0,
        "monthly_expenses": 2000.0,
        "monthly_emi": 500.0,
        "savings_balance": 30000.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _base_df_with_liquid(**overrides):
    """Return a base DataFrame with liquid_savings pre-computed for ratio tests."""
    df = _base_df(**overrides)
    return M.calculate_liquid_savings(df.copy())


# =============================================================================
# 1) calculate_liquid_savings
# =============================================================================

def test_liquid_savings_column_exists():
    """calculate_liquid_savings should add a liquid_savings column."""
    df = _base_df()
    out = M.calculate_liquid_savings(df)
    assert "liquid_savings" in out.columns


def test_liquid_savings_within_bounds():
    """liquid_savings must be >= 0 and <= savings_balance."""
    df = _base_df()
    out = M.calculate_liquid_savings(df)
    assert out.loc[0, "liquid_savings"] >= 0
    assert out.loc[0, "liquid_savings"] <= out.loc[0, "savings_balance"]


def test_liquid_savings_low_income_capped():
    """A low-income user with high savings should be capped at SCF range ($500-$3000)."""
    df = _base_df(monthly_income=500.0, savings_balance=200000.0)
    out = M.calculate_liquid_savings(df)
    # Cap range for <$1,500/mo is $500 - $3,000
    assert out.loc[0, "liquid_savings"] <= 3000.0
    assert out.loc[0, "liquid_savings"] >= 500.0


def test_liquid_savings_high_income_higher_cap():
    """A high-income user should have a higher cap range ($25K-$150K)."""
    df = _base_df(monthly_income=10000.0, savings_balance=900000.0)
    out = M.calculate_liquid_savings(df)
    assert out.loc[0, "liquid_savings"] <= 150000.0
    assert out.loc[0, "liquid_savings"] >= 0


def test_liquid_savings_zero_savings():
    """Zero savings_balance → liquid_savings is 0."""
    df = _base_df(savings_balance=0.0)
    out = M.calculate_liquid_savings(df)
    assert out.loc[0, "liquid_savings"] == 0.0


def test_liquid_savings_reproducibility():
    """Same seed produces identical output."""
    df1 = _base_df()
    df2 = _base_df()
    out1 = M.calculate_liquid_savings(df1, random_state=42)
    out2 = M.calculate_liquid_savings(df2, random_state=42)
    assert out1.loc[0, "liquid_savings"] == out2.loc[0, "liquid_savings"]


# =============================================================================
# 2) calculate_discretionary_income
# =============================================================================

def test_discretionary_income_basic():
    """income - (expenses + emi) = 5000 - 2500 = 2500"""
    df = _base_df()
    out = M.calculate_discretionary_income(df)
    assert "discretionary_income" in out.columns
    assert out.loc[0, "discretionary_income"] == pytest.approx(2500.0)


def test_discretionary_income_negative():
    """Expenses exceed income → negative discretionary income."""
    df = _base_df(monthly_income=1000.0, monthly_expenses=1200.0, monthly_emi=300.0)
    out = M.calculate_discretionary_income(df)
    assert out.loc[0, "discretionary_income"] == pytest.approx(-500.0)


def test_discretionary_income_zero_income():
    """Zero income → discretionary income equals -(expenses + emi)."""
    df = _base_df(monthly_income=0.0, monthly_expenses=500.0, monthly_emi=100.0)
    out = M.calculate_discretionary_income(df)
    assert out.loc[0, "discretionary_income"] == pytest.approx(-600.0)


# =============================================================================
# 3) calculate_ratios (requires liquid_savings column to exist)
# =============================================================================

def test_debt_to_income_ratio_basic():
    """emi / income = 500 / 5000 = 0.1"""
    df = _base_df_with_liquid()
    out = M.calculate_ratios(df)
    assert out.loc[0, "debt_to_income_ratio"] == pytest.approx(0.1)


def test_debt_to_income_ratio_zero_income():
    """Zero income → ratio is NaN."""
    df = _base_df_with_liquid(monthly_income=0.0)
    out = M.calculate_ratios(df)
    assert np.isnan(out.loc[0, "debt_to_income_ratio"])


def test_saving_to_income_ratio_basic():
    """STIR = liquid_savings / income. Uses SCF-derived liquid_savings."""
    df = _base_df_with_liquid()
    out = M.calculate_ratios(df)
    liquid = out.loc[0, "liquid_savings"]
    expected = liquid / 5000.0
    assert out.loc[0, "saving_to_income_ratio"] == pytest.approx(expected)


def test_saving_to_income_ratio_zero_income():
    """Zero income → ratio is NaN."""
    df = _base_df_with_liquid(monthly_income=0.0)
    out = M.calculate_ratios(df)
    assert np.isnan(out.loc[0, "saving_to_income_ratio"])


def test_monthly_expense_burden_ratio_basic():
    """(expenses + emi) / income = 2500 / 5000 = 0.5"""
    df = _base_df_with_liquid()
    out = M.calculate_ratios(df)
    assert out.loc[0, "monthly_expense_burden_ratio"] == pytest.approx(0.5)


def test_monthly_expense_burden_ratio_zero_income():
    """Zero income → ratio is NaN."""
    df = _base_df_with_liquid(monthly_income=0.0)
    out = M.calculate_ratios(df)
    assert np.isnan(out.loc[0, "monthly_expense_burden_ratio"])


def test_emergency_fund_months_basic():
    """EFM = liquid_savings / (expenses + emi). Uses SCF-derived liquid_savings."""
    df = _base_df_with_liquid()
    out = M.calculate_ratios(df)
    liquid = out.loc[0, "liquid_savings"]
    expected = liquid / (2000.0 + 500.0)
    assert out.loc[0, "emergency_fund_months"] == pytest.approx(expected)


def test_emergency_fund_months_zero_expenses():
    """Zero expenses + emi → emergency_fund_months is NaN."""
    df = _base_df_with_liquid(monthly_expenses=0.0, monthly_emi=0.0)
    out = M.calculate_ratios(df)
    assert np.isnan(out.loc[0, "emergency_fund_months"])


def test_calculate_ratios_all_columns_present():
    df = _base_df_with_liquid()
    out = M.calculate_ratios(df)
    for col in ["debt_to_income_ratio", "saving_to_income_ratio",
                "monthly_expense_burden_ratio", "emergency_fund_months"]:
        assert col in out.columns


def test_calculate_ratios_multiple_rows():
    """Ratios computed correctly across multiple rows."""
    df = pd.DataFrame([
        {"monthly_income": 4000.0, "monthly_expenses": 1000.0, "monthly_emi": 200.0, "savings_balance": 24000.0},
        {"monthly_income": 0.0,    "monthly_expenses": 500.0,  "monthly_emi": 100.0, "savings_balance": 5000.0},
    ])
    df = M.calculate_liquid_savings(df)
    out = M.calculate_ratios(df)
    assert out.loc[0, "debt_to_income_ratio"] == pytest.approx(200 / 4000)
    assert np.isnan(out.loc[1, "debt_to_income_ratio"])


# =============================================================================
# 3) run_financial_features (integration)
# =============================================================================

def test_run_financial_features_missing_input(tmp_path, caplog):
    """Missing input file should log an error and return without raising."""
    bad_path = str(tmp_path / "nonexistent.csv")
    out_path = str(tmp_path / "out.csv")
    M.run_financial_features(bad_path, out_path)  # should not raise
    assert not os.path.exists(out_path)


def test_run_financial_features_creates_output(tmp_path):
    """Valid input should produce an output CSV with expected columns."""
    inp = tmp_path / "financial_preprocessed.csv"
    out = tmp_path / "features" / "financial_featured.csv"

    pd.DataFrame([{
        "monthly_income": 6000.0,
        "monthly_expenses": 2000.0,
        "monthly_emi": 500.0,
        "savings_balance": 36000.0,
    }]).to_csv(inp, index=False)

    M.run_financial_features(str(inp), str(out))

    assert out.exists()
    df_out = pd.read_csv(out)
    for col in ["liquid_savings", "discretionary_income", "debt_to_income_ratio",
                "saving_to_income_ratio", "monthly_expense_burden_ratio", "emergency_fund_months"]:
        assert col in df_out.columns
    # liquid_savings must be realistic (not equal to raw savings_balance)
    assert df_out.loc[0, "liquid_savings"] <= df_out.loc[0, "savings_balance"]


def test_run_financial_features_inf_replaced(tmp_path):
    """Inf values in ratio columns should be replaced with NaN in output."""
    inp = tmp_path / "fin.csv"
    out = tmp_path / "features" / "out.csv"

    # Zero income triggers NaN (not inf) via np.where, but zero expenses+emi
    # with nonzero savings could produce inf in financial_runway if not guarded.
    pd.DataFrame([{
        "monthly_income": 5000.0,
        "monthly_expenses": 0.0,
        "monthly_emi": 0.0,
        "savings_balance": 10000.0,
    }]).to_csv(inp, index=False)

    M.run_financial_features(str(inp), str(out))
    df_out = pd.read_csv(out)

    ratio_cols = ["debt_to_income_ratio", "saving_to_income_ratio",
                  "monthly_expense_burden_ratio", "emergency_fund_months"]
    for col in ratio_cols:
        val = df_out.loc[0, col]
        assert not np.isinf(val) if not pd.isna(val) else True


def test_run_financial_features_discretionary_income_no_nan(tmp_path):
    """discretionary_income NaN values should be filled with 0.0 in output."""
    inp = tmp_path / "fin.csv"
    out = tmp_path / "features" / "out.csv"

    pd.DataFrame([{
        "monthly_income": float("nan"),
        "monthly_expenses": float("nan"),
        "monthly_emi": float("nan"),
        "savings_balance": 0.0,
    }]).to_csv(inp, index=False)

    M.run_financial_features(str(inp), str(out))
    df_out = pd.read_csv(out)
    assert not df_out["discretionary_income"].isna().any()