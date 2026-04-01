"""
Tests for Bias Detection — financial_bias.py.

Covers missingness logic, type inference, numerical banding, and the
integration function `run_financial_bias`.
"""
import os
import sys
import logging
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
        os.path.join(PROJECT_ROOT, "dags", "src", "bias", "financial_bias.py"),
        os.path.join(PROJECT_ROOT, "src", "bias", "financial_bias.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        try:
            spec = importlib.util.spec_from_file_location("financial_bias", fpath)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["financial_bias"] = mod
            spec.loader.exec_module(mod)
            return mod
        except Exception as e:
            raise ImportError(f"Found {fpath} but failed to load: {e}")
    raise ImportError(f"Could not find financial_bias.py. Searched:\n" + "\n".join(candidates))

M = _load()


# =============================================================================
# Helper function tests
# =============================================================================

def test_pct_basic():
    assert M._pct(5, 100) == 5.0
    assert M._pct(1, 3) == 33.33
    assert M._pct(0, 10) == 0.0

def test_pct_zero_total():
    assert M._pct(5, 0) == 0.0

def test_missing_mask():
    s_numeric = pd.Series([1.0, np.nan, 3.0])
    mask = M._missing_mask(s_numeric)
    assert mask.tolist() == [False, True, False]

    s_str = pd.Series(["hello", "", "   ", np.nan])
    mask2 = M._missing_mask(s_str)
    assert mask2.tolist() == [False, True, True, True]


# =============================================================================
# Type Inference
# =============================================================================

def test_infer_type_id():
    s = pd.Series(["user_1", "user_2", "user_3"])
    assert M._infer_type("user_id", s) == "id"

def test_infer_type_boolean():
    s = pd.Series(["True", "false", "1", "0", "yes", "no"])
    assert M._infer_type("has_loan", s) == "boolean"

def test_infer_type_numeric():
    s = pd.Series([100.0, 200.0, 300.0, np.nan])
    assert M._infer_type("income", s) == "numeric"


# =============================================================================
# Banding Functions
# =============================================================================

def test_band_age():
    ages = pd.Series([20, 30, 40, 60, 70, np.nan])
    bands = M._band_age(ages)
    assert bands.iloc[0] == "Young (18-24)"
    assert bands.iloc[1] == "Early-career (25-34)"
    assert bands.iloc[2] == "Mid-career (35-49)"
    assert bands.iloc[3] == "Late-career (50-64)"
    assert bands.iloc[4] == "Senior (65+)"
    assert bands.iloc[5] == "Out-of-range"

def test_band_dti():
    dti = pd.Series([0.1, 0.3, 0.5])
    bands = M._band_dti(dti)
    assert bands.iloc[0] == "Safe"
    assert bands.iloc[1] == "Warning"
    assert bands.iloc[2] == "Risky"


# =============================================================================
# Integration Test
# =============================================================================

def test_run_financial_bias(tmp_path, caplog):
    prep = tmp_path / "financial_preprocessed.csv"
    feat = tmp_path / "financial_featured.csv"
    
    # Valid CSV ensuring we hit numerical banding and boolean slicing
    pd.DataFrame({
        "user_id": ["u1", "u2", "u3", "u4", "u5"],
        "age": [20, 30, 40, 50, 60],
        "has_loan": ["yes", "no", "yes", "yes", "no"],
        "monthly_income": [1000, 4000, 8000, 2000, 5000],
    }).to_csv(prep, index=False)
    
    pd.DataFrame({
        "user_id": ["u1", "u2", "u3"],
        "debt_to_income_ratio": [0.1, 0.5, 0.3],
    }).to_csv(feat, index=False)

    with caplog.at_level(logging.INFO if hasattr(logging, "INFO") else 20):
        M.run_financial_bias(str(prep), str(feat))
    
    assert "Starting financial bias detection" in caplog.text
    assert "overall missingness" in caplog.text
    assert "Financial bias detection complete" in caplog.text
