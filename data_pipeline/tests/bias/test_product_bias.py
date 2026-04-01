"""
Tests for Bias Detection — product_bias.py.

Covers missingness logic, list parsing, numerical banding, and the
integration function `run_product_bias`.
"""
import logging
import os
import sys
import importlib.util

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "src", "bias", "product_bias.py"),
        os.path.join(PROJECT_ROOT, "src", "bias", "product_bias.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        try:
            spec = importlib.util.spec_from_file_location("product_bias", fpath)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["product_bias"] = mod
            spec.loader.exec_module(mod)
            return mod
        except Exception as e:
            raise ImportError(f"Found {fpath} but failed to load: {e}")
    raise ImportError(f"Could not find product_bias.py. Searched:\n" + "\n".join(candidates))

M = _load()


# =============================================================================
# Helper function tests
# =============================================================================

def test_safe_json_parse():
    assert M._safe_json_parse([1, 2, 3]) == [1, 2, 3]
    assert M._safe_json_parse('{"k": "v"}') == {"k": "v"}
    assert M._safe_json_parse(" [1, 2, 3] ") == [1, 2, 3]
    assert M._safe_json_parse("invalid json") is None
    assert M._safe_json_parse(None) is None

def test_missing_mask_generic():
    s_numeric = pd.Series([1.0, np.nan, 3.0])
    mask = M._missing_mask_generic(s_numeric)
    assert mask.tolist() == [False, True, False]

    s_str = pd.Series(["hello", "", "   ", np.nan, "null"])
    mask2 = M._missing_mask_generic(s_str)
    assert mask2.tolist() == [False, True, True, True, True]

# =============================================================================
# Type Inference
# =============================================================================

def test_infer_column_type_id():
    s = pd.Series(["p1", "p2", "p3"])
    assert M._infer_column_type("product_id", s) == "id"

def test_infer_column_type_list():
    s = pd.Series(['["f1"]', '[]'])
    assert M._infer_column_type("features", s) == "list"

def test_infer_column_type_dict():
    s = pd.Series(['{"a": 1}'])
    assert M._infer_column_type("details", s) == "dict"

def test_infer_column_type_numeric():
    s = pd.Series([10.5, 20.0, np.nan])
    assert M._infer_column_type("price", s) == "numeric"


# =============================================================================
# Banding Functions
# =============================================================================

def test_numeric_slices_price():
    prices = pd.Series([10.0, 50.0, 300.0, np.nan])
    labels, targets = M._numeric_slices("price", prices)
    assert labels.iloc[0] == "Budget"
    assert labels.iloc[1] == "Mid-range"
    assert labels.iloc[2] == "Premium"
    assert labels.iloc[3] == "Missing"
    assert targets == ["Missing", "Budget"]

def test_numeric_slices_rating():
    ratings = pd.Series([2.0, 3.5, 4.5, np.nan])
    labels, targets = M._numeric_slices("average_rating", ratings)
    assert labels.iloc[0] == "Low"
    assert labels.iloc[1] == "Medium"
    assert labels.iloc[2] == "High"
    assert labels.iloc[3] == "Missing"
    assert targets == ["Low"]


# =============================================================================
# Integration Test
# =============================================================================

def test_run_product_bias(tmp_path, caplog):
    prep = tmp_path / "product_preprocessed.jsonl"
    feat = tmp_path / "product_featured.jsonl"
    
    pd.DataFrame({
        "product_id": ["p1", "p2", "p3", "p4"],
        "price": [10.0, 150.0, 300.0, 25.0],
        "features": ['["a"]', '["b", "c"]', '[]', '["d"]'],
        "details": ['{"Brand": "BrandA"}', '{}', '{"Brand": "BrandB"}', '{}'],
        "title": ["A title"] * 4
    }).to_json(prep, orient="records", lines=True)
    
    pd.DataFrame({
        "product_id": ["p1", "p2"],
        "rating_variance": [0.0, 1.2],
    }).to_json(feat, orient="records", lines=True)

    with caplog.at_level(logging.INFO):
        M.run_product_bias(str(prep), str(feat))
    
    assert "Starting product bias detection" in caplog.text
    assert "overall missingness" in caplog.text
    assert "Product bias detection complete" in caplog.text
