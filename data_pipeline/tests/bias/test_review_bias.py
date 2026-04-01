"""
Tests for Bias Detection — review_bias.py.

Covers missingness logic, numerical banding, and the
integration function `run_review_bias`.
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
        os.path.join(PROJECT_ROOT, "dags", "src", "bias", "review_bias.py"),
        os.path.join(PROJECT_ROOT, "src", "bias", "review_bias.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        try:
            spec = importlib.util.spec_from_file_location("review_bias", fpath)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["review_bias"] = mod
            spec.loader.exec_module(mod)
            return mod
        except Exception as e:
            raise ImportError(f"Found {fpath} but failed to load: {e}")
    raise ImportError(f"Could not find review_bias.py. Searched:\n" + "\n".join(candidates))

M = _load()


# =============================================================================
# Helper function tests
# =============================================================================

def test_missing_mask():
    s_numeric = pd.Series([1.0, np.nan, 3.0])
    mask = M._missing_mask(s_numeric)
    assert mask.tolist() == [False, True, False]

    s_str = pd.Series(["hello", "", "   ", np.nan, "null"])
    mask2 = M._missing_mask(s_str)
    assert mask2.tolist() == [False, True, True, True, True]

def test_counts():
    s = pd.Series(["A", "B", "A", "C"])
    stats = M._counts(s, 4)
    labels = [stat.label for stat in stats]
    assert "A" in labels
    assert "B" in labels

# =============================================================================
# Type Inference
# =============================================================================

def test_infer_type_rating():
    s = pd.Series([5.0, 4.0, 1.0])
    assert M._infer_type("rating", s) == "numeric"

def test_infer_type_boolean():
    s = pd.Series(["true", "false"])
    assert M._infer_type("verified_purchase", s) == "boolean"

def test_infer_type_id():
    s = pd.Series(["u1", "u2", "u3"])
    assert M._infer_type("user_id", s) == "id"


# =============================================================================
# Integration Test
# =============================================================================

def test_run_review_bias(tmp_path, caplog):
    prep = tmp_path / "review_preprocessed.jsonl"
    feat = tmp_path / "review_featured.jsonl"
    
    pd.DataFrame({
        "user_id": ["u1", "u2", "u3", "u4"],
        "parent_asin": ["p1", "p2", "p1", "p3"],
        "rating": [5.0, 4.0, 1.0, 5.0],
        "verified_purchase": [True, False, True, True],
        "title": ["Good", "Okay", "Bad", "Great"],
        "text": ["Very good product", "It was fine", "Terrible", "Amazing"]
    }).to_json(prep, orient="records", lines=True)
    
    pd.DataFrame({
        "user_id": ["u1", "u2"],
        "helpful_vote": [10, 0]
    }).to_json(feat, orient="records", lines=True)

    with caplog.at_level(logging.INFO):
        M.run_review_bias(str(prep), str(feat))
    
    assert "Starting review bias detection" in caplog.text
    assert "total number of reviews" in caplog.text
    assert "Review bias detection complete" in caplog.text
