"""
Tests for Feature Engineering — product_review_features.py.

Covers compute_rating_variance (per-product std dev from individual reviews)
and the run_review_features pipeline (input validation, variance merge onto
products, review copy pass-through, NaN-free output).
"""
import os
import sys
import json
import importlib.util

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
        os.path.join(PROJECT_ROOT, "dags", "src", "features", "product_review_features.py"),
        os.path.join(PROJECT_ROOT, "dags", "src", "product_review_features.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        spec = importlib.util.spec_from_file_location("product_review_features", fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["product_review_features"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ImportError("Could not find product_review_features.py. Searched:\n" + "\n".join(candidates))

M = _load()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def _reviews(*rows):
    return pd.DataFrame(rows, columns=["product_id", "rating"])

def _products(*rows):
    return pd.DataFrame(rows, columns=["product_id", "product_name", "price"])


# =============================================================================
# 1) compute_rating_variance
# =============================================================================

def test_compute_rating_variance_basic():
    df = _reviews(("p1", 1), ("p1", 3), ("p1", 5))
    out = M.compute_rating_variance(df)
    assert set(out.columns) == {"product_id", "rating_variance"}
    assert len(out) == 1
    expected = pd.Series([1, 3, 5], dtype=float).std()
    assert out.loc[0, "rating_variance"] == pytest.approx(expected)

def test_compute_rating_variance_multiple_products():
    df = _reviews(("p1", 4), ("p1", 5), ("p2", 1), ("p2", 5))
    out = M.compute_rating_variance(df)
    assert len(out) == 2

def test_compute_rating_variance_single_review_zero():
    df = _reviews(("p1", 4))
    out = M.compute_rating_variance(df)
    assert out.loc[0, "rating_variance"] == pytest.approx(0.0)
    assert not out["rating_variance"].isna().any()

def test_compute_rating_variance_identical_ratings():
    df = _reviews(("p1", 3), ("p1", 3), ("p1", 3))
    out = M.compute_rating_variance(df)
    assert out.loc[0, "rating_variance"] == pytest.approx(0.0)

def test_compute_rating_variance_missing_product_id_raises():
    df = pd.DataFrame([{"rating": 4}])
    with pytest.raises(ValueError, match="product_id"):
        M.compute_rating_variance(df)

def test_compute_rating_variance_missing_rating_raises():
    df = pd.DataFrame([{"product_id": "p1"}])
    with pytest.raises(ValueError):
        M.compute_rating_variance(df)

def test_compute_rating_variance_no_nan():
    df = _reviews(("p1", 5), ("p2", 3))
    out = M.compute_rating_variance(df)
    assert not out["rating_variance"].isna().any()


# =============================================================================
# 2) run_review_features — input validation
# =============================================================================

def test_run_review_features_missing_reviews_raises(tmp_path):
    prod = tmp_path / "prod.jsonl"
    _write_jsonl(prod, [{"product_id": "p1", "product_name": "A", "price": 10}])
    with pytest.raises(FileNotFoundError, match="Reviews"):
        M.run_review_features(
            reviews_path="/nonexistent/rev.jsonl",
            products_path=str(prod),
            product_output_path=str(tmp_path / "prod_out.jsonl"),
            review_output_path=str(tmp_path / "rev_out.jsonl"),
        )

def test_run_review_features_missing_products_raises(tmp_path):
    rev = tmp_path / "rev.jsonl"
    _write_jsonl(rev, [{"product_id": "p1", "rating": 4}])
    with pytest.raises(FileNotFoundError, match="Products"):
        M.run_review_features(
            reviews_path=str(rev),
            products_path="/nonexistent/prod.jsonl",
            product_output_path=str(tmp_path / "prod_out.jsonl"),
            review_output_path=str(tmp_path / "rev_out.jsonl"),
        )


# =============================================================================
# 3) run_review_features — output files
# =============================================================================

def test_run_review_features_creates_output_files(tmp_path):
    rev = tmp_path / "rev.jsonl"
    prod = tmp_path / "prod.jsonl"
    _write_jsonl(rev,  [{"product_id": "p1", "rating": 4},
                        {"product_id": "p1", "rating": 5}])
    _write_jsonl(prod, [{"product_id": "p1", "product_name": "Widget", "price": 10}])

    M.run_review_features(
        str(rev), str(prod),
        str(tmp_path / "prod_out.jsonl"),
        str(tmp_path / "rev_out.jsonl"),
    )
    assert (tmp_path / "prod_out.jsonl").exists()
    assert (tmp_path / "rev_out.jsonl").exists()

def test_run_review_features_review_output_is_copy(tmp_path):
    """review_featured.jsonl should be identical to review_preprocessed.jsonl."""
    rev = tmp_path / "rev.jsonl"
    prod = tmp_path / "prod.jsonl"
    _write_jsonl(rev,  [{"product_id": "p1", "rating": 4, "review_text": "great"}])
    _write_jsonl(prod, [{"product_id": "p1", "product_name": "Widget", "price": 10}])
    rev_out = tmp_path / "rev_out.jsonl"

    M.run_review_features(str(rev), str(prod),
                          str(tmp_path / "prod_out.jsonl"), str(rev_out))

    assert rev_out.read_text() == rev.read_text()


# =============================================================================
# 4) run_review_features — rating_variance merged correctly
# =============================================================================

def test_run_review_features_variance_in_product_output(tmp_path):
    rev = tmp_path / "rev.jsonl"
    prod = tmp_path / "prod.jsonl"
    _write_jsonl(rev,  [{"product_id": "p1", "rating": 1},
                        {"product_id": "p1", "rating": 5}])
    _write_jsonl(prod, [{"product_id": "p1", "product_name": "Widget", "price": 10}])
    prod_out = tmp_path / "prod_out.jsonl"

    M.run_review_features(str(rev), str(prod),
                          str(prod_out), str(tmp_path / "rev_out.jsonl"))

    row = json.loads(prod_out.read_text().strip())
    assert "rating_variance" in row
    assert row["rating_variance"] > 0

def test_run_review_features_unmatched_product_gets_zero_variance(tmp_path):
    """Product with no reviews gets rating_variance = 0.0."""
    rev = tmp_path / "rev.jsonl"
    prod = tmp_path / "prod.jsonl"
    _write_jsonl(rev,  [{"product_id": "p1", "rating": 4}])
    _write_jsonl(prod, [{"product_id": "p1", "product_name": "A", "price": 10},
                        {"product_id": "p2", "product_name": "B", "price": 20}])
    prod_out = tmp_path / "prod_out.jsonl"

    M.run_review_features(str(rev), str(prod),
                          str(prod_out), str(tmp_path / "rev_out.jsonl"))

    rows = [json.loads(l) for l in prod_out.read_text().strip().split("\n")]
    p2 = next(r for r in rows if r["product_id"] == "p2")
    assert p2["rating_variance"] == pytest.approx(0.0)

def test_run_review_features_single_review_product_zero_variance(tmp_path):
    rev = tmp_path / "rev.jsonl"
    prod = tmp_path / "prod.jsonl"
    _write_jsonl(rev,  [{"product_id": "p1", "rating": 5}])
    _write_jsonl(prod, [{"product_id": "p1", "product_name": "Widget", "price": 10}])
    prod_out = tmp_path / "prod_out.jsonl"

    M.run_review_features(str(rev), str(prod),
                          str(prod_out), str(tmp_path / "rev_out.jsonl"))

    row = json.loads(prod_out.read_text().strip())
    assert row["rating_variance"] == pytest.approx(0.0)

def test_run_review_features_no_nan_in_output(tmp_path):
    rev = tmp_path / "rev.jsonl"
    prod = tmp_path / "prod.jsonl"
    _write_jsonl(rev,  [{"product_id": "p1", "rating": 4},
                        {"product_id": "p2", "rating": 5}])
    _write_jsonl(prod, [{"product_id": "p1", "product_name": "A", "price": 10},
                        {"product_id": "p2", "product_name": "B", "price": 20},
                        {"product_id": "p3", "product_name": "C", "price": 30}])
    prod_out = tmp_path / "prod_out.jsonl"

    M.run_review_features(str(rev), str(prod),
                          str(prod_out), str(tmp_path / "rev_out.jsonl"))

    rows = [json.loads(l) for l in prod_out.read_text().strip().split("\n")]
    df = pd.DataFrame(rows)
    assert not df["rating_variance"].isna().any()

def test_run_review_features_product_count_preserved(tmp_path):
    """Output should have same number of products as input."""
    rev = tmp_path / "rev.jsonl"
    prod = tmp_path / "prod.jsonl"
    _write_jsonl(rev,  [{"product_id": "p1", "rating": 4}])
    _write_jsonl(prod, [{"product_id": f"p{i}", "product_name": f"P{i}", "price": i*10}
                        for i in range(1, 6)])
    prod_out = tmp_path / "prod_out.jsonl"

    M.run_review_features(str(rev), str(prod),
                          str(prod_out), str(tmp_path / "rev_out.jsonl"))

    rows = [json.loads(l) for l in prod_out.read_text().strip().split("\n") if l]
    assert len(rows) == 5
