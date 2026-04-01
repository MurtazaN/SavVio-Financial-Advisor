"""
Tests for Load to Database — upload_to_db.py.

Covers the module that loads feature-engineered data (CSV/JSONL) into
PostgreSQL tables: financial_profiles, products, reviews.
Includes helper functions (_read_csv, _read_jsonl, _select_and_rename,
_ensure_jsonb, _upsert_df) and the three loaders + load_all orchestrator.
"""
import json
import os
import sys
import types
import importlib.util
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path constants  (sys.path set up by conftest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DAGS_SRC_DB = os.path.join(PROJECT_ROOT, "dags", "src", "database")

# ---------------------------------------------------------------------------
# Stub heavy dependencies before loading the module
# ---------------------------------------------------------------------------
if "src.database.db_connection" not in sys.modules:
    _stub = types.ModuleType("src.database.db_connection")
    _stub.get_engine = MagicMock(name="get_engine")
    sys.modules["src.database.db_connection"] = _stub
    # ensure parent packages exist
    for _p in ("src", "src.database"):
        if _p not in sys.modules:
            sys.modules[_p] = types.ModuleType(_p)

if "src.database.db_schema" not in sys.modules:
    _stub2 = types.ModuleType("src.database.db_schema")
    _stub2.create_tables = MagicMock(name="create_tables")
    sys.modules["src.database.db_schema"] = _stub2

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    fpath = os.path.join(DAGS_SRC_DB, "upload_to_db.py")
    if not os.path.isfile(fpath):
        raise ImportError(f"Could not find upload_to_db.py at {fpath}")
    spec = importlib.util.spec_from_file_location("upload_to_db", fpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["upload_to_db"] = mod
    spec.loader.exec_module(mod)
    return mod

M = _load()


# =============================================================================
# 1) _read_csv tests
# =============================================================================

def test_read_csv_returns_dataframe(tmp_path):
    p = tmp_path / "financial.csv"
    pd.DataFrame([{"user_id": "u1", "monthly_income": 5000}]).to_csv(p, index=False)
    df = M._read_csv(str(p))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 2)
    assert df.loc[0, "user_id"] == "u1"


def test_read_csv_multiple_rows(tmp_path):
    p = tmp_path / "financial.csv"
    pd.DataFrame([
        {"user_id": "u1", "monthly_income": 5000},
        {"user_id": "u2", "monthly_income": 3000},
    ]).to_csv(p, index=False)
    df = M._read_csv(str(p))
    assert len(df) == 2


# =============================================================================
# 2) _read_jsonl tests
# =============================================================================

def test_read_jsonl_returns_dataframe(tmp_path):
    p = tmp_path / "products.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"product_id": "p1", "price": 10.0}) + "\n")
        f.write(json.dumps({"product_id": "p2", "price": 20.0}) + "\n")
    df = M._read_jsonl(str(p))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df.loc[1, "product_id"] == "p2"


def test_read_jsonl_single_record(tmp_path):
    p = tmp_path / "reviews.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"user_id": "u1", "rating": 5}) + "\n")
    df = M._read_jsonl(str(p))
    assert df.shape == (1, 2)


# =============================================================================
# 3) _select_and_rename tests
# =============================================================================

def test_select_and_rename_keeps_mapped_columns():
    df = pd.DataFrame([{"a": 1, "b": 2, "c": 3}])
    col_map = {"a": "alpha", "b": "beta"}
    result = M._select_and_rename(df, col_map)
    assert list(result.columns) == ["alpha", "beta"]
    assert result.loc[0, "alpha"] == 1


def test_select_and_rename_drops_unmapped_columns():
    df = pd.DataFrame([{"a": 1, "extra_col": 99}])
    result = M._select_and_rename(df, {"a": "alpha"})
    assert "extra_col" not in result.columns


def test_select_and_rename_handles_missing_source_columns():
    df = pd.DataFrame([{"a": 1}])
    result = M._select_and_rename(df, {"a": "alpha", "missing": "m"})
    assert list(result.columns) == ["alpha"]


def test_select_and_rename_identity_mapping():
    df = pd.DataFrame([{"user_id": "u1", "rating": 4}])
    result = M._select_and_rename(df, {"user_id": "user_id", "rating": "rating"})
    assert list(result.columns) == ["user_id", "rating"]


# =============================================================================
# 4) _ensure_jsonb tests
# =============================================================================

def test_ensure_jsonb_serializes_dict():
    df = pd.DataFrame([{"details": {"Brand": "TestBrand", "Color": "Red"}}])
    result = M._ensure_jsonb(df.copy(), "details")
    val = result.loc[0, "details"]
    assert isinstance(val, str)
    parsed = json.loads(val)
    assert parsed["Brand"] == "TestBrand"


def test_ensure_jsonb_passes_through_string():
    df = pd.DataFrame([{"details": '{"Brand": "X"}'}])
    result = M._ensure_jsonb(df.copy(), "details")
    assert result.loc[0, "details"] == '{"Brand": "X"}'


def test_ensure_jsonb_handles_none():
    df = pd.DataFrame([{"details": None}])
    result = M._ensure_jsonb(df.copy(), "details")
    assert result.loc[0, "details"] is None


def test_ensure_jsonb_handles_nan():
    df = pd.DataFrame([{"details": float("nan")}])
    result = M._ensure_jsonb(df.copy(), "details")
    assert result.loc[0, "details"] is None


def test_ensure_jsonb_serializes_list():
    df = pd.DataFrame([{"details": ["a", "b"]}])
    result = M._ensure_jsonb(df.copy(), "details")
    assert json.loads(result.loc[0, "details"]) == ["a", "b"]


def test_ensure_jsonb_missing_column_noop():
    df = pd.DataFrame([{"other_col": 1}])
    result = M._ensure_jsonb(df.copy(), "details")
    assert "other_col" in result.columns
    assert "details" not in result.columns


# =============================================================================
# 5) Column mappings
# =============================================================================

def test_financial_cols_has_required_keys():
    required = {"user_id", "monthly_income", "monthly_expenses", "credit_score"}
    assert required.issubset(set(M.FINANCIAL_COLS.keys()))


def test_product_cols_has_required_keys():
    required = {"product_id", "product_name", "price", "category"}
    assert required.issubset(set(M.PRODUCT_COLS.keys()))


def test_review_cols_has_required_keys():
    required = {"user_id", "product_id", "rating", "review_text"}
    assert required.issubset(set(M.REVIEW_COLS.keys()))


# =============================================================================
# 6) _upsert_df tests
# =============================================================================

def _mock_engine():
    engine = MagicMock()
    conn = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=conn)
    cm.__exit__ = MagicMock(return_value=False)
    engine.begin.return_value = cm
    engine.connect.return_value = cm
    return engine, conn


def test_upsert_df_empty_dataframe():
    engine, conn = _mock_engine()
    df = pd.DataFrame(columns=["user_id", "name"])
    result = M._upsert_df(engine, df, "test_table", ["user_id"], ["name"])
    assert result == 0
    engine.begin.assert_not_called()


def test_upsert_df_builds_correct_sql():
    engine, conn = _mock_engine()
    df = pd.DataFrame([{"user_id": "u1", "name": "Alice"}])
    M._upsert_df(engine, df, "test_table", ["user_id"], ["name"])
    conn.execute.assert_called_once()
    sql_text = str(conn.execute.call_args[0][0])
    assert "INSERT INTO test_table" in sql_text
    assert "ON CONFLICT (user_id)" in sql_text
    assert "name = EXCLUDED.name" in sql_text
    assert "updated_at = CURRENT_TIMESTAMP" in sql_text


def test_upsert_df_returns_row_count():
    engine, conn = _mock_engine()
    df = pd.DataFrame([
        {"user_id": "u1", "name": "Alice"},
        {"user_id": "u2", "name": "Bob"},
    ])
    result = M._upsert_df(engine, df, "test_table", ["user_id"], ["name"])
    assert result == 2


def test_upsert_df_composite_conflict_cols():
    engine, conn = _mock_engine()
    df = pd.DataFrame([{"user_id": "u1", "product_id": "p1", "rating": 5}])
    M._upsert_df(engine, df, "reviews", ["user_id", "product_id"], ["rating"])
    sql_text = str(conn.execute.call_args[0][0])
    assert "ON CONFLICT (user_id, product_id)" in sql_text
    assert "rating = EXCLUDED.rating" in sql_text


# =============================================================================
# 7) load_financial tests
# =============================================================================

def test_load_financial_reads_and_loads(tmp_path):
    p = tmp_path / "financial_featured.csv"
    pd.DataFrame([{
        "user_id": "u1", "monthly_income": 5000, "monthly_expenses": 2000,
        "savings_balance": 30000, "credit_score": 720,
    }]).to_csv(p, index=False)

    engine, conn = _mock_engine()
    with patch.object(M, "_upsert_df", return_value=1) as mock_upsert:
        result = M.load_financial(engine, str(p))
    assert result == 1
    mock_upsert.assert_called_once()
    call_args = mock_upsert.call_args
    assert call_args[0][2] == "financial_profiles"
    assert call_args[0][3] == ["user_id"]


# =============================================================================
# 8) load_products tests
# =============================================================================

def test_load_products_reads_jsonl(tmp_path):
    p = tmp_path / "product_featured.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({
            "product_id": "p1", "product_name": "Widget",
            "price": 10.0, "category": "Tools",
        }) + "\n")

    engine, conn = _mock_engine()
    with patch.object(M, "_upsert_df", return_value=1) as mock_upsert:
        result = M.load_products(engine, str(p))
    assert result == 1
    mock_upsert.assert_called_once()
    call_args = mock_upsert.call_args
    assert call_args[0][2] == "products"
    assert call_args[0][3] == ["product_id"]


# =============================================================================
# 9) load_reviews tests
# =============================================================================

def test_load_reviews_reads_jsonl(tmp_path):
    p = tmp_path / "review_featured.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({
            "user_id": "u1", "product_id": "p1",
            "rating": 4.5, "review_text": "Great",
            "verified_purchase": True, "helpful_vote": 2,
        }) + "\n")

    engine, conn = _mock_engine()
    # Mock the product_id lookup (simulates existing products in DB)
    mock_read_sql = pd.DataFrame({"product_id": ["p1"]})
    with patch("pandas.read_sql", return_value=mock_read_sql), \
         patch.object(M, "_upsert_df", return_value=1) as mock_upsert:
        result = M.load_reviews(engine, str(p))
    assert result == 1
    mock_upsert.assert_called_once()
    call_args = mock_upsert.call_args
    assert call_args[0][2] == "reviews"
    assert call_args[0][3] == ["user_id", "product_id"]


def test_load_reviews_drops_orphan_reviews(tmp_path):
    p = tmp_path / "review_featured.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"user_id": "u1", "product_id": "p1", "rating": 5, "review_text": "ok"}) + "\n")
        f.write(json.dumps({"user_id": "u2", "product_id": "orphan", "rating": 3, "review_text": "no match"}) + "\n")

    engine, conn = _mock_engine()
    mock_read_sql = pd.DataFrame({"product_id": ["p1"]})  # only p1 exists
    with patch("pandas.read_sql", return_value=mock_read_sql), \
         patch.object(M, "_upsert_df", return_value=1) as mock_upsert:
        result = M.load_reviews(engine, str(p))
    assert result == 1  # orphan review filtered out
    # Verify only 1 row was passed to upsert (the orphan was dropped)
    upserted_df = mock_upsert.call_args[0][1]
    assert len(upserted_df) == 1
    assert upserted_df.iloc[0]["product_id"] == "p1"


# =============================================================================
# 10) load_all tests
# =============================================================================

def test_load_all_calls_all_loaders():
    mock_engine = MagicMock()
    with patch.object(M, "get_engine", return_value=mock_engine), \
         patch.object(M, "create_tables"), \
         patch.object(M, "load_financial", return_value=10) as mock_fin, \
         patch.object(M, "load_products", return_value=20) as mock_prod, \
         patch.object(M, "load_reviews", return_value=30) as mock_rev:
        result = M.load_all("/fin.csv", "/prod.jsonl", "/rev.jsonl")

    mock_fin.assert_called_once_with(mock_engine, "/fin.csv")
    mock_prod.assert_called_once_with(mock_engine, "/prod.jsonl")
    mock_rev.assert_called_once_with(mock_engine, "/rev.jsonl")
    assert result == {"financial_profiles": 10, "products": 20, "reviews": 30}
