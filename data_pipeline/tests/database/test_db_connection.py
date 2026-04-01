"""
Tests for Load to Database — connection and schema layer.

Covers:
  db_connection.py — _dev_url, _prod_url, get_engine, get_session, ensure_pgvector
  db_schema.py     — create_tables, table definitions, NOT NULL constraints, FKs
"""
import os
import sys

import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine, inspect, text

# sys.path set up by conftest.py

from savviocore.database.db_schema import (  # noqa: E402
    Base,
    FinancialProfile,
    Product,
    Review,
    create_tables,
)

from savviocore.database.db_connection import (
    get_engine,
    get_session,
    ensure_pgvector,
    _dev_url,
    _prod_url,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def engine():
    """Use in-memory SQLite DB for schema tests"""
    return create_engine("sqlite:///:memory:")


@pytest.fixture()
def inspector(engine):
    """SQLAlchemy inspector after tables are created"""
    create_tables(engine)
    return inspect(engine)


# =============================================================================
# 1) db_connection.py — _dev_url / _prod_url / get_engine / get_session
# =============================================================================

def test_dev_url_uses_defaults():
    with patch.dict(os.environ, {}, clear=True):
        url = _dev_url()
    assert "postgresql" in url
    assert "postgres" in url  # default user
    assert "localhost" in url or "5432" in url


def test_dev_url_reads_env_vars():
    env = {
        "DB_USER": "myuser",
        "DB_PASSWORD": "mypass",
        "DB_HOST": "myhost",
        "DB_PORT": "5433",
        "DB_NAME": "mydb",
    }
    with patch.dict(os.environ, env, clear=True):
        url = _dev_url()
    assert "myuser" in url
    assert "mypass" in url
    assert "myhost" in url
    assert "5433" in url
    assert "mydb" in url


def test_prod_url_reads_required_env_vars():
    env = {
        "DB_USER": "produser",
        "DB_PASSWORD": "prodpass",
        "DB_NAME": "savvio_prod",
    }
    with patch.dict(os.environ, env, clear=True):
        url = _prod_url()
    assert "produser" in url
    assert "prodpass" in url
    assert "savvio_prod" in url


def test_prod_url_raises_without_db_user():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(KeyError):
            _prod_url()


def test_get_engine_invalid_env_raises():
    with pytest.raises(ValueError, match="Unknown environment"):
        get_engine("staging")


@patch("savviocore.database.db_connection.create_engine")
def test_get_engine_dev_calls_create_engine(mock_create):
    mock_create.return_value = MagicMock()
    engine = get_engine("dev", echo=True)
    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args
    assert call_kwargs[1]["echo"] is True
    assert call_kwargs[1]["pool_pre_ping"] is True


@patch("savviocore.database.db_connection.sessionmaker")
def test_get_session_returns_session(mock_sm):
    mock_session_class = MagicMock()
    mock_sm.return_value = mock_session_class
    engine = MagicMock()
    session = get_session(engine)
    mock_sm.assert_called_once_with(bind=engine)
    mock_session_class.assert_called_once()


def test_ensure_pgvector_executes_create_extension():
    engine = MagicMock()
    conn = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=conn)
    cm.__exit__ = MagicMock(return_value=False)
    engine.connect.return_value = cm
    ensure_pgvector(engine)
    conn.execute.assert_called_once()
    sql_arg = str(conn.execute.call_args[0][0])
    assert "CREATE EXTENSION" in sql_arg and "vector" in sql_arg
    conn.commit.assert_called_once()


# =============================================================================
# 2) db_schema.py — create_tables
# =============================================================================

def test_create_tables_success(inspector):
    tables = set(inspector.get_table_names())
    assert "financial_profiles" in tables
    assert "products" in tables
    assert "reviews" in tables


# =============================================================================
# 3) db_schema.py — Products table
# =============================================================================

def test_products_table_has_expected_columns(inspector):
    cols = {c["name"]: c for c in inspector.get_columns("products")}
    expected = {
        "id", "product_id", "product_name", "price",
        "average_rating", "rating_number", "rating_variance",
        "description", "features", "details", "category", "created_at",
    }
    assert expected.issubset(set(cols.keys()))

def test_products_table_not_null_constraints(inspector):
    cols = {c["name"]: c for c in inspector.get_columns("products")}
    assert cols["product_id"]["nullable"] is False
    assert cols["product_name"]["nullable"] is False
    assert cols["price"]["nullable"] is False


# =============================================================================
# 4) db_schema.py — FinancialProfile table
# =============================================================================

def test_financial_profiles_table_has_expected_columns(inspector):
    cols = {c["name"]: c for c in inspector.get_columns("financial_profiles")}
    expected = {
        "id",
        "user_id",
        "monthly_income",
        "monthly_expenses",
        "savings_balance",
        "has_loan",
        "loan_amount",
        "monthly_emi",
        "loan_interest_rate",
        "loan_term_months",
        "credit_score",
        "employment_status",
        "region",
        "discretionary_income",
        "debt_to_income_ratio",
        "saving_to_income_ratio",
        "monthly_expense_burden_ratio",
        "emergency_fund_months",
        "created_at",
    }
    assert expected.issubset(set(cols.keys()))

def test_financial_profiles_not_null_constraints(inspector):
    cols = {c["name"]: c for c in inspector.get_columns("financial_profiles")}
    assert cols["user_id"]["nullable"] is False
    assert cols["monthly_income"]["nullable"] is False
    assert cols["monthly_expenses"]["nullable"] is False


# =============================================================================
# 5) db_schema.py — Reviews table
# =============================================================================

def test_reviews_table_has_expected_columns(inspector):
    cols = {c["name"]: c for c in inspector.get_columns("reviews")}
    expected = {
        "id", "user_id", "asin", "product_id", "rating",
        "review_title", "review_text", "verified_purchase",
        "helpful_vote", "created_at",
    }
    assert expected.issubset(set(cols.keys()))

def test_reviews_rating_not_null(inspector):
    cols = {c["name"]: c for c in inspector.get_columns("reviews")}
    assert cols["rating"]["nullable"] is False

def test_reviews_foreign_key_to_products(inspector):
    fks = inspector.get_foreign_keys("reviews")
    assert len(fks) >= 1
    matches = [
        fk for fk in fks
        if fk.get("referred_table") == "products"
        and fk.get("constrained_columns") == ["product_id"]
        and fk.get("referred_columns") == ["product_id"]
    ]
    assert matches, f"Expected FK reviews.product_id -> products.product_id, got: {fks}"


# =============================================================================
# 6) db_schema.py — model metadata
# =============================================================================

def test_model_tablenames_and_metadata():
    assert FinancialProfile.__tablename__ == "financial_profiles"
    assert Product.__tablename__ == "products"
    assert Review.__tablename__ == "reviews"
    table_names = set(Base.metadata.tables.keys())
    assert {"financial_profiles", "products", "reviews"}.issubset(table_names)