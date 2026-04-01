# model_pipeline/tests/test_data_loader.py
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Stub external dependencies
# ---------------------------------------------------------------------------
def _stub_modules():
    # savviocore
    savviocore = types.ModuleType("savviocore")
    savviocore.database = types.ModuleType("savviocore.database")
    savviocore.database.db_connection = types.ModuleType("savviocore.database.db_connection")
    savviocore.database.db_connection.get_engine = MagicMock(return_value=MagicMock())
    sys.modules.setdefault("savviocore", savviocore)
    sys.modules.setdefault("savviocore.database", savviocore.database)
    sys.modules.setdefault("savviocore.database.db_connection", savviocore.database.db_connection)

    # sqlalchemy
    sqlalchemy = types.ModuleType("sqlalchemy")
    sqlalchemy.text = lambda q: q
    sys.modules.setdefault("sqlalchemy", sqlalchemy)

_stub_modules()

from data.db_loader import (
    load_financial_profiles,
    load_products,
    load_reviews,
    load_all,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
FINANCIAL_COLS = [
    "user_id", "monthly_income", "monthly_expenses", "savings_balance",
    "has_loan", "loan_amount", "monthly_emi", "loan_interest_rate",
    "loan_term_months", "credit_score", "employment_status", "region",
    "discretionary_income", "debt_to_income_ratio", "saving_to_income_ratio",
    "monthly_expense_burden_ratio", "emergency_fund_months",
]

PRODUCT_COLS = [
    "product_id", "product_name", "price", "average_rating",
    "rating_number", "rating_variance", "description", "features",
    "details", "category",
]

REVIEW_COLS = [
    "user_id", "asin", "product_id", "rating", "review_title",
    "review_text", "verified_purchase", "helpful_vote",
]

def _make_df(cols, n=5):
    return pd.DataFrame({c: [f"val_{i}" for i in range(n)] for c in cols})


# ---------------------------------------------------------------------------
# Tests: load_financial_profiles
# ---------------------------------------------------------------------------
class TestLoadFinancialProfiles:

    def test_returns_dataframe(self):
        mock_engine = MagicMock()
        with patch("data.db_loader.pd.read_sql", return_value=_make_df(FINANCIAL_COLS)):
            df = load_financial_profiles(engine=mock_engine)
        assert isinstance(df, pd.DataFrame)

    def test_uses_provided_engine(self):
        mock_engine = MagicMock()
        with patch("data.db_loader.pd.read_sql", return_value=_make_df(FINANCIAL_COLS)) as mock_sql:
            load_financial_profiles(engine=mock_engine)
        assert mock_sql.called

    def test_calls_get_engine_when_none(self):
        with patch("data.db_loader.get_engine", return_value=MagicMock()) as mock_get, \
             patch("data.db_loader.pd.read_sql", return_value=_make_df(FINANCIAL_COLS)):
            load_financial_profiles(engine=None)
        mock_get.assert_called_once()

    def test_returns_expected_columns(self):
        mock_engine = MagicMock()
        with patch("data.db_loader.pd.read_sql", return_value=_make_df(FINANCIAL_COLS)):
            df = load_financial_profiles(engine=mock_engine)
        for col in FINANCIAL_COLS:
            assert col in df.columns

    def test_returns_correct_row_count(self):
        mock_engine = MagicMock()
        with patch("data.db_loader.pd.read_sql", return_value=_make_df(FINANCIAL_COLS, n=10)):
            df = load_financial_profiles(engine=mock_engine)
        assert len(df) == 10


# ---------------------------------------------------------------------------
# Tests: load_products
# ---------------------------------------------------------------------------
class TestLoadProducts:

    def test_returns_dataframe(self):
        mock_engine = MagicMock()
        with patch("data.db_loader.pd.read_sql", return_value=_make_df(PRODUCT_COLS)):
            df = load_products(engine=mock_engine)
        assert isinstance(df, pd.DataFrame)

    def test_returns_expected_columns(self):
        mock_engine = MagicMock()
        with patch("data.db_loader.pd.read_sql", return_value=_make_df(PRODUCT_COLS)):
            df = load_products(engine=mock_engine)
        for col in PRODUCT_COLS:
            assert col in df.columns

    def test_calls_get_engine_when_none(self):
        with patch("data.db_loader.get_engine", return_value=MagicMock()) as mock_get, \
             patch("data.db_loader.pd.read_sql", return_value=_make_df(PRODUCT_COLS)):
            load_products(engine=None)
        mock_get.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: load_reviews
# ---------------------------------------------------------------------------
class TestLoadReviews:

    def test_returns_dataframe(self):
        mock_engine = MagicMock()
        with patch("data.db_loader.pd.read_sql", return_value=_make_df(REVIEW_COLS)):
            df = load_reviews(engine=mock_engine)
        assert isinstance(df, pd.DataFrame)

    def test_returns_expected_columns(self):
        mock_engine = MagicMock()
        with patch("data.db_loader.pd.read_sql", return_value=_make_df(REVIEW_COLS)):
            df = load_reviews(engine=mock_engine)
        for col in REVIEW_COLS:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Tests: load_all
# ---------------------------------------------------------------------------
class TestLoadAll:

    def test_returns_dict_with_three_keys(self):
        mock_engine = MagicMock()
        with patch("data.db_loader.pd.read_sql", side_effect=[
            _make_df(FINANCIAL_COLS),
            _make_df(PRODUCT_COLS),
            _make_df(REVIEW_COLS),
        ]):
            result = load_all(engine=mock_engine)
        assert set(result.keys()) == {"financial", "products", "reviews"}

    def test_all_values_are_dataframes(self):
        mock_engine = MagicMock()
        with patch("data.db_loader.pd.read_sql", side_effect=[
            _make_df(FINANCIAL_COLS),
            _make_df(PRODUCT_COLS),
            _make_df(REVIEW_COLS),
        ]):
            result = load_all(engine=mock_engine)
        for v in result.values():
            assert isinstance(v, pd.DataFrame)