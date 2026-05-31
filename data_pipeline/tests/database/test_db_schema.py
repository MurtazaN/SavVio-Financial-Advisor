"""
Load to Database — schema cross-validation.

The comprehensive schema and table tests live in test_db_connection.py.
This file validates that the savviocore ORM schema can be imported and
that the expected tables are present.
"""
import pytest
from sqlalchemy import create_engine, inspect

from savviocore.database.db_schema import (  # noqa: E402
    Base,
    FinancialProfile,
    Product,
    Review,
    create_tables,
)


@pytest.fixture()
def engine():
    return create_engine("sqlite:///:memory:")


def test_schema_tables_present():
    """Schema exposes the expected ORM tables."""
    expected = {"financial_profiles", "products", "reviews"}
    actual = set(Base.metadata.tables.keys())
    assert expected <= actual, f"Missing tables: {expected - actual}"
