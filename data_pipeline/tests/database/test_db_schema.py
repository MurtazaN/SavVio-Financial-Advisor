"""
Load to Database — schema cross-validation.

The comprehensive schema and table tests live in test_db_connection.py.
This file validates:
  - The short import path (database.db_schema) resolves to the same ORM tables
    as the fully-qualified import (dags.src.database.db_schema).
"""
import sys
import pytest
from sqlalchemy import create_engine, inspect

# sys.path set up by conftest.py

from database.db_schema import (  # noqa: E402
    Base,
    FinancialProfile,
    Product,
    Review,
    create_tables,
)


@pytest.fixture()
def engine():
    return create_engine("sqlite:///:memory:")


def test_short_import_models_match_long_import():
    """Both import paths expose the same set of ORM tables."""
    from dags.src.database.db_schema import Base as BaseLong
    short_tables = set(Base.metadata.tables.keys())
    long_tables = set(BaseLong.metadata.tables.keys())
    assert short_tables == long_tables
