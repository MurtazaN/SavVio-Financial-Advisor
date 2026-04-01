"""
SQLAlchemy table definitions for SavVio.
Matches the actual columns from preprocessing (see Data Pipeline Plan).
"""
import logging

from sqlalchemy import (
    Column, Integer, Float, String, Boolean, Text, DateTime, ForeignKey,
    UniqueConstraint, JSON, text, inspect as sa_inspect
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


# ---------------------------------------------------------------------------
# Financial profiles
# ---------------------------------------------------------------------------

class FinancialProfile(Base):
    __tablename__ = "financial_profiles"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    user_id             = Column(String(100), unique=True, nullable=False)
    monthly_income      = Column(Float, nullable=False)
    monthly_expenses    = Column(Float, nullable=False)
    savings_balance     = Column(Float)
    has_loan            = Column(Integer)          # 1 or 0
    loan_amount         = Column(Float, default=0)
    monthly_emi         = Column(Float, default=0)
    loan_interest_rate  = Column(Float, default=0)
    loan_term_months    = Column(Float, default=0)
    credit_score        = Column(Integer)
    employment_status   = Column(String(50))
    region              = Column(String(50))
    # --- feature-engineered columns (optional, loaded if present) ---
    liquid_savings        = Column(Float)
    discretionary_income  = Column(Float)
    debt_to_income_ratio  = Column(Float)
    saving_to_income_ratio       = Column(Float)
    monthly_expense_burden_ratio = Column(Float)
    emergency_fund_months = Column(Float)
    created_at            = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    updated_at            = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

class Product(Base):
    __tablename__ = "products"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    product_id      = Column(String(100), unique=True, nullable=False)  # parent_asin
    product_name    = Column(Text, nullable=False)
    price           = Column(Float, nullable=False)
    average_rating  = Column(Float)
    rating_number   = Column(Integer)
    rating_variance = Column(Float)
    description     = Column(Text)
    features        = Column(Text)
    details         = Column(JSON)        # stored as JSONB in PostgreSQL
    category        = Column(Text)
    created_at      = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    updated_at      = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------

class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (
        UniqueConstraint("user_id", "product_id", name="uq_reviews_user_product"),
    )

    id                = Column(Integer, primary_key=True, autoincrement=True)
    user_id           = Column(String(255), nullable=False)
    asin              = Column(String(100))
    product_id        = Column(String(100), ForeignKey("products.product_id"), nullable=False)
    rating            = Column(Float, nullable=False)
    review_title      = Column(Text)
    review_text       = Column(Text)
    verified_purchase = Column(Boolean)
    helpful_vote      = Column(Integer, default=0)
    created_at        = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    updated_at        = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))


# ---------------------------------------------------------------------------
# Helper to create all tables
# ---------------------------------------------------------------------------

def create_tables(engine):
    """Create all tables that don't already exist, and add any missing columns to existing tables."""
    Base.metadata.create_all(engine, checkfirst=True)

    # Add any new columns to existing tables (handles schema evolution
    # without a full migration framework like Alembic).
    inspector = sa_inspect(engine)
    for table_name, table in Base.metadata.tables.items():
        if not inspector.has_table(table_name):
            continue
        existing_cols = {c["name"] for c in inspector.get_columns(table_name)}
        for col in table.columns:
            if col.name not in existing_cols:
                col_type = col.type.compile(engine.dialect)
                with engine.begin() as conn:
                    conn.execute(text(
                        f'ALTER TABLE {table_name} ADD COLUMN "{col.name}" {col_type}'
                    ))
                logging.getLogger(__name__).info(
                    "Added column '%s' (%s) to table '%s'",
                    col.name, col_type, table_name,
                )
