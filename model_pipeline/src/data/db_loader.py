"""
Database Data Loader for the Model Pipeline.

Reads training data directly from the SavVio PostgreSQL database
populated by the Airflow Data Pipeline, instead of relying on CSV exports.

Leverages the same connection pattern as data_pipeline/dags/src/database/db_connection.py
so both pipelines talk to the same DB with the same env vars.

Environment variables (same as data_pipeline):
    DB_USER      (default: postgres)
    DB_PASSWORD  (default: postgres)
    DB_HOST      (default: host.docker.internal — reaches Mac-local PG from Docker)
    DB_PORT      (default: 5432)
    DB_NAME      (default: savvio_dev)
    APP_ENV      (default: dev — set to "prod" for Cloud SQL)

Usage:
    from data.db_loader import load_financial_profiles, load_products, load_reviews

    financial_df = load_financial_profiles()
    products_df  = load_products()
    reviews_df   = load_reviews()
"""

import os
import logging
import pandas as pd
from savviocore.database.db_connection import get_engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection (Now imported from savviocore)
# ---------------------------------------------------------------------------
# DB Engine creation is handled by savviocore.database.db_connection


# ---------------------------------------------------------------------------
# Table readers
# ---------------------------------------------------------------------------

def load_financial_profiles(engine=None) -> pd.DataFrame:
    """
    Load all rows from the `financial_profiles` table.

    Columns returned match the schema in data_pipeline/dags/src/database/db_schema.py:
        user_id, monthly_income, monthly_expenses, savings_balance,
        liquid_savings, has_loan, loan_amount, monthly_emi, loan_interest_rate,
        loan_term_months, credit_score, employment_status, region,
        discretionary_income, debt_to_income_ratio, saving_to_income_ratio,
        monthly_expense_burden_ratio, emergency_fund_months
    """
    if engine is None:
        engine = get_engine()

    query = """
        SELECT user_id, monthly_income, monthly_expenses, savings_balance,
               liquid_savings, has_loan, loan_amount, monthly_emi, loan_interest_rate,
               loan_term_months, credit_score, employment_status, region,
               discretionary_income, debt_to_income_ratio,
               saving_to_income_ratio, monthly_expense_burden_ratio,
               emergency_fund_months
        FROM financial_profiles
    """
    df = pd.read_sql(text(query), con=engine)
    logger.info("Loaded financial_profiles: %d rows, %d cols", *df.shape)
    print(f"[DB] financial_profiles loaded — {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_products(engine=None) -> pd.DataFrame:
    """
    Load all rows from the `products` table.

    Columns: product_id, product_name, price, average_rating, rating_number,
             rating_variance, description, features, details, category
    """
    if engine is None:
        engine = get_engine()

    query = """
        SELECT product_id, product_name, price, average_rating,
               rating_number, rating_variance, description,
               features, details, category
        FROM products
    """
    df = pd.read_sql(text(query), con=engine)
    logger.info("Loaded products: %d rows, %d cols", *df.shape)
    print(f"[DB] products loaded — {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_reviews(engine=None) -> pd.DataFrame:
    """
    Load all rows from the `reviews` table.

    Columns: user_id, asin, product_id, rating, review_title, review_text,
             verified_purchase, helpful_vote
    """
    if engine is None:
        engine = get_engine()

    query = """
        SELECT user_id, asin, product_id, rating, review_title,
               review_text, verified_purchase, helpful_vote
        FROM reviews
    """
    df = pd.read_sql(text(query), con=engine)
    logger.info("Loaded reviews: %d rows, %d cols", *df.shape)
    print(f"[DB] reviews loaded — {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_all(engine=None):
    """
    Convenience function: load all three tables and return them as a dict.

    Returns:
        dict with keys: "financial", "products", "reviews"
    """
    if engine is None:
        engine = get_engine()

    return {
        "financial": load_financial_profiles(engine),
        "products": load_products(engine),
        "reviews": load_reviews(engine),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing DB connection and table loading...")

    try:
        engine = get_engine(echo=False)
        # Quick connectivity check
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print(f"Connection OK: {result.scalar()}")

        data = load_all(engine)
        for name, df in data.items():
            print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"    Columns: {list(df.columns)}")
    except Exception as e:
        print(f"DB connection failed: {e}")
        print("Make sure the SavVio PostgreSQL database is running and env vars are set.")
        print("Required: DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME")
