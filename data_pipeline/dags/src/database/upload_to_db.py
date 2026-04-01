"""
Upload finalized (feature-engineered) data to PostgreSQL.

Loads three datasets into their respective tables:
    - financial_featured.csv   (CSV)   → financial_profiles
    - products_featured.jsonl  (JSONL) → products
    - reviews_featured.jsonl   (JSONL) → reviews

Products must be loaded before reviews (FK dependency).
"""

import json
import os
import logging
import pandas as pd
from sqlalchemy import text

from savviocore.database.db_connection import get_engine
from savviocore.database.db_schema import create_tables

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column mapping: source field → DB column
# Only mapped columns get pushed to the database.
# Adjust keys if your file headers differ.
# ---------------------------------------------------------------------------

'''
Column mapping is quite important because it's your safety net — it ensures only the columns you expect end up in the database,
and it handles any naming mismatches between your JSONL field names and your DB column names. 
Without it, you risk pushing extra/unexpected columns into to_sql() which would throw errors.
'''

FINANCIAL_COLS = {
    "user_id": "user_id",
    "monthly_income": "monthly_income",
    "monthly_expenses": "monthly_expenses",
    "savings_balance": "savings_balance",
    "has_loan": "has_loan",
    "loan_amount": "loan_amount",
    "monthly_emi": "monthly_emi",
    "loan_interest_rate": "loan_interest_rate",
    "loan_term_months": "loan_term_months",
    "credit_score": "credit_score",
    "employment_status": "employment_status",
    "region": "region",
    # Feature-engineered (included if present)
    "liquid_savings": "liquid_savings",
    "discretionary_income": "discretionary_income",
    "debt_to_income_ratio": "debt_to_income_ratio",
    "saving_to_income_ratio": "saving_to_income_ratio",
    "monthly_expense_burden_ratio": "monthly_expense_burden_ratio",
    "emergency_fund_months": "emergency_fund_months",
}

PRODUCT_COLS = {
    "product_id": "product_id",
    "product_name": "product_name",
    "price": "price",
    "average_rating": "average_rating",
    "rating_number": "rating_number",
    "rating_variance": "rating_variance",
    "description": "description",
    "features": "features",
    "details": "details",       # stored as JSONB
    "category": "category",
}

REVIEW_COLS = {
    "user_id": "user_id",
    "asin": "asin",
    "product_id": "product_id",
    "rating": "rating",
    "review_title": "review_title",
    "review_text": "review_text",
    "verified_purchase": "verified_purchase",
    "helpful_vote": "helpful_vote",
}


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> pd.DataFrame:
    """Read a CSV file."""
    df = pd.read_csv(path)
    logger.info("Read %d rows from CSV: %s", len(df), path)
    return df


def _read_jsonl(path: str) -> pd.DataFrame:
    """Read a JSONL file (one JSON object per line)."""
    # df = pd.read_json(path, lines=True)
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    if file_size_mb > 300:
        chunks = pd.read_json(path, lines=True, chunksize=100_000)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_json(path, lines=True)
    logger.info("Read %d rows from JSONL: %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_and_rename(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Keep only columns that exist in both the source and the mapping."""
    available = {k: v for k, v in col_map.items() if k in df.columns}
    missing = [k for k in col_map if k not in df.columns]
    if missing:
        logger.warning("Columns not found in source (skipped): %s", missing)
    return df[list(available.keys())].rename(columns=available)


def _upsert_df(
    engine,
    df: pd.DataFrame,
    table_name: str,
    conflict_cols: list,
    update_cols: list,
    chunksize: int = 5_000,
) -> int:
    """
    Upsert a DataFrame into a PostgreSQL table.

    Uses INSERT ... ON CONFLICT (conflict_cols) DO UPDATE SET ...
    to insert new rows and update existing ones. No data is deleted.

    Args:
        engine:        SQLAlchemy engine.
        df:            DataFrame to upsert.
        table_name:    Target table name.
        conflict_cols: Column(s) forming the unique constraint.
        update_cols:   Column(s) to update on conflict.
        chunksize:     Number of rows per batch.

    Returns:
        Number of rows processed.
    """
    if df.empty:
        logger.info("Empty DataFrame — nothing to upsert into %s", table_name)
        return 0

    # Build the column lists for the INSERT.
    all_cols = list(df.columns)
    col_list = ", ".join(all_cols)
    val_placeholders = ", ".join(f":{c}" for c in all_cols)
    conflict_list = ", ".join(conflict_cols)
    update_set = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in update_cols
    )
    # Also update the updated_at timestamp on conflict.
    update_set += ", updated_at = CURRENT_TIMESTAMP"

    sql = text(
        f"INSERT INTO {table_name} ({col_list}) "
        f"VALUES ({val_placeholders}) "
        f"ON CONFLICT ({conflict_list}) DO UPDATE SET {update_set}"
    )

    rows_processed = 0
    with engine.begin() as conn:
        for start in range(0, len(df), chunksize):
            chunk = df.iloc[start : start + chunksize]
            records = chunk.to_dict(orient="records")
            conn.execute(sql, records)
            rows_processed += len(records)

    logger.info("Upserted %d rows into %s", rows_processed, table_name)
    return rows_processed


def _ensure_jsonb(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Ensure a column contains valid JSON strings for JSONB storage.
    If values are already dicts, serialize them. If strings, validate.
    """
    if col not in df.columns:
        return df

    def to_json_str(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        if isinstance(val, (dict, list)):
            return json.dumps(val)
        if isinstance(val, str):
            return val  # assume already valid JSON string
        return json.dumps(val)

    df[col] = df[col].apply(to_json_str)
    return df


# ---------------------------------------------------------------------------
# Per-table loaders
# ---------------------------------------------------------------------------

def load_financial(engine, csv_path: str) -> int:
    """Upsert financial profiles from CSV into financial_profiles table."""
    df = _read_csv(csv_path)
    df = _select_and_rename(df, FINANCIAL_COLS)

    conflict_cols = ["user_id"]
    update_cols = [c for c in df.columns if c not in conflict_cols]

    return _upsert_df(engine, df, "financial_profiles", conflict_cols, update_cols)


def load_products(engine, jsonl_path: str) -> int:
    """Upsert products from JSONL into products table."""
    df = _read_jsonl(jsonl_path)
    df = _select_and_rename(df, PRODUCT_COLS)

    # Ensure text columns are strings
    for col in ["description", "features"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Serialize details as JSON string for JSONB column
    df = _ensure_jsonb(df, "details")

    conflict_cols = ["product_id"]
    update_cols = [c for c in df.columns if c not in conflict_cols]

    return _upsert_df(engine, df, "products", conflict_cols, update_cols)


def load_reviews(engine, jsonl_path: str) -> int:
    """Upsert reviews from JSONL into reviews table."""
    df = _read_jsonl(jsonl_path)
    df = _select_and_rename(df, REVIEW_COLS)

    # Type fixes
    if "verified_purchase" in df.columns:
        df["verified_purchase"] = df["verified_purchase"].astype(bool)
    if "helpful_vote" in df.columns:
        df["helpful_vote"] = df["helpful_vote"].fillna(0).astype(int)

    # Filter out reviews whose product_id doesn't exist in the products table
    # to avoid ForeignKeyViolation errors
    with engine.connect() as conn:
        existing_ids = pd.read_sql(
            text("SELECT product_id FROM products"), conn
        )["product_id"].tolist()
    existing_ids_set = set(existing_ids)
    before_count = len(df)
    df = df[df["product_id"].isin(existing_ids_set)]
    dropped = before_count - len(df)
    if dropped:
        logger.warning(
            "Dropped %d orphaned reviews (product_id not in products table)",
            dropped,
        )

    conflict_cols = ["user_id", "product_id"]
    update_cols = [c for c in df.columns if c not in conflict_cols]

    return _upsert_df(engine, df, "reviews", conflict_cols, update_cols)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def load_all(
    financial_path: str,
    products_path: str,
    reviews_path: str,
    env: str = "dev",
):
    """
    Load all three datasets into PostgreSQL in FK-safe order.
    Uses upsert (INSERT ... ON CONFLICT DO UPDATE) — no data is deleted.
    Products first, then reviews (reviews reference products).
    """
    engine = get_engine(env)
    create_tables(engine)

    n_fin  = load_financial(engine, financial_path)
    n_prod = load_products(engine, products_path)
    n_rev  = load_reviews(engine, reviews_path)

    summary = {
        "financial_profiles": n_fin,
        "products": n_prod,
        "reviews": n_rev,
    }
    logger.info("Upload complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# CLI - optional, for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from src.utils import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(description="Upload featured data to PostgreSQL (upsert)")
    parser.add_argument("--financial", required=True, help="Path to financial CSV (e.g., data/featured/financial_featured.csv)")
    parser.add_argument("--products", required=True, help="Path to products JSONL (e.g., data/featured/products_featured.jsonl)")
    parser.add_argument("--reviews", required=True, help="Path to reviews JSONL (e.g., data/featured/reviews_featured.jsonl)")
    parser.add_argument("--env", default="dev", choices=["dev", "prod"])
    args = parser.parse_args()

    result = load_all(
        args.financial,
        args.products,
        args.reviews,
        env=args.env,
    )
    print("Upload summary:", result)
