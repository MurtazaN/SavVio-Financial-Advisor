"""Deterministic preprocessing pipeline for financial decision logic."""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from .utils import ensure_output_dir, get_processed_path, get_raw_path, setup_logging


LOGGER = logging.getLogger("preprocess.financial")

INPUT_FILENAME = "financial_data.csv"
OUTPUT_FILENAME = "financial_preprocessed.csv"

KEEP_COLUMNS: List[str] = [
    "user_id",
    "monthly_income",
    "monthly_expenses",
    "savings_balance",
    "has_loan",
    "loan_amount",
    "monthly_loan_payment",
    "loan_interest_rate",
    "loan_term_months",
    "credit_score",
    "employment_status",
    "region",
]

DEMOGRAPHIC_COLUMNS_DROPPED = [
    "age",
    "gender",
    "education_level",
    "job_title",
    "loan_type",
    "record_date",
    "debt_to_income_ratio",
    "savings_to_income_ratio",
]

MONETARY_FLOAT_COLUMNS: List[str] = [
    "monthly_income",
    "monthly_expenses",
    "savings_balance",
    "loan_amount",
    "monthly_emi",
    "loan_interest_rate",
]

CRITICAL_REQUIRED_COLUMNS: List[str] = [
    "monthly_income",
    "monthly_expenses",
    "savings_balance",
    "credit_score",
]

LOAN_NUMERIC_COLUMNS: List[str] = ["loan_amount", "monthly_emi", "loan_interest_rate"]


def _validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Raise a clear error when expected columns are missing from source data."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")


def _print_frame_snapshot(df: pd.DataFrame, title: str, rows: int = 5) -> None:
    """Print compact, deterministic frame preview."""
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(rows).to_string(index=False))


def _to_binary_has_loan(value: object) -> int:
    """Normalize has_loan into deterministic binary values."""
    if pd.isna(value):
        return 0

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "loan", "has_loan"}:
        return 1
    if normalized in {"0", "false", "no", "n", "none", "no_loan", ""}:
        return 0

    # Fallback: any non-empty unknown token is treated conservatively as loan present.
    return 1


def preprocess_financial_data(input_path: str, output_path: str) -> pd.DataFrame:
    """Run deterministic preprocessing and save processed financial data."""
    LOGGER.info("Loading dataset from: %s", input_path)
    df = pd.read_csv(input_path)
    # Define original columns expected in raw CSV
    RAW_EXPECTED_COLUMNS = [
        "user_id", "monthly_income_usd", "monthly_expenses_usd", "savings_usd",
        "has_loan", "loan_amount_usd", "monthly_emi_usd", "loan_interest_rate_pct",
        "loan_term_months", "credit_score", "employment_status", "region"
    ]
    _validate_required_columns(df, RAW_EXPECTED_COLUMNS)

    rows_loaded = len(df)
    LOGGER.info("Rows loaded: %d", rows_loaded)
    _print_frame_snapshot(df, title="Loaded Dataset")

    # Keep only explicitly approved fields for affordability logic.
    # Note: Using original names here before renaming
    original_keep_cols = [
        "user_id", "monthly_income_usd", "monthly_expenses_usd", "savings_usd",
        "has_loan", "loan_amount_usd", "monthly_emi_usd", "loan_interest_rate_pct",
        "loan_term_months", "credit_score", "employment_status", "region"
    ]
    df = df.loc[:, original_keep_cols].copy()
    LOGGER.info(
        "Dropped non-required columns including demographics (%s).",
        ", ".join(DEMOGRAPHIC_COLUMNS_DROPPED),
    )
    LOGGER.info(
        "Demographic fields are removed to reduce bias risk and keep decisions based on financial behavior only."
    )

    # Contextual renaming map
    rename_map = {
        "monthly_income_usd": "monthly_income",
        "monthly_expenses_usd": "monthly_expenses",
        "savings_usd": "savings_balance",
        "loan_amount_usd": "loan_amount",
        "monthly_emi_usd": "monthly_emi",
        "loan_interest_rate_pct": "loan_interest_rate",
    }
    df.rename(columns=rename_map, inplace=True)

    # Enforce numeric types deterministically; invalid parsing becomes NaN for explicit handling later.
    for col in MONETARY_FLOAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["loan_term_months"] = pd.to_numeric(df["loan_term_months"], errors="coerce").astype("Int64")

    df["credit_score"] = pd.to_numeric(df["credit_score"], errors="coerce")
    df["has_loan"] = df["has_loan"].apply(_to_binary_has_loan).astype(np.int8)

    # Remove duplicate user records deterministically by keeping first occurrence.
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["user_id"], keep="first")
    duplicates_removed = before_dedup - len(df)
    LOGGER.info("Duplicates removed (by user_id): %d", duplicates_removed)

    # Strict missing policy for critical financial integrity columns.
    before_missing_drop = len(df)
    df = df.dropna(subset=CRITICAL_REQUIRED_COLUMNS)
    missing_rows_dropped = before_missing_drop - len(df)
    LOGGER.info("Rows dropped (missing critical fields): %d", missing_rows_dropped)

    # Non-critical has_loan defaults to no-loan.
    df["has_loan"] = df["has_loan"].fillna(0).astype(np.int8)

    # Only set loan metrics to zero when there is no loan.
    no_loan_mask = df["has_loan"] == 0
    for col in LOAN_NUMERIC_COLUMNS:
        df.loc[no_loan_mask, col] = df.loc[no_loan_mask, col].fillna(0.0)

    # Finalize credit score as int after missing values are removed.
    df["credit_score"] = df["credit_score"].astype(int)

    # Apply range validation rules one by one and log removals per rule.
    rules: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "monthly_income < 0": lambda frame: frame["monthly_income"] < 0,
        "monthly_expenses < 0": lambda frame: frame["monthly_expenses"] < 0,
        "savings_balance < 0": lambda frame: frame["savings_balance"] < 0,
        "credit_score not in [300, 850]": lambda frame: ~frame["credit_score"].between(300, 850, inclusive="both"),
    }

    total_range_violations_dropped = 0
    for rule_name, rule_fn in rules.items():
        violation_mask = rule_fn(df)
        removed_for_rule = int(violation_mask.sum())
        if removed_for_rule:
            df = df.loc[~violation_mask].copy()
            total_range_violations_dropped += removed_for_rule
        LOGGER.info("Rows dropped (range rule '%s'): %d", rule_name, removed_for_rule)

    LOGGER.info("Rows dropped (range violations total): %d", total_range_violations_dropped)

    _print_frame_snapshot(df, title="Final Preprocessed Dataset")

    ensure_output_dir(output_path)

    # --- Incremental merge: merge new output with existing preprocessed file ---
    temp_output = output_path + ".new.tmp"
    df.to_csv(temp_output, index=False)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        from src.incremental import merge_csv
        merge_stats = merge_csv(temp_output, output_path, key_cols=["user_id"])
        os.remove(temp_output)
        LOGGER.info("Incremental merge stats: %s", merge_stats)
    else:
        # First run — just rename temp to output.
        os.replace(temp_output, output_path)

    LOGGER.info("Saved preprocessed dataset to: %s", output_path)
    LOGGER.info("Final row count: %d", len(df))

    return df


def main() -> None:
    """Entry point for running deterministic financial preprocessing."""
    setup_logging()
    input_path = get_raw_path(INPUT_FILENAME)
    output_path = get_processed_path(OUTPUT_FILENAME)
    preprocess_financial_data(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    main()
