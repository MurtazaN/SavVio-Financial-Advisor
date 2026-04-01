"""
Feature Engineering for Financial Data.

Transforms preprocessed financial records into actionable risk metrics.
Calculates ratios and scores used for downstream affordability analysis.

Steps:
    1. calculate_liquid_savings()       — Derive realistic liquid cash from total wealth
    2. calculate_discretionary_income() — Income minus obligations
    3. calculate_ratios()               — STIR, EFM, DTI, MEB (STIR & EFM use liquid_savings)

Input: data/processed/financial_preprocessed.csv
Output: data/features/financial_featured.csv
"""

import os
import logging
import pandas as pd
import numpy as np

from src.features.utils import setup_logging, ensure_output_dir
from src.incremental import merge_csv

# Configure module logging.
setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SCF-based liquidity tiers (2022 Survey of Consumer Finances)
# ---------------------------------------------------------------------------

# Each tier: (income_upper_bound, liquid_pct_low, liquid_pct_high, cap_low, cap_high)
# income_upper_bound is exclusive; last tier uses np.inf.
# Sources:
#   https://www.usnews.com/banking/articles/the-average-savings-account-balance
#   https://finance.yahoo.com/news/average-amount-u-savings-accounts-195054018.html
LIQUIDITY_TIERS = [
    # (income_upper, pct_low, pct_high, cap_low,  cap_high)
    (1_500,          0.60,    0.80,     500,      3_000),      # <$1,500/mo — SCF median ~$900
    (3_000,          0.30,    0.50,     2_000,    10_000),     # $1,500–$3,000 — median ~$2,500
    (5_000,          0.15,    0.25,     5_000,    30_000),     # $3,000–$5,000 — median ~$8,000
    (8_000,          0.08,    0.15,     10_000,   60_000),     # $5,000–$8,000 — median ~$15,000
    (np.inf,         0.05,    0.10,     25_000,   150_000),    # $8,000+       — median ~$112,000
]


def calculate_liquid_savings(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Derive realistic liquid savings from total savings_balance using
    2022 SCF (Survey of Consumer Finances) data.

    Two-step correction per user:
        1. Liquid fraction: savings_balance × pct  (pct sampled uniformly within tier)
        2. Absolute cap: min(fractional, cap)       (cap sampled within tier range)

    The fraction handles the decomposition (what % is liquid), the cap handles
    unrealistic Kaggle data (a $1,500/mo earner shouldn't have $140K liquid).

    Args:
        df: DataFrame with 'monthly_income' and 'savings_balance' columns.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with new 'liquid_savings' column added.
    """
    rng = np.random.default_rng(random_state)
    n = len(df)

    income = df["monthly_income"].values
    savings = df["savings_balance"].values

    liquid = np.zeros(n, dtype=np.float64)

    for upper, pct_lo, pct_hi, cap_lo, cap_hi in LIQUIDITY_TIERS:
        mask = income < upper if upper != np.inf else np.ones(n, dtype=bool)
        # Exclude rows already assigned by a lower tier.
        mask = mask & (liquid == 0) & (savings > 0)
        count = mask.sum()
        if count == 0:
            continue

        # Step 1: Liquid fraction.
        pcts = rng.uniform(pct_lo, pct_hi, size=count)
        fractional = savings[mask] * pcts

        # Step 2: Absolute cap.
        caps = rng.uniform(cap_lo, cap_hi, size=count)
        liquid[mask] = np.minimum(fractional, caps)

    df["liquid_savings"] = np.round(liquid, 2)

    logger.info(
        "Liquid savings derived — mean: $%.2f, median: $%.2f (from savings_balance mean: $%.2f)",
        df["liquid_savings"].mean(),
        df["liquid_savings"].median(),
        df["savings_balance"].mean(),
    )
    return df


def calculate_discretionary_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Discretionary Income (Monthly Income - Monthly Expenses).
    Definition: Money left after all monthly obligations (Income - (Expenses + EMI)).
    """
    df["discretionary_income"] = df["monthly_income"] - (df["monthly_expenses"] + df["monthly_emi"])
    return df

def calculate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes financial risk ratios.
    Handles division by zero by assigning np.nan where denominators are invalid.
    """
    # Metric 1: Debt-to-income ratio.
    # Definition: Percentage of gross monthly income that goes toward paying debts.
    df["debt_to_income_ratio"] = np.where(
        df["monthly_income"] > 0,
        df["monthly_emi"] / df["monthly_income"],
        np.nan
    )

    # Metric 2: Savings-to-Income Ratio (STIR).
    # Definition: Ratio of LIQUID savings to monthly income.
    # Uses liquid_savings (SCF-adjusted) instead of raw savings_balance.
    df["saving_to_income_ratio"] = np.where(
        df["monthly_income"] > 0,
        df["liquid_savings"] / df["monthly_income"],
        np.nan
    )

    # Metric 3: Monthly Expense Burden Ratio.
    # Definition: Percentage of income consumed by all monthly outflows (living costs + debt).
    df["monthly_expense_burden_ratio"] = np.where(
        df["monthly_income"] > 0,
        (df["monthly_expenses"] + df["monthly_emi"]) / df["monthly_income"],
        np.nan
    )

    # Metric 4: Emergency Fund Months (EFM).
    # Definition: Months a user could survive on LIQUID savings if they lost income.
    # Uses liquid_savings (SCF-adjusted) instead of raw savings_balance.
    df["emergency_fund_months"] = np.where(
        (df["monthly_expenses"] + df["monthly_emi"]) > 0,
        df["liquid_savings"] / (df["monthly_expenses"] + df["monthly_emi"]),
        np.nan
    )

    return df

def run_financial_features(input_path: str, output_path: str) -> None:
    """
    Executes the financial feature engineering pipeline.
    """
    logger.info("Starting financial feature engineering pipeline...")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} records from {input_path}")

        # Compute feature columns.
        df = calculate_liquid_savings(df)
        df = calculate_discretionary_income(df)
        df = calculate_ratios(df)

        # Cleanup values before persisting output.
        # Keep undefined ratios as NaN; only normalize arithmetic output column.
        ratio_cols = ["debt_to_income_ratio", "saving_to_income_ratio", "monthly_expense_burden_ratio", "emergency_fund_months"]
        df[ratio_cols] = df[ratio_cols].replace([np.inf, -np.inf], np.nan)
        df["liquid_savings"] = df["liquid_savings"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["discretionary_income"] = df["discretionary_income"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        numeric_cols = ["liquid_savings", "discretionary_income", *ratio_cols]

        # Persist output dataset with incremental merge.
        ensure_output_dir(output_path)
        temp_output = output_path + ".new.tmp"
        df.to_csv(temp_output, index=False)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            merge_stats = merge_csv(temp_output, output_path, key_cols=["user_id"])
            os.remove(temp_output)
            logger.info("Incremental merge stats: %s", merge_stats)
        else:
            os.replace(temp_output, output_path)

        logger.info(f"Saved featured data to {output_path}")
        logger.info(f"Sample features:\n{df[numeric_cols].head()}")

    except Exception as e:
        logger.error(f"Failed to process financial features: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Default local run paths.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INPUT_FILE = os.path.join(BASE_DIR, "data/processed/financial_preprocessed.csv")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/features/financial_featured.csv")

    run_financial_features(INPUT_FILE, OUTPUT_FILE)
