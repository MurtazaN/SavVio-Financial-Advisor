"""
Training Data Generator — Stratified Scenario Sampling.

Provides a library API that takes real financial profiles and products,
samples user–product pairs using a single stratified strategy, and
returns raw (user, product) scenarios suitable as input to the feature
engineering and labeling pipeline.

Sampling strategy:
    - stratified (only): Equal representation across income × price
      bracket combinations (3 income × 3 price = 9 cells). This ensures
      good coverage of edge cases for model training.

Library usage:
    from features.training_data_generator import generate_scenarios
    scenarios_df = generate_scenarios(financial_df, products_df, n_scenarios=50000)

CLI usage (from model_pipeline/, raw sampling only):
    PYTHONPATH=$PYTHONPATH:$(pwd)/src:/path/to/savviocore/src \\
        python3 src/features/training_data_generator.py \\
        --n-scenarios 50000 \\
        --output data/raw_scenarios.csv
"""

import argparse
import logging
import os
import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# Ensure project root (src) is on sys.path when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.db_loader import load_financial_profiles, load_products, load_reviews

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bracket definitions
# ---------------------------------------------------------------------------

INCOME_BINS: List[float] = [0, 3_000, 7_000, float("inf")]
INCOME_LABELS: List[str] = ["low", "mid", "high"]

PRICE_BINS: List[float] = [100, 500, 1_500, float("inf")]
PRICE_LABELS: List[str] = ["budget", "mid", "premium"]

# ---------------------------------------------------------------------------
# Stratified sampling (single strategy)
# ---------------------------------------------------------------------------

def _sample_stratified(
    financial_profiles: pd.DataFrame,
    products: pd.DataFrame,
    n_scenarios: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified pairing across income × price bracket combinations.

    Divides users into 3 income brackets and products into 3 price
    brackets, then samples equally from each of the 9 (income, price)
    cells so edge-case combinations are well represented.

    Empty cells are skipped and their quota is redistributed evenly
    across remaining cells.

    Args:
        financial_profiles: DataFrame containing user financial profiles.
        products: DataFrame containing product data.
        n_scenarios: Total number of scenarios to generate.
        rng: NumPy random generator instance for reproducibility.

    Returns:
        A tuple of (sampled_users_df, sampled_products_df), both of length n_scenarios.
    """
    income_bracket = pd.cut(
        financial_profiles["monthly_income"],
        bins=INCOME_BINS, labels=INCOME_LABELS, include_lowest=True,
    )
    price_bracket = pd.cut(
        products["price"],
        bins=PRICE_BINS, labels=PRICE_LABELS, include_lowest=True,
    )

    income_groups = {
        label: financial_profiles[income_bracket == label]
        for label in INCOME_LABELS
    }
    price_groups = {
        label: products[price_bracket == label]
        for label in PRICE_LABELS
    }

    valid_cells = [
        (ig, pg)
        for ig in INCOME_LABELS
        for pg in PRICE_LABELS
        if len(income_groups[ig]) > 0 and len(price_groups[pg]) > 0
    ]

    if not valid_cells:
        raise ValueError(
            "No valid (income, price) bracket combinations — check your data."
        )

    base, remainder = divmod(n_scenarios, len(valid_cells))
    cell_counts = [base + (1 if i < remainder else 0) for i in range(len(valid_cells))]

    user_chunks: List[pd.DataFrame] = []
    prod_chunks: List[pd.DataFrame] = []

    for (ig_label, pg_label), count in zip(valid_cells, cell_counts):
        seed = int(rng.integers(0, 2**31))

        user_sample = income_groups[ig_label].sample(
            n=count, replace=True, random_state=seed,
        )
        prod_group = price_groups[pg_label]
        weights = None
        if "sample_weight" in prod_group.columns:
            w = pd.to_numeric(prod_group["sample_weight"], errors="coerce").fillna(1.0)
            # Guard against invalid/non-positive weights.
            if (w > 0).any():
                weights = w
        prod_sample = prod_group.sample(
            n=count, replace=True, random_state=seed + 1, weights=weights,
        )
        user_chunks.append(user_sample)
        prod_chunks.append(prod_sample)

        logger.debug(
            "Cell (%s income, %s price): %d scenarios",
            ig_label, pg_label, count,
        )

    users = pd.concat(user_chunks, ignore_index=True)
    prods = pd.concat(prod_chunks, ignore_index=True)

    logger.info(
        "Stratified sampling: %d valid cells out of 9, %d total scenarios",
        len(valid_cells), len(users),
    )
    return users, prods


# ---------------------------------------------------------------------------
# Pair merging
# ---------------------------------------------------------------------------

def _merge_pairs(
    users: pd.DataFrame,
    prods: pd.DataFrame,
) -> pd.DataFrame:
    """Combine user and product DataFrames into scenario rows."""
    scenarios = users.copy()
    scenarios["product_id"] = prods["product_id"].values
    scenarios["product_price"] = prods["price"].values

    for col in ("average_rating", "rating_number", "rating_variance", "category"):
        if col in prods.columns:
            scenarios[col] = prods[col].values

    return scenarios


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
from data.bias_mitigation import apply_all_mitigations
def generate_scenarios(
    financial_profiles: pd.DataFrame,
    products: pd.DataFrame,
    n_scenarios: int = 10_000,
    random_state: int = 42,
    reviews_df: Optional[pd.DataFrame] = None,
    apply_mitigations: bool = False,
) -> pd.DataFrame:
    """
    Generate raw user-product scenario pairs.

    Returns a DataFrame with user columns merged with product columns,
    ready for feature computation and labeling by feature_engineering.py.

    Args:
        financial_profiles: DataFrame from the financial_profiles table.
        products: DataFrame from the products table.
        n_scenarios: Target number of (user, product) rows to generate.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with one row per (user, product) pair containing
        raw user columns and product columns.  No computed features
        or labels are included.
    """
    rng = np.random.default_rng(random_state)

    if apply_mitigations:
        financial_profiles, products, _ = apply_all_mitigations(
            financial_profiles,
            products,
            reviews_df if reviews_df is not None else pd.DataFrame(),
            random_state=random_state,
        )

    # Single, stratified sampling strategy.
    users, prods = _sample_stratified(
        financial_profiles, products, n_scenarios, rng,
    )
    scenarios = _merge_pairs(users, prods)

    logger.info("Generated %d scenario pairs.", len(scenarios))
    return scenarios


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sample raw (user, product) scenarios using stratified sampling."
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=Config.N_SCENARIOS,
        help=f"Number of scenarios to generate (default: {Config.N_SCENARIOS})",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=Config.RANDOM_STATE,
        help=f"Random seed (default: {Config.RANDOM_STATE})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=Config.SCENARIO_OUTPUT_PATH,
        help="Output CSV path for raw sampled scenarios.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SavVio — Raw Scenario Sampler (stratified)")
    print("=" * 60)

    print("\n[1/2] Loading data from PostgreSQL...")
    fin_df = load_financial_profiles()
    prod_df = load_products()
    rev_df = load_reviews()
    print(f"      Financial profiles: {len(fin_df):,}")
    print(f"      Products:           {len(prod_df):,}")
    print(f"      Reviews:            {len(rev_df):,}")

    # Apply pre-training bias mitigations BEFORE scenario generation
    fin_df, prod_df, rev_df = apply_all_mitigations(fin_df, prod_df, rev_df)


    print(f"\n[2/2] Sampling {args.n_scenarios:,} raw scenarios (stratified)...")
    scenarios = generate_scenarios(
        fin_df,
        prod_df,
        n_scenarios=args.n_scenarios,
        random_state=args.random_state,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    scenarios.to_csv(args.output, index=False)
    print(f"      Saved raw scenarios to: {args.output}")
    print(f"      Rows: {len(scenarios):,}, Columns: {len(scenarios.columns)}")
    print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
