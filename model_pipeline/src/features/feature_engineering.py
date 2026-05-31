"""
Feature Engineering — Model Pipeline (Phase 2).

Transforms raw scenario pairs into a model-ready feature matrix with
deterministic GREEN/YELLOW/RED labels.

Pipeline:
    1. generate_training_data() — Load data, sample pairs, compute features, label
    2. transform_features()     — Impute, encode, scale, drop non-features
    3. build_feature_matrix()   — Orchestrator calls 1 then 2 and returns (X, y, scenarios_raw)

Input:  PostgreSQL tables — financial_profiles, products, reviews
Output: Feature matrix (X), label vector (y), raw scenarios CSV
"""

import os
import logging

import pandas as pd

from config import Config
from features.training_data_generator import generate_scenarios
from features.affordability import add_affordability_features
from features.feature_preprocessing import FeaturePipeline
from deterministic_engine.labeling_pipeline import apply_deterministic_labels
from data.db_loader import load_financial_profiles, load_products, load_reviews
from data.bias_mitigation import assert_label_balance


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature Computation & Labeling
# ---------------------------------------------------------------------------

def compute_scenario_features(scenarios: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 6 financial features for each user-product scenario pair.

    Takes a raw scenario DataFrame (user columns + product columns) and
    adds: affordability_score, price_to_income_ratio, residual_utility_score,
    savings_to_price_ratio, net_worth_indicator, credit_risk_indicator.
    """
    df = add_affordability_features(scenarios, product_price_col="product_price")
    logger.info("Computed 6 financial features for %d scenarios.", len(df))
    return df


# ---------------------------------------------------------------------------
# Functions for Downstream Use
# ---------------------------------------------------------------------------

def generate_training_data(
    financial_df=None,
    products_df=None,
    reviews_df=None,
    n_scenarios=None,
    random_state=None,
    output_path=None,
):
    """
    End-to-end training data creation: sample pairs → compute features → label.

    Steps:
        1. Load data from PostgreSQL (if DataFrames not provided).
        2. Sample user-product pairs via generate_scenarios().
        3. Compute 6 financial features.
        4. Apply Layer 1 + Layer 2 deterministic labels.
        5. Save labeled scenarios to CSV.
    """
    if n_scenarios is None:
        n_scenarios = Config.N_SCENARIOS
    if random_state is None:
        random_state = Config.RANDOM_STATE

    if financial_df is None or products_df is None or reviews_df is None:
        logger.info("Loading data from PostgreSQL...")
        financial_df = financial_df if financial_df is not None else load_financial_profiles()
        products_df = products_df if products_df is not None else load_products()
        reviews_df = reviews_df if reviews_df is not None else load_reviews()

    logger.info("Generating %d scenarios (stratified sampling)...", n_scenarios)
    scenarios_raw = generate_scenarios(
        financial_df,
        products_df,
        n_scenarios=n_scenarios,
        random_state=random_state,
        reviews_df=reviews_df,       #  pass reviews for full mitigation
        apply_mitigations=True,       # explicit flag
    )

    scenarios_raw = compute_scenario_features(scenarios_raw)

    scenarios_raw = apply_deterministic_labels(scenarios_raw, products_df, reviews_df)

    assert_label_balance(scenarios_raw, label_col="final_recommendation")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        scenarios_raw.to_csv(output_path, index=False)
        logger.info("Saved raw scenarios to %s", output_path)

    y = scenarios_raw[Config.LABEL_COL].copy()
    logger.info("Training data generated — %d scenarios", len(scenarios_raw))

    return scenarios_raw, y


def transform_features(
    scenarios: pd.DataFrame,
    y: pd.Series,
    is_training: bool = True,
):
    """
    Transform raw scenarios into a model-ready feature matrix using the pipeline.
    """
    pipeline = FeaturePipeline()
    if is_training:
        X = pipeline.fit_transform(scenarios)
    else:
        X = pipeline.transform(scenarios)

    logger.info("Feature matrix built — X shape: %s", X.shape)
    return X, y


def build_training_data(
    financial_df: pd.DataFrame = None,
    products_df: pd.DataFrame = None,
    reviews_df: pd.DataFrame = None,
    n_scenarios: int = None,
    output_path: str = Config.SCENARIO_OUTPUT_PATH,
    is_training: bool = True,
):
    """
    End-to-end feature engineering pipeline. Orchestrates loading then transformation.
    """
    scenarios_raw, y = generate_training_data(
        financial_df=financial_df,
        products_df=products_df,
        reviews_df=reviews_df,
        n_scenarios=n_scenarios,
        output_path=output_path,
    )
    X, y = transform_features(scenarios_raw, y, is_training=is_training)

    logger.info("build_feature_matrix complete — X: %s", X.shape)
    return X, y, scenarios_raw
