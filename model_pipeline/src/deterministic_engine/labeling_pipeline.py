"""Deterministic labeling orchestration for training scenarios.

This module owns the Layer 1 + Layer 2 flow so feature engineering can
focus on feature computation and transformation only.
"""

from __future__ import annotations

import logging

import pandas as pd

from deterministic_engine.downgrade_engine import DowngradeEngine
from deterministic_engine.financial_engine import DecisionEngine
from features.product_features import ProductFeatures, compute_product_features_batch
from features.review_features import ReviewFeatures, compute_review_features_batch

logger = logging.getLogger(__name__)


PRODUCT_FEATURE_COLS = [
    "value_density",
    "review_confidence",
    "rating_polarization",
    "quality_risk_score",
    "cold_start_flag",
    "price_category_rank",
    "category_rating_deviation",
]

REVIEW_FEATURE_COLS = [
    "verified_purchase_ratio",
    "helpful_concentration",
    "sentiment_spread",
    "review_depth_score",
    "reviewer_diversity",
    "extreme_rating_ratio",
]


def _attach_layer2_features(
    scenarios: pd.DataFrame,
    products_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach product/review features used by Layer 2 downgrade rules."""
    unique_prods = products_df.drop_duplicates(subset=["product_id"]).copy()
    product_feats_df = compute_product_features_batch(unique_prods).set_index("product_id")
    review_feats_df = compute_review_features_batch(reviews_df)

    merged = scenarios.merge(
        product_feats_df[PRODUCT_FEATURE_COLS],
        left_on="product_id",
        right_index=True,
        how="left",
    )
    merged = merged.merge(
        review_feats_df[REVIEW_FEATURE_COLS],
        left_on="product_id",
        right_index=True,
        how="left",
    )

    # Keep downgrade rules stable even when a product has no reviews.
    merged[REVIEW_FEATURE_COLS] = merged[REVIEW_FEATURE_COLS].fillna(0.0)
    return merged


def _apply_layer2_downgrade(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Apply downgrade rules to a frame that already has Layer 2 features."""
    downgrade_engine = DowngradeEngine()

    def _downgrade(row: pd.Series) -> pd.Series:
        product_features = ProductFeatures(
            value_density=float(row["value_density"]),
            review_confidence=float(row["review_confidence"]),
            rating_polarization=float(row["rating_polarization"]),
            quality_risk_score=float(row["quality_risk_score"]),
            cold_start_flag=int(row["cold_start_flag"]),
            price_category_rank=float(row["price_category_rank"]),
            category_rating_deviation=float(row["category_rating_deviation"]),
        )
        review_features = ReviewFeatures(
            verified_purchase_ratio=float(row["verified_purchase_ratio"]),
            helpful_concentration=float(row["helpful_concentration"]),
            sentiment_spread=float(row["sentiment_spread"]),
            review_depth_score=float(row["review_depth_score"]),
            reviewer_diversity=float(row["reviewer_diversity"]),
            extreme_rating_ratio=float(row["extreme_rating_ratio"]),
        )

        result = downgrade_engine.evaluate(
            financial_label=row["_l1_label"],
            product_features=product_features,
            review_features=review_features,
        )
        return pd.Series(
            {
                "final_recommendation": result.final_label,
                "downgraded": int(result.final_label != row["_l1_label"]),
            }
        )

    downgraded = scenarios.apply(_downgrade, axis=1)
    scenarios["final_recommendation"] = downgraded["final_recommendation"]
    scenarios["downgraded"] = downgraded["downgraded"]
    return scenarios


def apply_deterministic_labels(
    scenarios: pd.DataFrame,
    products_df: pd.DataFrame,
    reviews_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply Layer 1 financial labels and optional Layer 2 downgrade labels."""
    engine = DecisionEngine()

    labeled = scenarios.copy()
    labeled["_l1_label"] = labeled.apply(engine.decide_row, axis=1)
    labeled["final_recommendation"] = labeled["_l1_label"]

    if reviews_df is None:
        labeled["downgraded"] = 0
        return labeled.drop(columns=["_l1_label"], errors="ignore")

    logger.info("Applying deterministic Layer 2 downgrade rules...")
    labeled = _attach_layer2_features(labeled, products_df, reviews_df)
    labeled = _apply_layer2_downgrade(labeled)
    return labeled.drop(columns=["_l1_label"], errors="ignore")
