from dataclasses import dataclass, asdict
from typing import Dict, Any

import numpy as np
import pandas as pd


EPS = 1e-3


@dataclass
class ProductFeatures:
    value_density: float
    review_confidence: float
    rating_polarization: float
    quality_risk_score: float
    cold_start_flag: int
    price_category_rank: float
    category_rating_deviation: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_category_stats(products_df: pd.DataFrame) -> Dict[Any, Dict[str, float]]:
    """
    Pre-compute per-category statistics required for PCR and CRD.

    Returns a mapping:
        category -> {
            "min_price": float,
            "max_price": float,
            "mean_rating": float,
        }

    Also returns the global max_rating_number separately.
    """
    if "category" not in products_df.columns:
        raise ValueError("products_df must contain a 'category' column.")

    grouped = products_df.groupby("category", dropna=False)
    stats = {}
    for cat, g in grouped:
        stats[cat] = {
            "min_price": float(g["price"].min()),
            "max_price": float(g["price"].max()),
            "mean_rating": float(g["average_rating"].mean()),
        }
    return stats


def _compute_single_product_features(
    row: pd.Series,
    category_stats: Dict[Any, Dict[str, float]],
    max_rating_number: float,
) -> ProductFeatures:
    price = float(row["price"])
    avg_rating = float(row["average_rating"])
    rating_number = float(row["rating_number"])
    rating_variance = float(row["rating_variance"])
    category = row.get("category")

    # Value Density (VD)
    value_density = avg_rating / np.log(price + 1.0)

    # Review Confidence (RC)
    review_confidence = np.log(rating_number + 1.0) / np.log(max_rating_number + 1.0) if max_rating_number > 0 else 0.0

    # Rating Polarization (RP)
    denom = avg_rating * (5.0 - avg_rating) + EPS
    rating_polarization = rating_variance / denom

    # Quality Risk Score (QRS)
    quality_risk_score = (5.0 - avg_rating) * (1.0 - review_confidence)

    # Cold Start Flag (CSF)
    cold_start_flag = 1 if rating_number < 10 else 0

    # Category-based stats
    cat_stat = category_stats.get(category)
    if cat_stat is None:
        # Fallback: treat as its own category
        cat_min = price
        cat_max = price
        cat_mean_rating = avg_rating
    else:
        cat_min = cat_stat["min_price"]
        cat_max = cat_stat["max_price"]
        cat_mean_rating = cat_stat["mean_rating"]

    # Price Category Rank (PCR)
    price_category_rank = (price - cat_min) / (cat_max - cat_min + EPS)

    # Category Rating Deviation (CRD)
    category_rating_deviation = avg_rating - cat_mean_rating

    return ProductFeatures(
        value_density=value_density,
        review_confidence=review_confidence,
        rating_polarization=rating_polarization,
        quality_risk_score=quality_risk_score,
        cold_start_flag=cold_start_flag,
        price_category_rank=price_category_rank,
        category_rating_deviation=category_rating_deviation,
    )


def compute_product_features(product_row: pd.Series, category_stats: Dict[Any, Dict[str, float]], max_rating_number: float) -> ProductFeatures:
    """
    Public single-row API mirroring the training-time batch computation.
    """
    return _compute_single_product_features(product_row, category_stats, max_rating_number)


def compute_product_features_batch(products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Batch API: computes all 7 product features for every product row.

    Adds the following columns to a copy of products_df and returns it:
        value_density, review_confidence, rating_polarization,
        quality_risk_score, cold_start_flag, price_category_rank,
        category_rating_deviation.
    """
    if not {"price", "average_rating", "rating_number", "rating_variance"}.issubset(products_df.columns):
        missing = {"price", "average_rating", "rating_number", "rating_variance"} - set(products_df.columns)
        raise ValueError(f"products_df missing required columns: {missing}")

    category_stats = compute_category_stats(products_df)
    max_rating_number = float(products_df["rating_number"].max() or 0.0)

    def _apply(row: pd.Series) -> pd.Series:
        feats = _compute_single_product_features(row, category_stats, max_rating_number)
        return pd.Series(feats.to_dict())

    features_df = products_df.apply(_apply, axis=1)
    return pd.concat([products_df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

