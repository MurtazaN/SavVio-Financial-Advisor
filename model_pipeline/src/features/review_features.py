from dataclasses import dataclass, asdict
from typing import Dict, Any

import numpy as np
import pandas as pd

EPS = 1e-3


@dataclass
class ReviewFeatures:
    verified_purchase_ratio: float
    helpful_concentration: float
    sentiment_spread: float
    review_depth_score: float
    reviewer_diversity: float
    extreme_rating_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _compute_single_review_features(reviews_for_product: pd.DataFrame) -> ReviewFeatures:
    if reviews_for_product.empty:
        # No reviews — neutral / zeroed features.
        return ReviewFeatures(
            verified_purchase_ratio=0.0,
            helpful_concentration=0.0,
            sentiment_spread=0.0,
            review_depth_score=0.0,
            reviewer_diversity=0.0,
            extreme_rating_ratio=0.0,
        )

    n = float(len(reviews_for_product))

    verified = reviews_for_product.get("verified_purchase", pd.Series([False] * len(reviews_for_product))).astype(bool)
    helpful = reviews_for_product.get("helpful_vote", pd.Series([0] * len(reviews_for_product))).astype(float)
    rating = reviews_for_product.get("rating", pd.Series([0] * len(reviews_for_product))).astype(float)
    review_text = reviews_for_product.get("review_text", pd.Series([""] * len(reviews_for_product))).astype(str)
    user_id = reviews_for_product.get("user_id", pd.Series([None] * len(reviews_for_product)))

    # Verified Purchase Ratio (VPR)
    verified_purchase_ratio = float(verified.sum()) / n

    # Helpful Concentration (HC)
    total_helpful = float(helpful.sum())
    max_helpful = float(helpful.max()) if len(helpful) > 0 else 0.0
    helpful_concentration = max_helpful / (total_helpful + EPS)

    # Sentiment Spread (SS)
    positive = float((rating >= 4).sum())
    negative = float((rating <= 2).sum())
    sentiment_spread = (positive - negative) / n

    # Review Depth Score (RDS)
    word_counts = review_text.str.split().apply(len)
    mean_words = float(word_counts.mean())
    review_depth_score = min(mean_words / 100.0, 1.0)

    # Reviewer Diversity (RD)
    unique_users = float(user_id.nunique(dropna=True))
    reviewer_diversity = unique_users / n

    # Extreme Rating Ratio (ERR)
    extreme = float(((rating == 1) | (rating == 5)).sum())
    extreme_rating_ratio = extreme / n

    return ReviewFeatures(
        verified_purchase_ratio=verified_purchase_ratio,
        helpful_concentration=helpful_concentration,
        sentiment_spread=sentiment_spread,
        review_depth_score=review_depth_score,
        reviewer_diversity=reviewer_diversity,
        extreme_rating_ratio=extreme_rating_ratio,
    )


def compute_review_features(reviews_for_product: pd.DataFrame) -> ReviewFeatures:
    """
    Public single-product API.
    """
    return _compute_single_review_features(reviews_for_product)


def compute_review_features_batch(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Batch API: groupby product_id and compute 6 review features.

    Returns a DataFrame indexed by product_id with 6 columns.
    """
    if "product_id" not in reviews_df.columns:
        raise ValueError("reviews_df must contain a 'product_id' column.")

    grouped = reviews_df.groupby("product_id", dropna=False)

    def _apply(group: pd.DataFrame) -> pd.Series:
        feats = _compute_single_review_features(group)
        return pd.Series(feats.to_dict())

    return grouped.apply(_apply, include_groups=False)

