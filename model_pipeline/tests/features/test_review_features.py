import sys
import os

import numpy as np
import pandas as pd


from features.review_features import (
    compute_review_features,
    compute_review_features_batch,
)


def _make_reviews_df(product_id="P1", n=10):
    return pd.DataFrame(
        {
            "product_id": [product_id] * n,
            "user_id": [f"U{i}" for i in range(n)],
            "rating": [1, 2, 3, 4, 5, 5, 1, 4, 2, 5][:n],
            "review_text": ["good product" if r >= 4 else "bad" for r in [1, 2, 3, 4, 5, 5, 1, 4, 2, 5][:n]],
            "verified_purchase": [True, False] * (n // 2) + [True] * (n % 2),
            "helpful_vote": list(range(1, n + 1)),
        }
    )


class TestReviewFeaturesSingle:
    def test_empty_reviews(self):
        df = _make_reviews_df(n=0)
        feats = compute_review_features(df)
        d = feats.to_dict()
        assert all(v == 0.0 for v in d.values())

    def test_basic_computation(self):
        df = _make_reviews_df()
        feats = compute_review_features(df)
        assert 0.0 <= feats.verified_purchase_ratio <= 1.0
        assert 0.0 <= feats.helpful_concentration <= 1.0
        assert -1.0 <= feats.sentiment_spread <= 1.0
        assert 0.0 <= feats.review_depth_score <= 1.0
        assert 0.0 <= feats.reviewer_diversity <= 1.0
        assert 0.0 <= feats.extreme_rating_ratio <= 1.0


class TestReviewFeaturesBatch:
    def test_batch_groupby_product(self):
        df = pd.concat(
            [
                _make_reviews_df("P1", n=5),
                _make_reviews_df("P2", n=7),
            ],
            ignore_index=True,
        )
        out = compute_review_features_batch(df)
        assert set(out.index) == {"P1", "P2"}
        for col in [
            "verified_purchase_ratio",
            "helpful_concentration",
            "sentiment_spread",
            "review_depth_score",
            "reviewer_diversity",
            "extreme_rating_ratio",
        ]:
            assert col in out.columns
            assert np.isfinite(out[col]).all()

