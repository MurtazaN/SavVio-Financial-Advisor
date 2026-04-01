import sys
import os

import numpy as np
import pandas as pd
import pytest


from features.product_features import (
    compute_category_stats,
    compute_product_features,
    compute_product_features_batch,
)


def _make_products_df():
    return pd.DataFrame(
        {
            "product_id": ["P1", "P2", "P3"],
            "product_name": ["A", "B", "C"],
            "price": [10.0, 100.0, 1000.0],
            "average_rating": [4.5, 3.0, 2.0],
            "rating_number": [5, 50, 500],
            "rating_variance": [0.1, 0.5, 1.5],
            "description": ["", "", ""],
            "features": ["", "", ""],
            "details": ["", "", ""],
            "category": ["cat1", "cat1", "cat1"],
        }
    )


class TestProductFeaturesSingle:
    def test_compute_category_stats(self):
        df = _make_products_df()
        stats = compute_category_stats(df)
        assert "cat1" in stats
        s = stats["cat1"]
        assert s["min_price"] == 10.0
        assert s["max_price"] == 1000.0
        assert pytest.approx(s["mean_rating"], rel=1e-6) == df["average_rating"].mean()

    def test_single_product_features_shapes(self):
        df = _make_products_df()
        stats = compute_category_stats(df)
        max_rating_number = float(df["rating_number"].max())
        row = df.iloc[1]
        feats = compute_product_features(row, stats, max_rating_number)
        d = feats.to_dict()
        assert set(d.keys()) == {
            "value_density",
            "review_confidence",
            "rating_polarization",
            "quality_risk_score",
            "cold_start_flag",
            "price_category_rank",
            "category_rating_deviation",
        }

    def test_cold_start_flag_threshold(self):
        df = _make_products_df()
        df.loc[0, "rating_number"] = 5  # cold-start
        stats = compute_category_stats(df)
        max_rating_number = float(df["rating_number"].max())
        feats = compute_product_features(df.iloc[0], stats, max_rating_number)
        assert feats.cold_start_flag == 1
        feats2 = compute_product_features(df.iloc[2], stats, max_rating_number)
        assert feats2.cold_start_flag == 0


class TestProductFeaturesBatch:
    def test_batch_adds_columns(self):
        df = _make_products_df()
        out = compute_product_features_batch(df)
        for col in [
            "value_density",
            "review_confidence",
            "rating_polarization",
            "quality_risk_score",
            "cold_start_flag",
            "price_category_rank",
            "category_rating_deviation",
        ]:
            assert col in out.columns

    def test_price_category_rank_range(self):
        df = _make_products_df()
        out = compute_product_features_batch(df)
        assert (out["price_category_rank"] >= 0.0).all()
        assert (out["price_category_rank"] <= 1.0 + 1e-3).all()

    def test_handles_single_category_products(self):
        df = _make_products_df()
        df["category"] = "only_cat"
        out = compute_product_features_batch(df)
        # All in same category; ranks should still be finite.
        assert np.isfinite(out["price_category_rank"]).all()

