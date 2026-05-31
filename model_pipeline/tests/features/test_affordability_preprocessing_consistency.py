"""Smoke tests for shared affordability + preprocessing parity across training and inference."""

import os
import sys

import numpy as np
import pandas as pd
import pytest


import config
from features.affordability import add_affordability_features
from features.financial_features import compute_affordability
from features.feature_preprocessing import FeaturePipeline



def _base_rows():
    return pd.DataFrame(
        {
            "monthly_income": [5000.0, 0.0, 7000.0],
            "discretionary_income": [1200.0, -50.0, 2200.0],
            "liquid_savings": [8000.0, 100.0, 12000.0],
            "monthly_expenses": [2800.0, 0.0, 3000.0],
            "monthly_emi": [1000.0, 0.0, 1200.0],
            "loan_amount": [15000.0, 0.0, 5000.0],
            "credit_score": [720, 0, 640],
            "product_price": [200.0, 50.0, 900.0],
            "employment_status": ["employed", None, "self-employed"],
            "has_loan": [True, False, True],
            "region": ["east", "west", None],
            "user_id": ["u1", "u2", "u3"],
            "product_id": ["p1", "p2", "p3"],
            "product_name": ["A", "B", "C"],
            "description": ["d", "d", "d"],
            "features": ["f", "f", "f"],
            "details": ["x", "x", "x"],
            "category": ["c", "c", "c"],
            "final_recommendation": ["GREEN", "RED", "YELLOW"],
            "price": [200.0, 50.0, 900.0],
            "average_rating": [4.5, 4.1, 3.8],
            "rating_number": [100, 10, 20],
            "rating_variance": [0.2, np.nan, 0.4],
            "debt_to_income_ratio": [0.2, 0.1, 0.3],
            "saving_to_income_ratio": [1.6, 0.4, 2.0],
            "monthly_expense_burden_ratio": [0.7, 0.9, 0.6],
            "emergency_fund_months": [5.0, 1.0, 8.0],
        }
    )


class TestAffordabilityConsistency:
    def test_batch_and_single_pair_match(self):
        rows = _base_rows()
        batch = add_affordability_features(rows, product_price_col="product_price")

        for idx, row in rows.iterrows():
            single = compute_affordability(
                {
                    "monthly_income": row["monthly_income"],
                    "discretionary_income": row["discretionary_income"],
                    "liquid_savings": row["liquid_savings"],
                    "monthly_expenses": row["monthly_expenses"],
                    "monthly_emi": row["monthly_emi"],
                    "loan_amount": row["loan_amount"],
                    "credit_score": row["credit_score"],
                },
                product_price=row["product_price"],
            ).to_dict()

            for key, expected in single.items():
                if key == "affordability_score_unreliable":
                    continue
                got = batch.loc[idx, key]
                if expected is None:
                    assert pd.isna(got)
                else:
                    assert got == pytest.approx(expected, abs=1e-4)


class TestFeaturePipelinePersistence:
    def test_saved_pipeline_reproduces_transform(self, tmp_path):
        original_model_dir = config.Config.MODEL_SAVE_DIR
        config.Config.MODEL_SAVE_DIR = str(tmp_path)

        try:
            rows = add_affordability_features(_base_rows(), product_price_col="product_price")
            pipeline_fit = FeaturePipeline()
            transformed_train = pipeline_fit.fit_transform(rows)

            pipeline_infer = FeaturePipeline()
            transformed_loaded = pipeline_infer.transform(rows)

            pd.testing.assert_frame_equal(
                transformed_train.sort_index(axis=1),
                transformed_loaded.sort_index(axis=1),
                check_exact=False,
                rtol=1e-9,
                atol=1e-9,
            )
        finally:
            config.Config.MODEL_SAVE_DIR = original_model_dir
