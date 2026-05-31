import pandas as pd
import numpy as np

from data.bias_mitigation import (
    apply_all_mitigations,
    oversample_unverified_purchases,
    oversample_minority_employment,
)


def _financial_df():
    return pd.DataFrame(
        {
            "user_id": [f"U{i}" for i in range(20)],
            "monthly_income": [2500] * 20,
            "monthly_emi": [500] * 20,
            "employment_status": ["Employed"] * 16 + ["Unemployed"] * 2 + ["Student"] * 2,
        }
    )


def _products_df():
    return pd.DataFrame(
        {
            "product_id": [f"P{i}" for i in range(10)],
            "price": [150, 220, 180, 260, 190, 500, 120, 140, 300, 170],
            "category": ["A > B > C"] * 10,
        }
    )


def test_apply_all_mitigations_with_empty_reviews_does_not_crash():
    fin_df = _financial_df()
    prod_df = _products_df()
    rev_df = pd.DataFrame()

    fin_out, prod_out, rev_out = apply_all_mitigations(fin_df, prod_df, rev_df, random_state=42)

    assert len(fin_out) > 0
    assert len(prod_out) > 0
    assert isinstance(rev_out, pd.DataFrame)


def test_oversample_unverified_purchases_missing_column_does_not_crash():
    rev_df = pd.DataFrame({"rating": [1, 2, 3, 4, 5]})
    out = oversample_unverified_purchases(rev_df, rng=np.random.default_rng(42))
    assert len(out) == len(rev_df)


def test_oversample_minority_employment_increases_unemployed_share():
    fin_df = _financial_df()
    before_share = (fin_df["employment_status"] == "Unemployed").mean()

    out = oversample_minority_employment(fin_df, rng=np.random.default_rng(42))
    after_share = (out["employment_status"] == "Unemployed").mean()

    assert after_share >= before_share
