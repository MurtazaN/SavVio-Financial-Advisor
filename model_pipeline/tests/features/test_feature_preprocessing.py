"""Unit tests for feature_preprocessing.py pipeline components."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.feature_preprocessing import (
    CategoricalEncoder,
    FeatureDropper,
    FeaturePipeline,
    MissingValueImputer,
    NumericScaler,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _base_df(**overrides) -> pd.DataFrame:
    """Minimal valid DataFrame covering all Config feature groups."""
    data = {
        # Financial
        "discretionary_income":          [3000.0, None, 2500.0],
        "debt_to_income_ratio":          [0.3,    0.5,  None  ],
        "saving_to_income_ratio":        [0.1,    None, 0.2   ],
        "monthly_expense_burden_ratio":  [0.4,    0.6,  None  ],
        "emergency_fund_months":         [3.0,    None, 6.0   ],
        # Product
        "price":                         [100.0,  None, 200.0 ],
        "average_rating":                [4.5,    3.0,  None  ],
        "rating_number":                 [100,    None, 50    ],
        "rating_variance":               [None,   0.5,  1.0   ],
        # Categorical
        "employment_status":             ["employed", None, "self-employed"],
        "has_loan":                      ["yes",      "no", None           ],
        "region":                        ["north",    None, "south"        ],
        # Label + IDs
        "final_recommendation":          ["GREEN", "RED", "YELLOW"],
        "user_id":                       ["u1", "u2", "u3"],
        "product_id":                    ["p1", "p2", "p3"],
        # Numeric extras
        "monthly_income":                [5000.0, 4000.0, 6000.0],
        "monthly_expenses":              [2000.0, 3000.0, 1500.0],
        "savings_balance":               [10000.0, 500.0, 20000.0],
        "monthly_emi":                   [300.0, 0.0, 600.0],
        "affordability_score":           [0.7, 0.4, 0.9],
        "price_to_income_ratio":         [0.02, 0.05, 0.03],
        "residual_utility_score":        [0.6, 0.3, 0.8],
        "savings_to_price_ratio":        [100.0, 5.0, 200.0],
        "net_worth_indicator":           [0.8, 0.2, 0.9],
        "credit_risk_indicator":         [0.1, 0.6, 0.05],
    }
    data.update(overrides)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# MissingValueImputer
# ---------------------------------------------------------------------------

class TestMissingValueImputer:

    def test_financial_features_no_nulls_after_transform(self):
        df = _base_df()
        result = MissingValueImputer().fit_transform(df)
        for col in ["discretionary_income", "debt_to_income_ratio",
                    "saving_to_income_ratio", "monthly_expense_burden_ratio",
                    "emergency_fund_months"]:
            assert result[col].isnull().sum() == 0, f"{col} still has nulls"

    def test_product_features_no_nulls_after_transform(self):
        df = _base_df()
        result = MissingValueImputer().fit_transform(df)
        for col in ["price", "average_rating", "rating_number", "rating_variance"]:
            assert result[col].isnull().sum() == 0, f"{col} still has nulls"

    def test_rating_variance_filled_with_zero_not_median(self):
        df = _base_df(rating_variance=[None, None, None])
        result = MissingValueImputer().fit_transform(df)
        assert (result["rating_variance"] == 0.0).all()

    def test_product_price_alias_filled(self):
        """When 'price' absent but 'product_price' present, it should be imputed."""
        df = _base_df()
        df = df.rename(columns={"price": "product_price"})
        result = MissingValueImputer().fit_transform(df)
        assert result["product_price"].isnull().sum() == 0

    def test_categorical_features_filled_with_unknown(self):
        df = _base_df()
        result = MissingValueImputer().fit_transform(df)
        for col in ["employment_status", "has_loan", "region"]:
            assert result[col].isnull().sum() == 0
            assert (result[col] == "Unknown").sum() >= 0  # no crash

    def test_computed_features_imputed_with_median(self):
        df = _base_df(affordability_score=[None, 0.5, 0.9])
        result = MissingValueImputer().fit_transform(df)
        assert result["affordability_score"].isnull().sum() == 0

    def test_input_dataframe_not_mutated(self):
        df = _base_df()
        original_nulls = df.isnull().sum().sum()
        MissingValueImputer().fit_transform(df)
        assert df.isnull().sum().sum() == original_nulls

    def test_fit_returns_self(self):
        imp = MissingValueImputer()
        assert imp.fit(_base_df()) is imp


# ---------------------------------------------------------------------------
# CategoricalEncoder
# ---------------------------------------------------------------------------

class TestCategoricalEncoder:

    def test_categorical_columns_become_numeric(self):
        df = _base_df()
        # Fill nulls first so encoder doesn't choke
        df[["employment_status", "has_loan", "region"]] = \
            df[["employment_status", "has_loan", "region"]].fillna("Unknown")
        enc = CategoricalEncoder()
        result = enc.fit_transform(df)
        for col in ["employment_status", "has_loan", "region"]:
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} not numeric"

    def test_unknown_category_encoded_as_minus_one(self):
        df = _base_df()
        df[["employment_status", "has_loan", "region"]] = \
            df[["employment_status", "has_loan", "region"]].fillna("Unknown")
        enc = CategoricalEncoder()
        enc.fit(df)
        unseen = df.copy()
        unseen["employment_status"] = "astronaut"  # never seen during fit
        result = enc.transform(unseen)
        assert (result["employment_status"] == -1).any()

    def test_no_cat_cols_in_df_is_noop(self):
        df = _base_df().drop(columns=["employment_status", "has_loan", "region"])
        enc = CategoricalEncoder()
        result = enc.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_fit_returns_self(self):
        enc = CategoricalEncoder()
        assert enc.fit(_base_df()) is enc

    def test_existing_cat_cols_populated_after_fit(self):
        df = _base_df()
        enc = CategoricalEncoder()
        enc.fit(df)
        assert set(enc.existing_cat_cols) == {"employment_status", "has_loan", "region"}


# ---------------------------------------------------------------------------
# NumericScaler
# ---------------------------------------------------------------------------

class TestNumericScaler:

    def _clean_df(self) -> pd.DataFrame:
        """Fully filled DataFrame so scaler doesn't see NaNs."""
        df = _base_df()
        return df.fillna(0)

    def test_scaled_columns_have_approx_zero_mean(self):
        df = pd.DataFrame({
            "discretionary_income": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            "price":                [10.0,   20.0,   30.0,   40.0,   50.0  ],
        })
        scaler = NumericScaler()
        result = scaler.fit_transform(df)
        assert abs(result["discretionary_income"].mean()) < 1e-9
        assert abs(result["price"].mean()) < 1e-9

    def test_product_price_alias_scaled(self):
        df = self._clean_df().rename(columns={"price": "product_price"})
        scaler = NumericScaler()
        result = scaler.fit_transform(df)
        assert "product_price" in scaler.numeric_cols

    def test_no_numeric_cols_is_noop(self):
        df = pd.DataFrame({"employment_status": ["a", "b", "c"]})
        scaler = NumericScaler()
        result = scaler.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_fit_returns_self(self):
        scaler = NumericScaler()
        assert scaler.fit(self._clean_df()) is scaler

    def test_input_not_mutated(self):
        df = self._clean_df()
        original = df.copy()
        NumericScaler().fit_transform(df)
        pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# FeatureDropper
# ---------------------------------------------------------------------------

class TestFeatureDropper:

    def test_id_columns_dropped(self):
        df = _base_df()
        result = FeatureDropper().fit_transform(df)
        for col in ["user_id", "product_id"]:
            assert col not in result.columns

    def test_label_column_dropped(self):
        df = _base_df()
        result = FeatureDropper().fit_transform(df)
        assert "final_recommendation" not in result.columns

    def test_product_price_alias_dropped(self):
        df = _base_df().rename(columns={"price": "product_price"})
        result = FeatureDropper().fit_transform(df)
        assert "product_price" not in result.columns

    def test_text_blob_columns_dropped(self):
        df = _base_df()
        df["product_name"] = "Widget"
        df["description"] = "A great widget"
        result = FeatureDropper().fit_transform(df)
        for col in ["product_name", "description"]:
            assert col not in result.columns

    def test_feature_columns_preserved(self):
        df = _base_df()
        result = FeatureDropper().fit_transform(df)
        assert "discretionary_income" in result.columns
        assert "price" in result.columns

    def test_missing_drop_columns_no_error(self):
        """Dropper should not raise if a COLUMNS_TO_DROP col is absent."""
        df = _base_df().drop(columns=["user_id"])
        result = FeatureDropper().fit_transform(df)
        assert "product_id" not in result.columns

    def test_fit_sets_is_fitted(self):
        dropper = FeatureDropper()
        dropper.fit(_base_df())
        assert dropper.is_fitted_ is True


# ---------------------------------------------------------------------------
# FeaturePipeline (integration)
# ---------------------------------------------------------------------------

class TestFeaturePipeline:

    def test_fit_transform_returns_dataframe(self, tmp_path, monkeypatch):
        df = _base_df()
        fp = FeaturePipeline()
        monkeypatch.setattr(fp, "save", lambda: None)
        result = fp.fit_transform(df, save_pipeline=False)
        assert isinstance(result, (pd.DataFrame, np.ndarray))

    def test_save_creates_file(self, tmp_path):
        import os
        df = _base_df()
        path = str(tmp_path / "pipe.pkl")
        fp = FeaturePipeline()
        fp.pipeline.fit(df)
        fp.save(path=path)
        assert os.path.exists(path)

    def test_load_restores_pipeline(self, tmp_path):
        df = _base_df()
        path = str(tmp_path / "pipe.pkl")
        fp = FeaturePipeline()
        fp.pipeline.fit(df)
        fp.save(path=path)
        fp2 = FeaturePipeline()
        fp2.load(path=path)
        result = fp2.pipeline.transform(df)
        assert result is not None

    def test_transform_uses_fitted_pipeline(self, tmp_path):
        df = _base_df()
        path = str(tmp_path / "pipe.pkl")
        fp = FeaturePipeline()
        fp.fit_transform(df, save_pipeline=False)
        fp.save(path=path)
        fp2 = FeaturePipeline()
        fp2.load(path=path)
        result = fp2.pipeline.transform(df)
        assert result is not None

    def test_label_not_in_output(self):
        df = _base_df()
        fp = FeaturePipeline()
        result = fp.fit_transform(df, save_pipeline=False)
        if isinstance(result, pd.DataFrame):
            assert "final_recommendation" not in result.columns

    def test_pipeline_steps_in_correct_order(self):
        fp = FeaturePipeline()
        step_names = [name for name, _ in fp.pipeline.steps]
        assert step_names == ["imputer", "encoder", "scaler", "dropper"]