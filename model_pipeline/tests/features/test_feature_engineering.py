"""
Unit tests for Feature Engineering pipeline.

Tests missing-value handling, encoding, scaling, and build_training_data().
Uses synthetic data so tests run without a live PostgreSQL connection.
"""

import pytest
import numpy as np
import pandas as pd


# ── Synthetic data helpers ───────────────────────────────────────────────────

def _make_financial_df(n=50):
    rng = np.random.default_rng(42)
    savings_balance = rng.uniform(500, 50000, n)
    return pd.DataFrame({
        "user_id": [f"U{i}" for i in range(n)],
        "monthly_income": rng.uniform(2000, 10000, n),
        "monthly_expenses": rng.uniform(1000, 5000, n),
        "savings_balance": savings_balance,
        "liquid_savings": savings_balance * rng.uniform(0.10, 0.60, n),
        "has_loan": rng.choice([True, False], n),
        "loan_amount": rng.uniform(0, 200000, n),
        "monthly_emi": rng.uniform(0, 2000, n),
        "loan_interest_rate": rng.uniform(0, 20, n),
        "loan_term_months": rng.choice([0, 60, 120, 240], n),
        "credit_score": rng.integers(300, 850, n),
        "employment_status": rng.choice(["employed", "self-employed", "unemployed"], n),
        "region": rng.choice(["east", "west", "south", "north"], n),
        "discretionary_income": rng.uniform(-2000, 5000, n),
        "debt_to_income_ratio": rng.uniform(0, 0.6, n),
        "saving_to_income_ratio": rng.uniform(0.1, 5.0, n),
        "monthly_expense_burden_ratio": rng.uniform(0.3, 1.2, n),
        "emergency_fund_months": rng.uniform(0.5, 12, n),
    })


def _make_reviews_df(n=100):
    rng = np.random.default_rng(77)
    return pd.DataFrame({
        "review_id": [f"R{i}" for i in range(n)],
        "product_id": [f"P{rng.integers(0, 30)}" for i in range(n)],
        "user_id": [f"U{rng.integers(0, 50)}" for i in range(n)],
        "rating": rng.integers(1, 6, n),
        "review_text": ["This is a test review"] * n,
        "verified_purchase": rng.choice([True, False], n),
        "helpful_vote": rng.integers(0, 100, n),
    })

def _make_products_df(n=32):
    """Create products spanning all 3 price tiers."""
    rng = np.random.default_rng(99)
    per_tier = max(n // 3, 2)
    remainder = n - 2 * per_tier
    prices = np.concatenate([
        rng.uniform(110, 490, per_tier),       # budget: $100–$500
        rng.uniform(550, 1400, per_tier),      # mid: $500–$1500
        rng.uniform(1600, 5000, remainder),    # premium: $1500+
    ])
    rng.shuffle(prices)
    total = len(prices)
    return pd.DataFrame({
        "product_id": [f"P{i}" for i in range(total)],
        "product_name": [f"Product {i}" for i in range(total)],
        "price": prices,
        "average_rating": rng.uniform(1, 5, total),
        "rating_number": rng.integers(1, 5000, total),
        "rating_variance": rng.uniform(0, 2, total),
        "description": ["desc"] * total,
        "features": ["feat"] * total,
        "details": ["det"] * total,
        "category": ["cat"] * total,
    })


# ── Missing-value tests ─────────────────────────────────────────────────────

class TestMissingValues:

    def test_financial_nulls_filled(self):
        from features.feature_preprocessing import MissingValueImputer
        df = _make_financial_df(20)
        df.loc[0, "discretionary_income"] = np.nan
        df.loc[1, "debt_to_income_ratio"] = np.nan
        result = MissingValueImputer().transform(df)
        assert result["discretionary_income"].isnull().sum() == 0
        assert result["debt_to_income_ratio"].isnull().sum() == 0

    def test_product_variance_null_filled_with_zero(self):
        from features.feature_preprocessing import MissingValueImputer
        df = pd.DataFrame({
            "rating_variance": [1.0, np.nan, 0.5],
            "price": [10, 20, 30],
        })
        result = MissingValueImputer().transform(df)
        assert result["rating_variance"].iloc[1] == 0.0

    def test_categorical_nulls_filled_with_unknown(self):
        from features.feature_preprocessing import MissingValueImputer
        df = pd.DataFrame({
            "employment_status": ["employed", None, "self-employed"],
            "has_loan": [True, None, False],
            "region": ["east", "west", None],
        })
        result = MissingValueImputer().transform(df)
        assert result["employment_status"].iloc[1] == "Unknown"
        assert result["region"].iloc[2] == "Unknown"

    def test_computed_features_nulls_filled_with_median(self):
        """All 6 computed financial features should have NaN filled with median."""
        from features.feature_preprocessing import MissingValueImputer
        computed_cols = [
            "affordability_score", "price_to_income_ratio", "residual_utility_score",
            "savings_to_price_ratio", "net_worth_indicator", "credit_risk_indicator",
        ]
        df = pd.DataFrame({col: [np.nan, 1.0, 3.0] for col in computed_cols})
        result = MissingValueImputer().transform(df)
        for col in computed_cols:
            assert result[col].isnull().sum() == 0, f"NaN not filled in {col}"
            assert result[col].iloc[0] == pytest.approx(2.0), f"Expected median 2.0 for {col}"


# ── Encoding tests ───────────────────────────────────────────────────────────

class TestEncoding:

    def test_ordinal_encoder_produces_numeric(self, tmp_path):
        from features.feature_preprocessing import CategoricalEncoder
        
        df = pd.DataFrame({
            "employment_status": ["employed", "self-employed", "unemployed"],
            "has_loan": [True, False, True],
            "region": ["east", "west", "south"],
        })
        encoder = CategoricalEncoder()
        result = encoder.fit_transform(df)
        assert result["employment_status"].dtype in (np.float64, np.int64)
        assert result["region"].dtype in (np.float64, np.int64)

        # Verify inference with same encoder produces same values.
        result2 = encoder.transform(df)
        assert (result["employment_status"] == result2["employment_status"]).all()

    def test_unknown_category_at_inference(self):
        from features.feature_preprocessing import CategoricalEncoder
        train_df = pd.DataFrame({
            "employment_status": ["employed", "self-employed"],
            "has_loan": [True, False],
            "region": ["east", "west"],
        })
        encoder = CategoricalEncoder()
        encoder.fit(train_df)

        test_df = pd.DataFrame({
            "employment_status": ["NEVER_SEEN_BEFORE"],
            "has_loan": [True],
            "region": ["UNKNOWN_REGION"],
        })
        result = encoder.transform(test_df)
        assert result["employment_status"].iloc[0] == -1
        assert result["region"].iloc[0] == -1


# ── Scaling tests ────────────────────────────────────────────────────────────

class TestScaling:

    def test_scaled_features_near_zero_mean(self, tmp_path):
        from features.feature_preprocessing import NumericScaler
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "discretionary_income": rng.uniform(-1000, 5000, 100),
            "price": rng.uniform(10, 1000, 100),
            "savings_to_price_ratio": rng.uniform(0.1, 50, 100),
            "credit_risk_indicator": rng.uniform(0, 1, 100),
        })
        scaler = NumericScaler()
        result = scaler.fit_transform(df)

        for col in ["discretionary_income", "price", "savings_to_price_ratio"]:
            if col in result.columns:
                assert abs(result[col].mean()) < 0.1


# ── build_training_data integration ─────────────────────────────────────────

class TestBuildFeatureMatrix:

    def test_returns_correct_shapes(self, tmp_path):
        from features.feature_engineering import build_training_data
        import config
        original = config.Config.MODEL_SAVE_DIR
        config.Config.MODEL_SAVE_DIR = str(tmp_path)
        original_scenario = config.Config.SCENARIO_OUTPUT_PATH
        config.Config.SCENARIO_OUTPUT_PATH = str(tmp_path / "scenarios.csv")

        fin = _make_financial_df(50)
        prod = _make_products_df(30)

        X, y, raw = build_training_data(
            financial_df=fin, products_df=prod, reviews_df=_make_reviews_df(100), n_scenarios=200
        )

        assert len(X) <= 200
        assert len(X) >= 150, f"Expected ~200 rows, got {len(X)}"
        assert len(y) == len(X)
        assert len(raw) == len(X)
        assert X.isnull().any().any() == False
        assert set(y.unique()).issubset({"GREEN", "YELLOW", "RED"})

        config.Config.MODEL_SAVE_DIR = original
        config.Config.SCENARIO_OUTPUT_PATH = original_scenario

    def test_label_column_not_in_X(self, tmp_path):
        from features.feature_engineering import build_training_data
        import config
        original = config.Config.MODEL_SAVE_DIR
        config.Config.MODEL_SAVE_DIR = str(tmp_path)
        original_scenario = config.Config.SCENARIO_OUTPUT_PATH
        config.Config.SCENARIO_OUTPUT_PATH = str(tmp_path / "scenarios.csv")

        fin = _make_financial_df(50)
        prod = _make_products_df(30)

        X, y, _ = build_training_data(
            financial_df=fin, products_df=prod, reviews_df=_make_reviews_df(100), n_scenarios=100
        )
        assert "final_recommendation" not in X.columns

        config.Config.MODEL_SAVE_DIR = original
        config.Config.SCENARIO_OUTPUT_PATH = original_scenario

    def test_no_id_columns_in_X(self, tmp_path):
        from features.feature_engineering import build_training_data
        import config
        original = config.Config.MODEL_SAVE_DIR
        config.Config.MODEL_SAVE_DIR = str(tmp_path)
        original_scenario = config.Config.SCENARIO_OUTPUT_PATH
        config.Config.SCENARIO_OUTPUT_PATH = str(tmp_path / "scenarios.csv")

        fin = _make_financial_df(50)
        prod = _make_products_df(30)

        X, _, _ = build_training_data(
            financial_df=fin, products_df=prod, reviews_df=_make_reviews_df(100), n_scenarios=100
        )
        assert "user_id" not in X.columns
        assert "product_id" not in X.columns

        config.Config.MODEL_SAVE_DIR = original
        config.Config.SCENARIO_OUTPUT_PATH = original_scenario

    def test_computed_features_present_in_X(self, tmp_path):
        """Verify all 6 computed financial features survive the pipeline into X."""
        from features.feature_engineering import build_training_data
        import config
        original = config.Config.MODEL_SAVE_DIR
        config.Config.MODEL_SAVE_DIR = str(tmp_path)
        original_scenario = config.Config.SCENARIO_OUTPUT_PATH
        config.Config.SCENARIO_OUTPUT_PATH = str(tmp_path / "scenarios.csv")

        fin = _make_financial_df(50)
        prod = _make_products_df(30)

        X, _, _ = build_training_data(
            financial_df=fin, products_df=prod, reviews_df=_make_reviews_df(100), n_scenarios=100
        )

        feature_cols = [
            "affordability_score", "price_to_income_ratio",
            "residual_utility_score", "savings_to_price_ratio",
            "net_worth_indicator", "credit_risk_indicator",
        ]
        for col in feature_cols:
            assert col in X.columns, f"Missing computed feature in X: {col}"

        # Verify review features are NOT in X (engine is purely financial).
        assert "review_confidence_score" not in X.columns
        assert "review_polarization_index" not in X.columns

        config.Config.MODEL_SAVE_DIR = original
        config.Config.SCENARIO_OUTPUT_PATH = original_scenario

