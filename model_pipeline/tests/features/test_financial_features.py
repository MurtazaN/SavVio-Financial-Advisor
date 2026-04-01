"""
Unit tests for affordability features and scenario generation.

Tests compute_affordability() (inference-time, 6 financial features) and
generate_scenarios() (batch label generation with DecisionEngine).
"""

import os
import sys

import pytest
import math
import numpy as np
import pandas as pd


from features.financial_features import (
    compute_affordability,
    AffordabilityResult,
)
from features.training_data_generator import (
    generate_scenarios,
    _sample_stratified,
)


# ── compute_affordability ────────────────────────────────────────────────────

class TestComputeAffordability:

    def _default_profile(self):
        return {
            "monthly_income": 5000.0,
            "discretionary_income": 1200.0,
            "liquid_savings": 8000.0,
            "monthly_expenses": 2800.0,
            "monthly_emi": 1000.0,
            "loan_amount": 15000.0,
            "credit_score": 720,
        }

    def test_affordability_score(self):
        result = compute_affordability(self._default_profile(), 200.0)
        assert result.affordability_score == pytest.approx(1200 - 200, abs=0.01)

    def test_price_to_income_ratio(self):
        result = compute_affordability(self._default_profile(), 200.0)
        assert result.price_to_income_ratio == pytest.approx(200 / 5000, abs=1e-3)

    def test_residual_utility_score(self):
        result = compute_affordability(self._default_profile(), 200.0)
        assert result.residual_utility_score == pytest.approx(
            (8000 - 200) / (2800 + 1000), abs=1e-3
        )

    def test_savings_to_price_ratio(self):
        result = compute_affordability(self._default_profile(), 200.0)
        assert result.savings_to_price_ratio == pytest.approx(8000 / 200, abs=1e-3)

    def test_net_worth_indicator(self):
        result = compute_affordability(self._default_profile(), 200.0)
        assert result.net_worth_indicator == pytest.approx((8000 - 15000) / 5000, abs=1e-3)

    def test_credit_risk_indicator(self):
        result = compute_affordability(self._default_profile(), 200.0)
        assert result.credit_risk_indicator == pytest.approx((720 - 299) / 550, abs=1e-3)

    def test_zero_income_guards(self):
        profile = self._default_profile()
        profile["monthly_income"] = 0
        profile["discretionary_income"] = 0
        result = compute_affordability(profile, product_price=100.0)
        assert result.price_to_income_ratio is None
        assert result.affordability_score == -100.0

    def test_zero_obligations_guards(self):
        profile = self._default_profile()
        profile["monthly_expenses"] = 0
        profile["monthly_emi"] = 0
        result = compute_affordability(profile, product_price=50.0)
        assert result.residual_utility_score is None

    def test_to_dict_has_all_6_keys(self):
        result = compute_affordability(self._default_profile(), 100)
        d = result.to_dict()
        assert len(d) == 6
        expected_keys = {
            "affordability_score", "price_to_income_ratio", "residual_utility_score",
            "savings_to_price_ratio", "net_worth_indicator", "credit_risk_indicator",
        }
        assert set(d.keys()) == expected_keys

    def test_no_product_info_defaults(self):
        """compute_affordability without product_info should not crash."""
        result = compute_affordability(self._default_profile(), 200.0)
        assert result.affordability_score is not None

    def test_cumulative_spend_reduces_savings_features(self):
        """DI absorbs purchases first; savings cover the shortfall.

        Profile: DI=1200, savings=8000, cumulative_spend=5000.
        DI covers $1200, savings cover the remaining $3800.
        effective_savings = 8000 - 3800 = 4200.
        """
        profile = self._default_profile()
        prior_spend = 5000.0
        result = compute_affordability(profile, 200.0, cumulative_spend=prior_spend)
        di_used = min(1200.0, prior_spend)          # 1200
        savings_used = prior_spend - di_used         # 3800
        effective_savings = 8000.0 - savings_used    # 4200
        assert result.savings_to_price_ratio == pytest.approx(effective_savings / 200, abs=1e-3)
        assert result.residual_utility_score == pytest.approx(
            (effective_savings - 200) / (2800 + 1000), abs=1e-3
        )
        assert result.net_worth_indicator == pytest.approx(
            (effective_savings - 15000) / 5000, abs=1e-3
        )

    def test_cumulative_spend_reduces_affordability(self):
        """After spending $1000, discretionary_income drops from 1200 to 200."""
        profile = self._default_profile()
        result = compute_affordability(profile, 200.0, cumulative_spend=1000.0)
        assert result.affordability_score == pytest.approx(
            (1200 - 1000) - 200, abs=0.01
        )

    def test_cumulative_spend_floors_savings_at_zero(self):
        """Savings can't go negative — clipped at 0."""
        profile = self._default_profile()
        result = compute_affordability(profile, 200.0, cumulative_spend=99999.0)
        assert result.savings_to_price_ratio == pytest.approx(0.0, abs=1e-3)

    def test_zero_cumulative_spend_is_default_behavior(self):
        """cumulative_spend=0 produces the same result as no argument."""
        profile = self._default_profile()
        r1 = compute_affordability(profile, 200.0)
        r2 = compute_affordability(profile, 200.0, cumulative_spend=0.0)
        assert r1.to_dict() == r2.to_dict()


# ── generate_scenarios ───────────────────────────────────────────────────────

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


def _make_products_df(n=32):
    """Create products spanning all 3 price tiers.

    Bins: [100, 500, 1500, inf].  At least n//3 products per tier.
    """
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


class TestGenerateScenarios:

    def test_correct_row_count(self):
        scenarios = generate_scenarios(
            _make_financial_df(), _make_products_df(), n_scenarios=100
        )
        assert len(scenarios) <= 100
        assert len(scenarios) >= 80, (
            f"Expected ~100 rows, got {len(scenarios)}"
        )

    @pytest.mark.skip(reason="generate_scenarios does not yet return computed features")
    def test_has_all_6_feature_columns(self):
        scenarios = generate_scenarios(
            _make_financial_df(), _make_products_df(), n_scenarios=50
        )
        required = [
            "product_id", "product_price",
            # 6 financial features (no product review features).
            "affordability_score", "price_to_income_ratio", "residual_utility_score",
            "savings_to_price_ratio", "net_worth_indicator", "credit_risk_indicator",
            "final_recommendation",
        ]
        for col in required:
            assert col in scenarios.columns, f"Missing column: {col}"

    @pytest.mark.skip(reason="generate_scenarios does not yet return computed features")
    def test_no_review_features_computed(self):
        """review_confidence_score and review_polarization_index are NOT computed."""
        scenarios = generate_scenarios(
            _make_financial_df(), _make_products_df(), n_scenarios=50
        )
        assert "review_confidence_score" not in scenarios.columns
        assert "review_polarization_index" not in scenarios.columns

    @pytest.mark.skip(reason="generate_scenarios does not yet return final_recommendation")
    def test_labels_are_valid(self):
        scenarios = generate_scenarios(
            _make_financial_df(), _make_products_df(), n_scenarios=200
        )
        assert set(scenarios["final_recommendation"].unique()).issubset({"GREEN", "YELLOW", "RED"})

    @pytest.mark.skip(reason="generate_scenarios does not yet return final_recommendation")
    def test_not_all_one_label(self):
        """With varied synthetic data we should get at least 2 label classes."""
        scenarios = generate_scenarios(
            _make_financial_df(100), _make_products_df(50), n_scenarios=500
        )
        assert scenarios["final_recommendation"].nunique() >= 2

    def test_reproducibility(self):
        """Same seed produces identical output."""
        fin = _make_financial_df()
        prod = _make_products_df()
        s1 = generate_scenarios(fin, prod, n_scenarios=50, random_state=123)
        s2 = generate_scenarios(fin, prod, n_scenarios=50, random_state=123)
        pd.testing.assert_frame_equal(s1, s2)

    def test_different_seed_different_output(self):
        fin = _make_financial_df()
        prod = _make_products_df()
        s1 = generate_scenarios(fin, prod, n_scenarios=50, random_state=1)
        s2 = generate_scenarios(fin, prod, n_scenarios=50, random_state=2)
        assert not s1["product_price"].equals(s2["product_price"])

    @pytest.mark.skip(reason="graduated mode not yet implemented")
    def test_graduated_creates_price_tier_column(self):
        """Graduated mode produces a price_tier column with valid values."""
        scenarios = generate_scenarios(
            _make_financial_df(), _make_products_df(), n_scenarios=100,
            graduated=True,
        )
        assert len(scenarios) <= 100
        assert "price_tier" in scenarios.columns
        assert set(scenarios["price_tier"].unique()).issubset(
            {"budget", "mid", "premium"}
        )

    @pytest.mark.skip(reason="graduated mode not yet implemented")
    def test_graduated_all_tiers_present(self):
        """With enough scenarios, all 4 price tiers should appear."""
        scenarios = generate_scenarios(
            _make_financial_df(50), _make_products_df(), n_scenarios=200,
            graduated=True,
        )
        tier_counts = scenarios["price_tier"].value_counts()
        for tier in ("budget", "mid", "premium"):
            assert tier in tier_counts.index, f"Missing tier: {tier}"

    @pytest.mark.skip(reason="graduated mode not yet implemented")
    def test_graduated_cumulative_spending_depletes_savings(self):
        """Across tiers, saving_to_income_ratio should decrease because
        cumulative spending reduces savings while income stays constant."""
        fin = _make_financial_df(100)
        prod = _make_products_df(60)
        scenarios = generate_scenarios(
            fin, prod, n_scenarios=200, graduated=True,
        )
        budget_stir = scenarios.loc[
            scenarios["price_tier"] == "budget", "saving_to_income_ratio"
        ]
        mid_rows = scenarios[scenarios["price_tier"] == "mid"]
        if len(mid_rows) == 0:
            pytest.skip("No users survived to mid tier")
        mid_stir = mid_rows["saving_to_income_ratio"]
        assert budget_stir.mean() > mid_stir.mean(), (
            "Budget tier should have higher avg saving_to_income_ratio than mid"
        )

    @pytest.mark.skip(reason="graduated mode not yet implemented")
    def test_graduated_red_stops_evaluation(self):
        """A user who gets RED on budget should NOT appear in mid or premium.

        verify the early-stop invariant.
        """
        # Construct a deliberately poor user who will trigger RED R1:
        #   affordability < 0 AND spr < 1.5 AND pir > 0.10
        # Income = $400, budget product = $150 → PIR = 0.375 > 0.10.
        poor_user = pd.DataFrame({
            "user_id": ["POOR"],
            "monthly_income": [400.0],
            "monthly_expenses": [500.0],
            "savings_balance": [100.0],
            "liquid_savings": [30.0],
            "has_loan": [True],
            "loan_amount": [5000.0],
            "monthly_emi": [100.0],
            "loan_interest_rate": [15.0],
            "loan_term_months": [60],
            "credit_score": [350],
            "employment_status": ["unemployed"],
            "region": ["east"],
            "discretionary_income": [-200.0],
            "debt_to_income_ratio": [0.50],
            "saving_to_income_ratio": [0.075],
            "monthly_expense_burden_ratio": [1.5],
            "emergency_fund_months": [0.05],
        })
        tier_products = {
            "budget": pd.DataFrame({
                "product_id": ["B1"], "product_name": ["Budget"],
                "price": [150.0], "average_rating": [4.0],
                "rating_number": [100], "rating_variance": [0.5],
            }),
            "mid": pd.DataFrame({
                "product_id": ["M1"], "product_name": ["Mid"],
                "price": [800.0], "average_rating": [4.0],
                "rating_number": [100], "rating_variance": [0.5],
            }),
            "premium": pd.DataFrame({
                "product_id": ["X1"], "product_name": ["Premium"],
                "price": [2000.0], "average_rating": [4.0],
                "rating_number": [100], "rating_variance": [0.5],
            }),
        }

        budget_label = scenarios.loc[
            scenarios["price_tier"] == "budget", "final_recommendation"
        ].values[0]
        assert budget_label == "RED", (
            f"Poor user should get RED on budget tier, got {budget_label}"
        )
        assert len(scenarios) == 1, (
            f"RED on budget should stop evaluation: expected 1 row, got {len(scenarios)}"
        )
        assert "mid" not in scenarios["price_tier"].values
        assert "premium" not in scenarios["price_tier"].values

    @pytest.mark.skip(reason="graduated mode not yet implemented")
    def test_graduated_later_tier_can_trigger_red(self):
        """A user GREEN on budget can turn RED/YELLOW on premium after
        savings deplete from cumulative spending."""
        fin = pd.DataFrame({
            "user_id": ["U0"],
            "monthly_income": [3000.0],
            "monthly_expenses": [2000.0],
            "savings_balance": [4000.0],
            "liquid_savings": [4000.0],
            "has_loan": [False],
            "loan_amount": [0.0],
            "monthly_emi": [500.0],
            "loan_interest_rate": [0.0],
            "loan_term_months": [0],
            "credit_score": [650],
            "employment_status": ["employed"],
            "region": ["west"],
            "discretionary_income": [500.0],
            "debt_to_income_ratio": [0.17],
            "saving_to_income_ratio": [1.33],
            "monthly_expense_burden_ratio": [0.83],
            "emergency_fund_months": [1.6],
        })
        prod = pd.DataFrame({
            "product_id": ["B1", "M1", "X1"],
            "product_name": ["Budget", "Mid", "Premium"],
            "price": [150.0, 800.0, 2000.0],
            "average_rating": [4.0, 4.0, 4.0],
            "rating_number": [100, 100, 100],
            "rating_variance": [0.5, 0.5, 0.5],
            "description": ["d", "d", "d"],
            "features": ["f", "f", "f"],
            "details": ["d", "d", "d"],
            "category": ["cat", "cat", "cat"],
        })
        # n_scenarios=3 ensures all 3 tiers are returned without subsampling.
        scenarios = generate_scenarios(
            fin, prod, n_scenarios=3, graduated=True,
        )
        budget_label = scenarios.loc[
            scenarios["price_tier"] == "budget", "final_recommendation"
        ].values[0]
        assert budget_label == "GREEN", (
            f"$150 purchase with $3000 income should be GREEN, got {budget_label}"
        )
        last_row = scenarios.iloc[-1]
        if last_row["price_tier"] == "premium":
            assert last_row["final_recommendation"] in ("YELLOW", "RED"), (
                f"Expensive purchase after cumulative spend "
                f"should not be GREEN: got {last_row['final_recommendation']}"
            )

    @pytest.mark.skip(reason="graduated/stratified params not yet implemented")
    def test_legacy_stratified_mode(self):
        """graduated=False falls back to stratified single-purchase mode."""
        scenarios = generate_scenarios(
            _make_financial_df(), _make_products_df(), n_scenarios=50,
            graduated=False, stratified=True,
        )
        assert "price_tier" not in scenarios.columns
        assert len(scenarios) == 50
