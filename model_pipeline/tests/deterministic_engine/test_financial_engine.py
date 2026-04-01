"""
Unit tests for the Deterministic Decision Engine.

Tests the cross-group compound AND labeling rules:
  RED    — 4 rules crossing 2+ groups, each with PIR escape hatch
  YELLOW — 5 rules crossing 2+ groups, triggers when >= 2 fire
  GREEN  — default when no RED triggers and < 2 YELLOW rules fire
"""

import os
import sys

import pytest
import pandas as pd


from deterministic_engine.financial_engine import DecisionEngine, DecisionResult


@pytest.fixture
def engine():
    return DecisionEngine()


def _healthy_financial():
    """Financial profile where everything is solid — should be GREEN.
    All values set to clearly pass every RED and YELLOW threshold."""
    return {
        # Per-pair computed features.
        "affordability_score": 2000,       # > 0
        "price_to_income_ratio": 0.02,     # < 0.10
        "residual_utility_score": 20.0,    # > 3.0
        "savings_to_price_ratio": 100.0,   # > 10.0
        "net_worth_indicator": 5.0,        # > 1.0
        "credit_risk_indicator": 0.75,     # > 0.35
        # Per-entity DB features.
        "debt_to_income_ratio": 0.10,      # < 0.30
        "saving_to_income_ratio": 2.0,     # > 0.25
        "monthly_expense_burden_ratio": 0.40,  # < 0.70
        "emergency_fund_months": 8.0,      # > 4.0
    }


def _healthy_product():
    """Product — only price is used by the engine now."""
    return {
        "price": 50.0,
    }


# ── GREEN ───────────────────────────────────────────────────────────────────

class TestGreen:

    def test_healthy_user_good_product(self, engine):
        result = engine.decide(_healthy_financial(), _healthy_product())
        assert result.decision_category == "GREEN"

    def test_single_yellow_rule_stays_green(self, engine):
        """Only 1 YELLOW rule triggers → need 2+ to trigger YELLOW."""
        fin = _healthy_financial()
        fin["affordability_score"] = -100
        fin["price_to_income_ratio"] = 0.30
        fin["savings_to_price_ratio"] = 8.0  # below 10 for Rule 1
        result = engine.decide(fin, _healthy_product())
        assert result.decision_category == "GREEN"

    def test_trivial_purchase_for_stressed_user_stays_green(self, engine):
        """PIR < 0.10 escape: even stressed users get GREEN for trivial buys."""
        fin = _healthy_financial()
        fin["affordability_score"] = -100
        fin["savings_to_price_ratio"] = 1.0
        fin["residual_utility_score"] = 0.5
        fin["price_to_income_ratio"] = 0.05  # trivial purchase
        result = engine.decide(fin, _healthy_product())
        assert result.decision_category != "RED"


# ── RED RULES ──────────────────────────────────────────────────────────────

class TestRedRules:

    def test_rule1_groups_1_and_2(self, engine):
        """Income can't handle AND savings can't absorb AND non-trivial."""
        fin = _healthy_financial()
        fin["affordability_score"] = -500
        fin["savings_to_price_ratio"] = 1.0
        fin["residual_utility_score"] = 0.5
        fin["price_to_income_ratio"] = 0.15
        result = engine.decide(fin, _healthy_product())
        assert result.decision_category == "RED"
        assert any("cant_afford" in r for r in result.triggered_rules)

    def test_rule2_groups_3_1_2(self, engine):
        """Budget maxed AND significant purchase AND thin EFM and savings."""
        fin = _healthy_financial()
        fin["monthly_expense_burden_ratio"] = 0.85
        fin["price_to_income_ratio"] = 0.25
        fin["emergency_fund_months"] = 2.0
        fin["savings_to_price_ratio"] = 2.0
        result = engine.decide(fin, _healthy_product())
        assert result.decision_category == "RED"
        assert any("maxed_budget" in r for r in result.triggered_rules)

    def test_rule3_groups_4_1_2(self, engine):
        """Deeply underwater AND no surplus AND significant AND thin EFM."""
        fin = _healthy_financial()
        fin["net_worth_indicator"] = -3.0
        fin["affordability_score"] = -200
        fin["price_to_income_ratio"] = 0.20
        fin["emergency_fund_months"] = 2.0
        result = engine.decide(fin, _healthy_product())
        assert result.decision_category == "RED"

    def test_rule4_groups_2_3(self, engine):
        """No emergency fund AND purchase wipes runway AND high DTI AND non-trivial."""
        fin = _healthy_financial()
        fin["emergency_fund_months"] = 0.5
        fin["residual_utility_score"] = 0.3
        fin["debt_to_income_ratio"] = 0.35
        fin["price_to_income_ratio"] = 0.15
        result = engine.decide(fin, _healthy_product())
        assert result.decision_category == "RED"


# ── PIR ESCAPE HATCH (trivial purchases never RED) ────────────────────────

class TestPIREscape:

    def test_rule1_trivial_purchase_not_red(self, engine):
        """affordability < 0 AND spr < 1.5 AND rus < 1.0, but PIR tiny → not RED."""
        fin = _healthy_financial()
        fin["affordability_score"] = -100
        fin["savings_to_price_ratio"] = 1.0
        fin["residual_utility_score"] = 0.5
        fin["price_to_income_ratio"] = 0.05  # below 0.10 escape
        result = engine.decide(fin, _healthy_product())
        assert result.decision_category != "RED"

    def test_rule2_cheap_product_for_stressed_user(self, engine):
        """MEB > 0.80 AND efm < 3.0, but PIR tiny → not RED."""
        fin = _healthy_financial()
        fin["monthly_expense_burden_ratio"] = 0.90
        fin["emergency_fund_months"] = 1.0
        fin["savings_to_price_ratio"] = 2.0
        fin["price_to_income_ratio"] = 0.01  # below 0.20 escape
        result = engine.decide(fin, _healthy_product())
        assert result.decision_category != "RED"

    def test_rule3_underwater_but_cheap(self, engine):
        """NWI < -2 AND affordability < 0, but PIR tiny → not RED."""
        fin = _healthy_financial()
        fin["net_worth_indicator"] = -3.0
        fin["affordability_score"] = -100
        fin["emergency_fund_months"] = 2.0
        fin["price_to_income_ratio"] = 0.05  # below 0.15 escape
        result = engine.decide(fin, _healthy_product())
        assert result.decision_category != "RED"

    def test_rule4_paycheck_to_paycheck_cheap(self, engine):
        """EFM < 1 AND rus < 0.5 AND DTI > 0.30, but PIR tiny → not RED."""
        fin = _healthy_financial()
        fin["emergency_fund_months"] = 0.5
        fin["residual_utility_score"] = 0.3
        fin["debt_to_income_ratio"] = 0.35
        fin["price_to_income_ratio"] = 0.05  # below 0.10 escape
        result = engine.decide(fin, _healthy_product())
        assert result.decision_category != "RED"


# ── YELLOW RULES (cross-group, needs 2+ to trigger) ─────────────────────

class TestYellowRules:

    def test_rules_1_and_2_trigger_yellow(self, engine):
        """income_pressure + savings_strain → 2 rules → YELLOW."""
        fin = _healthy_financial()
        fin["affordability_score"] = -100
        fin["price_to_income_ratio"] = 0.30
        fin["savings_to_price_ratio"] = 3.0   # < 10 for Rule 1, < 5 for Rule 2
        fin["residual_utility_score"] = 2.0   # < 3 for Rule 2
        result = engine.decide(fin, _healthy_product())
        assert result.decision_category == "YELLOW"
        assert any("income_pressure" in r for r in result.triggered_rules)
        assert any("savings_strain" in r for r in result.triggered_rules)

    def test_rules_3_and_4_trigger_yellow(self, engine):
        """debt_stress + low_resilience → 2 rules → YELLOW."""
        fin = _healthy_financial()
        fin["monthly_expense_burden_ratio"] = 0.75
        fin["savings_to_price_ratio"] = 4.0
        fin["price_to_income_ratio"] = 0.15
        fin["emergency_fund_months"] = 2.0
        fin["affordability_score"] = -50
        result = engine.decide(fin, _healthy_product())
        assert result.decision_category == "YELLOW"
        assert any("debt_stress" in r for r in result.triggered_rules)
        assert any("low_resilience" in r for r in result.triggered_rules)

    def test_rules_1_and_5_trigger_yellow(self, engine):
        """income_pressure + weak_profile → 2 rules → YELLOW."""
        fin = _healthy_financial()
        fin["affordability_score"] = -100
        fin["price_to_income_ratio"] = 0.30
        fin["savings_to_price_ratio"] = 4.0
        fin["credit_risk_indicator"] = 0.30
        fin["net_worth_indicator"] = 0.5
        result = engine.decide(fin, _healthy_product())
        assert result.decision_category == "YELLOW"

    def test_savings_escape_prevents_rule1(self, engine):
        """income_pressure doesn't trigger if savings heavily cover the price."""
        fin = _healthy_financial()
        fin["affordability_score"] = -100
        fin["price_to_income_ratio"] = 0.30
        fin["savings_to_price_ratio"] = 15.0  # > 10 → Rule 1 doesn't fire
        result = engine.decide(fin, _healthy_product())
        # Only 1 rule triggers at most → GREEN
        assert result.decision_category == "GREEN"

    def test_trivial_purchase_prevents_rule2(self, engine):
        """savings_strain doesn't trigger if PIR is tiny (cheap product)."""
        fin = _healthy_financial()
        fin["savings_to_price_ratio"] = 3.0
        fin["residual_utility_score"] = 2.0
        fin["price_to_income_ratio"] = 0.05  # < 0.10 → Rule 2 doesn't fire
        result = engine.decide(fin, _healthy_product())
        assert result.decision_category == "GREEN"

    def test_one_rule_not_enough_for_yellow(self, engine):
        """Only 1 YELLOW rule triggers → stays GREEN."""
        fin = _healthy_financial()
        fin["affordability_score"] = -100
        fin["price_to_income_ratio"] = 0.30
        fin["savings_to_price_ratio"] = 8.0  # triggers Rule 1 only
        result = engine.decide(fin, _healthy_product())
        assert result.decision_category == "GREEN"

    def test_three_rules_still_yellow_not_red(self, engine):
        """3 YELLOW rules → YELLOW (not RED — RED has its own rules)."""
        fin = _healthy_financial()
        fin["affordability_score"] = -100
        fin["price_to_income_ratio"] = 0.30
        fin["savings_to_price_ratio"] = 4.0
        fin["residual_utility_score"] = 2.0
        fin["debt_to_income_ratio"] = 0.35
        fin["emergency_fund_months"] = 3.5   # < 4 for Rule 3
        result = engine.decide(fin, _healthy_product())
        assert result.decision_category == "YELLOW"


# ── CROSS-GROUP VALIDATION ──────────────────────────────────────────────

class TestCrossGroupValidation:

    def test_rule3_crosses_group3_and_group2(self, engine):
        """debt_stress requires DTI (G3) AND EFM (G2) AND PIR (G1)."""
        fin = _healthy_financial()
        # Only DTI is bad (same group as MEB) — shouldn't trigger alone
        fin["debt_to_income_ratio"] = 0.35
        fin["emergency_fund_months"] = 8.0  # healthy EFM → Rule 3 doesn't fire
        fin["price_to_income_ratio"] = 0.15
        # Need 2 rules; only 0 trigger here
        result = engine.decide(fin, _healthy_product())
        assert result.decision_category == "GREEN"

    def test_rule4_requires_income_pressure(self, engine):
        """low_resilience requires affordability < 0 (G1) — thin savings
        alone isn't enough."""
        fin = _healthy_financial()
        fin["emergency_fund_months"] = 2.0
        fin["saving_to_income_ratio"] = 0.20
        fin["affordability_score"] = 500  # positive → Rule 4 doesn't fire
        result = engine.decide(fin, _healthy_product())
        # Rule 4 won't fire, so we might get 0 or 1 rule → GREEN
        assert result.decision_category == "GREEN"

    def test_rule5_requires_significant_purchase(self, engine):
        """weak_profile requires PIR > 0.15 — cheap purchases don't warn."""
        fin = _healthy_financial()
        fin["credit_risk_indicator"] = 0.30
        fin["net_worth_indicator"] = 0.5
        fin["price_to_income_ratio"] = 0.05  # cheap → Rule 5 doesn't fire
        fin["savings_to_price_ratio"] = 8.0
        result = engine.decide(fin, _healthy_product())
        assert result.decision_category == "GREEN"


# ── EDGE CASES ─────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_dicts_dont_crash(self, engine):
        with pytest.raises(ValueError, match="Missing required financial feature: affordability_score"):
            engine.decide({}, {})

    def test_nan_values_handled(self, engine):
        fin = _healthy_financial()
        fin["affordability_score"] = float("nan")
        with pytest.raises(ValueError, match="NaN passed for financial feature: affordability_score"):
            engine.decide(fin, _healthy_product())

    def test_none_values_handled(self, engine):
        fin = _healthy_financial()
        fin["affordability_score"] = None
        with pytest.raises(ValueError, match="Missing required financial feature: affordability_score"):
            engine.decide(fin, _healthy_product())

    def test_decide_row_wrapper(self, engine):
        row = pd.Series({
            **_healthy_financial(),
            "price": 50.0,
        })
        assert engine.decide_row(row) == "GREEN"

    def test_result_contains_rules(self, engine):
        fin = _healthy_financial()
        fin["affordability_score"] = -200
        fin["price_to_income_ratio"] = 0.30
        fin["savings_to_price_ratio"] = 3.0
        fin["residual_utility_score"] = 2.0
        result = engine.decide(fin, _healthy_product())
        assert len(result.triggered_rules) >= 1

    def test_no_product_reviews_used(self, engine):
        """Product reviews have zero influence on labeling."""
        prod_with_bad_reviews = {"price": 50.0, "average_rating": 1.0, "rating_number": 1}
        prod_with_good_reviews = {"price": 50.0, "average_rating": 5.0, "rating_number": 10000}
        fin = _healthy_financial()
        result_bad = engine.decide(fin, prod_with_bad_reviews)
        result_good = engine.decide(fin, prod_with_good_reviews)
        assert result_bad.decision_category == result_good.decision_category == "GREEN"
