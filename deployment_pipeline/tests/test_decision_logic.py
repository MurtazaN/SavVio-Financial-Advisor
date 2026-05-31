"""
Deterministic Engine Rule Tests for SavVio Deployment.

Tests the DecisionEngine (Layer 1) and DowngradeEngine (Layer 2)
directly with crafted financial/product/review inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure model_pipeline/src is on the path
_project_root = Path(__file__).resolve().parent.parent.parent
_model_src = _project_root / "model_pipeline" / "src"
for p in [str(_project_root), str(_model_src)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from deterministic_engine.financial_engine import DecisionEngine, DecisionResult
from deterministic_engine.downgrade_engine import DowngradeEngine, DowngradeResult
from features.product_features import ProductFeatures
from features.review_features import ReviewFeatures


# ---------------------------------------------------------------------------
# Helpers — build financial dicts with specific characteristics
# ---------------------------------------------------------------------------

def _make_financial(
    affordability_score=500.0,
    price_to_income_ratio=0.05,
    savings_to_price_ratio=10.0,
    net_worth_indicator=3.0,
    credit_risk_indicator=0.8,
    debt_to_income_ratio=0.10,
    monthly_expense_burden_ratio=0.40,
    emergency_fund_months=6.0,
) -> dict:
    """Build a financial features dict with controllable values."""
    return {
        "affordability_score": affordability_score,
        "price_to_income_ratio": price_to_income_ratio,
        "savings_to_price_ratio": savings_to_price_ratio,
        "net_worth_indicator": net_worth_indicator,
        "credit_risk_indicator": credit_risk_indicator,
        "debt_to_income_ratio": debt_to_income_ratio,
        "monthly_expense_burden_ratio": monthly_expense_burden_ratio,
        "emergency_fund_months": emergency_fund_months,
    }


def _make_product_features(
    value_density=1.0,
    review_confidence=0.7,
    rating_polarization=0.2,
    quality_risk_score=0.3,
    cold_start_flag=0,
    price_category_rank=0.4,
    category_rating_deviation=0.1,
) -> ProductFeatures:
    return ProductFeatures(
        value_density=value_density,
        review_confidence=review_confidence,
        rating_polarization=rating_polarization,
        quality_risk_score=quality_risk_score,
        cold_start_flag=cold_start_flag,
        price_category_rank=price_category_rank,
        category_rating_deviation=category_rating_deviation,
    )


def _make_review_features(
    verified_purchase_ratio=0.7,
    helpful_concentration=0.3,
    sentiment_spread=0.5,
    review_depth_score=0.6,
    reviewer_diversity=0.8,
    extreme_rating_ratio=0.3,
) -> ReviewFeatures:
    return ReviewFeatures(
        verified_purchase_ratio=verified_purchase_ratio,
        helpful_concentration=helpful_concentration,
        sentiment_spread=sentiment_spread,
        review_depth_score=review_depth_score,
        reviewer_diversity=reviewer_diversity,
        extreme_rating_ratio=extreme_rating_ratio,
    )


# ---------------------------------------------------------------------------
# Layer 1 — DecisionEngine Tests
# ---------------------------------------------------------------------------

class TestDecisionEngineGreen:
    """GREEN is the default when no RED or YELLOW rules trigger."""

    def test_green_all_clear(self):
        engine = DecisionEngine()
        result = engine.decide(_make_financial(), {"price": 100})
        assert result.decision_category == "GREEN"
        assert "green:all_clear" in result.triggered_rules

    def test_green_with_moderate_values(self):
        """Healthy but not perfect financials → still GREEN."""
        engine = DecisionEngine()
        financial = _make_financial(
            affordability_score=200.0,
            savings_to_price_ratio=6.0,
            emergency_fund_months=4.0,
            debt_to_income_ratio=0.15,
        )
        result = engine.decide(financial, {"price": 200})
        assert result.decision_category == "GREEN"


class TestDecisionEngineRed:
    """RED rules require compound conditions across correlation groups."""

    def test_red_rule_1_cant_afford_from_any_angle(self):
        """RED Rule 1: affordability < 0 AND spr < 1.5 AND pir > 0.10."""
        engine = DecisionEngine()
        financial = _make_financial(
            affordability_score=-200.0,
            savings_to_price_ratio=1.0,
            price_to_income_ratio=0.15,
        )
        result = engine.decide(financial, {"price": 500})
        assert result.decision_category == "RED"
        assert any("cant_afford" in r for r in result.triggered_rules)

    def test_red_rule_2_maxed_budget(self):
        """RED Rule 2: meb > 0.80 AND pir > 0.20 AND efm < 3.0."""
        engine = DecisionEngine()
        financial = _make_financial(
            monthly_expense_burden_ratio=0.85,
            price_to_income_ratio=0.25,
            emergency_fund_months=2.0,
        )
        result = engine.decide(financial, {"price": 1000})
        assert result.decision_category == "RED"
        assert any("maxed_budget" in r for r in result.triggered_rules)

    def test_red_rule_3_underwater(self):
        """RED Rule 3: nwi < -2.0 AND affordability < 0 AND pir > 0.15 AND efm < 3.0."""
        engine = DecisionEngine()
        financial = _make_financial(
            net_worth_indicator=-3.0,
            affordability_score=-100.0,
            price_to_income_ratio=0.20,
            emergency_fund_months=2.0,
        )
        result = engine.decide(financial, {"price": 500})
        assert result.decision_category == "RED"
        assert any("underwater" in r for r in result.triggered_rules)

    def test_red_rule_4_paycheck_to_paycheck(self):
        """RED Rule 4: efm < 1.0 AND dti > 0.30 AND pir > 0.10."""
        engine = DecisionEngine()
        financial = _make_financial(
            emergency_fund_months=0.5,
            debt_to_income_ratio=0.35,
            price_to_income_ratio=0.15,
        )
        result = engine.decide(financial, {"price": 500})
        assert result.decision_category == "RED"
        assert any("paycheck" in r for r in result.triggered_rules)

    def test_pir_escape_hatch_prevents_red(self):
        """Trivial purchases (low PIR) should NEVER trigger RED."""
        engine = DecisionEngine()
        # Same as RED Rule 1 conditions but PIR is very low (trivial purchase)
        financial = _make_financial(
            affordability_score=-200.0,
            savings_to_price_ratio=1.0,
            price_to_income_ratio=0.02,  # Very cheap relative to income
        )
        result = engine.decide(financial, {"price": 5})
        # Should NOT be RED because PIR escape hatch
        assert result.decision_category != "RED"


class TestDecisionEngineYellow:
    """YELLOW rules trigger when >= 1 rule fires."""

    def test_yellow_income_pressure(self):
        """YELLOW Rule 1: affordability < 0 AND pir > 0.25 AND spr < 5.0."""
        engine = DecisionEngine()
        financial = _make_financial(
            affordability_score=-50.0,
            price_to_income_ratio=0.30,
            savings_to_price_ratio=4.0,
        )
        result = engine.decide(financial, {"price": 500})
        assert result.decision_category == "YELLOW"
        assert any("income_pressure" in r for r in result.triggered_rules)

    def test_yellow_savings_strain(self):
        """YELLOW Rule 2: spr < 5.0 AND pir > 0.10."""
        engine = DecisionEngine()
        financial = _make_financial(
            savings_to_price_ratio=3.0,
            price_to_income_ratio=0.15,
        )
        result = engine.decide(financial, {"price": 500})
        assert result.decision_category == "YELLOW"
        assert any("savings_strain" in r for r in result.triggered_rules)

    def test_yellow_low_resilience(self):
        """YELLOW Rule 4: efm < 3.0 AND affordability < 0."""
        engine = DecisionEngine()
        financial = _make_financial(
            emergency_fund_months=2.0,
            affordability_score=-100.0,
        )
        result = engine.decide(financial, {"price": 300})
        assert result.decision_category == "YELLOW"
        assert any("low_resilience" in r for r in result.triggered_rules)


class TestDecisionEngineMissingFeatures:
    """Missing or NaN features should raise ValueError."""

    def test_missing_affordability_raises(self):
        engine = DecisionEngine()
        financial = _make_financial()
        del financial["affordability_score"]
        financial["affordability_score"] = None
        with pytest.raises(ValueError, match="Missing required"):
            engine.decide(financial, {"price": 100})

    def test_nan_feature_raises(self):
        engine = DecisionEngine()
        financial = _make_financial()
        financial["affordability_score"] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            engine.decide(financial, {"price": 100})


# ---------------------------------------------------------------------------
# Layer 2 — DowngradeEngine Tests
# ---------------------------------------------------------------------------

class TestDowngradeEngine:
    """Tests for the product/review-based downgrade engine."""

    def test_no_downgrade_with_good_signals(self):
        """Clean product + review signals → no downgrade."""
        engine = DowngradeEngine()
        result = engine.evaluate(
            "GREEN",
            _make_product_features(),
            _make_review_features(),
        )
        assert result.final_label == "GREEN"
        assert not result.was_downgraded

    def test_green_to_yellow_downgrade(self):
        """Product AND review concerns → GREEN downgraded to YELLOW."""
        engine = DowngradeEngine()
        # PR1: category_rating_deviation < -0.5 AND review_confidence < 0.3
        product = _make_product_features(
            category_rating_deviation=-0.7,
            review_confidence=0.2,
        )
        # RR1: verified_purchase_ratio < 0.3 AND extreme_rating_ratio > 0.8 AND review_depth_score < 0.2
        review = _make_review_features(
            verified_purchase_ratio=0.2,
            extreme_rating_ratio=0.9,
            review_depth_score=0.1,
        )
        result = engine.evaluate("GREEN", product, review)
        assert result.final_label == "YELLOW"
        assert result.was_downgraded

    def test_yellow_to_red_downgrade(self):
        """Product AND review concerns → YELLOW downgraded to RED."""
        engine = DowngradeEngine()
        product = _make_product_features(
            category_rating_deviation=-0.7,
            review_confidence=0.2,
        )
        review = _make_review_features(
            verified_purchase_ratio=0.2,
            extreme_rating_ratio=0.9,
            review_depth_score=0.1,
        )
        result = engine.evaluate("YELLOW", product, review)
        assert result.final_label == "RED"
        assert result.was_downgraded

    def test_red_stays_red(self):
        """RED cannot be upgraded — stays RED even with no triggers."""
        engine = DowngradeEngine()
        result = engine.evaluate(
            "RED",
            _make_product_features(),
            _make_review_features(),
        )
        assert result.final_label == "RED"
        assert not result.was_downgraded

    def test_product_only_no_downgrade(self):
        """Only product triggers (no review triggers) → no downgrade."""
        engine = DowngradeEngine()
        product = _make_product_features(
            category_rating_deviation=-0.7,
            review_confidence=0.2,
        )
        # Clean review features
        review = _make_review_features()
        result = engine.evaluate("GREEN", product, review)
        assert result.final_label == "GREEN"
        assert not result.was_downgraded

    def test_review_only_no_downgrade(self):
        """Only review triggers (no product triggers) → no downgrade."""
        engine = DowngradeEngine()
        # Clean product features
        product = _make_product_features()
        review = _make_review_features(
            verified_purchase_ratio=0.2,
            extreme_rating_ratio=0.9,
            review_depth_score=0.1,
        )
        result = engine.evaluate("GREEN", product, review)
        assert result.final_label == "GREEN"
        assert not result.was_downgraded
