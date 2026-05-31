"""Tests for guardrails — all 6 safety checks (G1–G6)."""

import pytest

from llm.guardrails import check_response, GuardrailResult
from llm.response_generator import RecommendationContext


@pytest.fixture
def green_context():
    return RecommendationContext(
        product_name="Sony WH-1000XM5",
        product_price=349.99,
        recommendation_color="GREEN",
    )


@pytest.fixture
def red_context():
    return RecommendationContext(
        product_name="MacBook Pro",
        product_price=2499.00,
        recommendation_color="RED",
    )


class TestG1ColorContradiction:
    """G1: Response must not contradict the recommendation color."""

    def test_green_with_alarm_fails(self, green_context):
        response = "Don't buy this product. Stay away from it."
        result = check_response(response, "GREEN", green_context)
        assert not result.passed
        assert any("G1" in v for v in result.violations)

    def test_red_with_encouragement_fails(self, red_context):
        response = "Go ahead and buy this! Great choice for you."
        result = check_response(response, "RED", red_context)
        assert not result.passed
        assert any("G1" in v for v in result.violations)

    def test_green_encouraging_passes(self, green_context):
        response = "This purchase fits well within your budget."
        result = check_response(response, "GREEN", green_context)
        # Should not flag G1
        g1_violations = [v for v in result.violations if "G1" in v]
        assert len(g1_violations) == 0

    def test_red_discouraging_passes(self, red_context):
        response = "I'd recommend holding off on this purchase right now."
        result = check_response(response, "RED", red_context)
        g1_violations = [v for v in result.violations if "G1" in v]
        assert len(g1_violations) == 0


class TestG2HallucinatedFigures:
    """G2: Response must not contain dollar amounts not in the input."""

    def test_correct_price_passes(self, green_context):
        response = "Purchasing at $349.99 is a good choice."
        result = check_response(response, "GREEN", green_context)
        g2_violations = [v for v in result.violations if "G2" in v]
        assert len(g2_violations) == 0

    def test_invented_price_fails(self, green_context):
        response = "This will cost you $999.99 per month in payments."
        result = check_response(response, "GREEN", green_context)
        assert any("G2" in v for v in result.violations)

    def test_no_dollar_amounts_passes(self, green_context):
        response = "This purchase fits well within your budget."
        result = check_response(response, "GREEN", green_context)
        g2_violations = [v for v in result.violations if "G2" in v]
        assert len(g2_violations) == 0


class TestG3OutOfScope:
    """G3: Response must not contain out-of-scope financial advice."""

    def test_investment_advice_fails(self, green_context):
        response = "Instead of buying this, invest in index funds for better returns."
        result = check_response(response, "GREEN", green_context)
        assert any("G3" in v for v in result.violations)

    def test_credit_card_advice_fails(self, green_context):
        response = "Apply for a credit card with rewards to buy this cheaper."
        result = check_response(response, "GREEN", green_context)
        assert any("G3" in v for v in result.violations)

    def test_stock_advice_fails(self, green_context):
        response = "Consider putting this money in the stock market instead."
        result = check_response(response, "GREEN", green_context)
        assert any("G3" in v for v in result.violations)

    def test_purchase_advice_passes(self, green_context):
        response = "This purchase fits your budget well. Enjoy your new headphones!"
        result = check_response(response, "GREEN", green_context)
        g3_violations = [v for v in result.violations if "G3" in v]
        assert len(g3_violations) == 0


class TestG4InternalLeakage:
    """G4: Response must not expose system internals."""

    def test_deterministic_engine_leak_fails(self, green_context):
        response = "The deterministic engine classified this as GREEN."
        result = check_response(response, "GREEN", green_context)
        assert any("G4" in v for v in result.violations)

    def test_model_name_leak_fails(self, green_context):
        response = "Our LightGBM model is 87% confident you can buy this."
        result = check_response(response, "GREEN", green_context)
        assert any("G4" in v for v in result.violations)

    def test_rule_name_leak_fails(self, red_context):
        response = "Rule RED_1 fired because you can't afford this from any angle."
        result = check_response(response, "RED", red_context)
        assert any("G4" in v for v in result.violations)

    def test_pgvector_leak_fails(self, green_context):
        response = "I found your product using pgvector cosine similarity search."
        result = check_response(response, "GREEN", green_context)
        assert any("G4" in v for v in result.violations)

    def test_normal_response_passes(self, green_context):
        response = "This looks like a great purchase for your budget!"
        result = check_response(response, "GREEN", green_context)
        g4_violations = [v for v in result.violations if "G4" in v]
        assert len(g4_violations) == 0


class TestG5ToneMismatch:
    """G5: Response tone must match the color."""

    def test_green_with_many_alarm_words_fails(self, green_context):
        response = (
            "Don't buy this product. You cannot afford it. "
            "Stay away from this purchase. Avoid it."
        )
        result = check_response(response, "GREEN", green_context)
        # G1 and/or G5 should catch this
        has_tone_issue = any("G1" in v or "G5" in v for v in result.violations)
        assert has_tone_issue

    def test_red_with_many_encourage_words_fails(self, red_context):
        response = (
            "Go ahead and buy this! Absolutely perfect choice. "
            "Definitely buy it — great pick!"
        )
        result = check_response(response, "RED", red_context)
        has_tone_issue = any("G1" in v or "G5" in v for v in result.violations)
        assert has_tone_issue


class TestG6LengthCheck:
    """G6: Response must not exceed maximum word limit."""

    def test_short_response_passes(self, green_context):
        response = "This purchase fits well within your budget."
        result = check_response(response, "GREEN", green_context)
        g6_violations = [v for v in result.violations if "G6" in v]
        assert len(g6_violations) == 0

    def test_excessively_long_response_fails(self, green_context):
        # Generate a response way over the limit
        response = " ".join(["word"] * 200)
        result = check_response(response, "GREEN", green_context)
        assert any("G6" in v for v in result.violations)


class TestGuardrailResult:
    def test_clean_response_passes_all(self, green_context):
        response = "This purchase at $349.99 fits well within your budget."
        result = check_response(response, "GREEN", green_context)
        assert result.passed
        assert len(result.violations) == 0

    def test_result_is_dataclass(self, green_context):
        result = check_response("test", "GREEN", green_context)
        assert isinstance(result, GuardrailResult)
        assert isinstance(result.violations, list)

    def test_multiple_violations_captured(self, red_context):
        response = (
            "Go ahead and buy this! The deterministic engine says it's fine. "
            "Your savings of $99999.99 will cover it."
        )
        result = check_response(response, "RED", red_context)
        assert not result.passed
        # Multiple guardrails should fire (G1, G2, G4)
        assert len(result.violations) >= 2
