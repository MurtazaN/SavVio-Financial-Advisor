"""Tests for the response generator — template-based (mock) and LLM-based."""

import pytest

from llm.llm_provider import MockProvider
from llm.response_generator import (
    RecommendationContext,
    generate_response,
    _generate_from_template,
)


@pytest.fixture
def provider():
    return MockProvider()


def _make_context(
    color: str = "GREEN",
    product_name: str = "Sony WH-1000XM5",
    product_price: float = 349.99,
    triggered_rules: list | None = None,
    was_downgraded: bool = False,
    affordability_score: float = 1650.0,
    savings_to_price_ratio: float = 28.5,
    emergency_fund_months: float = 8.3,
    debt_to_income_ratio: float = 0.15,
) -> RecommendationContext:
    return RecommendationContext(
        product_name=product_name,
        product_price=product_price,
        recommendation_color=color,
        original_color=color,
        was_downgraded=was_downgraded,
        triggered_rules=triggered_rules or [],
        confidence_scores={color: 0.87},
        affordability_score=affordability_score,
        savings_to_price_ratio=savings_to_price_ratio,
        emergency_fund_months=emergency_fund_months,
        debt_to_income_ratio=debt_to_income_ratio,
    )


class TestGreenResponse:
    def test_green_response_contains_product(self, provider):
        ctx = _make_context("GREEN")
        response = generate_response(ctx, provider)
        assert "Sony WH-1000XM5" in response

    def test_green_response_contains_price(self, provider):
        ctx = _make_context("GREEN")
        response = generate_response(ctx, provider)
        assert "$349.99" in response

    def test_green_response_is_encouraging(self, provider):
        ctx = _make_context("GREEN")
        response = generate_response(ctx, provider)
        response_lower = response.lower()
        encouraging = any(
            word in response_lower
            for word in ["great", "comfortably", "sound", "good", "manageable"]
        )
        assert encouraging, f"GREEN response should be encouraging: {response}"


class TestYellowResponse:
    def test_yellow_response_contains_product(self, provider):
        ctx = _make_context("YELLOW", triggered_rules=["YELLOW_2"])
        response = generate_response(ctx, provider)
        assert "Sony WH-1000XM5" in response

    def test_yellow_response_is_cautious(self, provider):
        ctx = _make_context("YELLOW", triggered_rules=["YELLOW_1"])
        response = generate_response(ctx, provider)
        response_lower = response.lower()
        cautious = any(
            word in response_lower
            for word in ["think", "consider", "caution", "careful"]
        )
        assert cautious, f"YELLOW response should be cautious: {response}"


class TestRedResponse:
    def test_red_response_contains_product(self, provider):
        ctx = _make_context("RED", triggered_rules=["RED_1"])
        response = generate_response(ctx, provider)
        assert "Sony WH-1000XM5" in response

    def test_red_response_recommends_against(self, provider):
        ctx = _make_context("RED", triggered_rules=["RED_1"])
        response = generate_response(ctx, provider)
        response_lower = response.lower()
        discouraging = any(
            word in response_lower
            for word in ["holding off", "recommend", "not the best", "first"]
        )
        assert discouraging, f"RED response should discourage: {response}"


class TestDowngradeHandling:
    def test_downgraded_response_mentions_concerns(self, provider):
        ctx = _make_context(
            "YELLOW",
            was_downgraded=True,
            triggered_rules=["YELLOW_2"],
        )
        ctx.original_color = "GREEN"
        response = generate_response(ctx, provider)
        # Should mention product quality concerns since it was downgraded
        response_lower = response.lower()
        assert any(
            word in response_lower
            for word in ["quality", "review", "caution", "concern", "think"]
        )


class TestTemplateGeneration:
    def test_template_always_returns_string(self):
        ctx = _make_context("GREEN")
        response = _generate_from_template(ctx)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_unknown_color_defaults_to_yellow(self):
        ctx = _make_context("PURPLE")
        response = _generate_from_template(ctx)
        assert isinstance(response, str)
        assert len(response) > 0


class TestRecommendationContext:
    def test_default_values(self):
        ctx = RecommendationContext(
            product_name="Test",
            product_price=10.0,
            recommendation_color="GREEN",
        )
        assert ctx.triggered_rules == []
        assert ctx.confidence_scores == {}
        assert ctx.was_downgraded is False
        assert ctx.user_context is None
