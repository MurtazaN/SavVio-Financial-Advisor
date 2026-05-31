"""Tests for the intent parser — all input patterns from the team Slack conversation."""

import pytest

from llm.intent_parser import parse_user_input, ParsedIntent
from llm.llm_provider import MockProvider


@pytest.fixture
def provider():
    return MockProvider()


class TestPurchaseIntentDetection:
    """Test all the input patterns mentioned in the Slack conversation."""

    def test_can_i_buy(self, provider):
        result = parse_user_input("Can I buy the iPhone 15?", provider)
        assert result.intent == "purchase_query"
        assert "iPhone 15" in result.product_reference

    def test_should_i_buy(self, provider):
        result = parse_user_input("Should I buy the Sony WH-1000XM5?", provider)
        assert result.intent == "purchase_query"
        assert "Sony WH-1000XM5" in result.product_reference

    def test_can_i_purchase(self, provider):
        result = parse_user_input("Can I purchase the MacBook Pro?", provider)
        assert result.intent == "purchase_query"
        assert "MacBook Pro" in result.product_reference

    def test_thinking_of_buying_with_context(self, provider):
        query = (
            "I have some money saved up and am thinking of buying "
            "the Samsung Galaxy S24, what do you think savvio?"
        )
        result = parse_user_input(query, provider)
        assert result.intent == "purchase_query"
        assert "Samsung Galaxy S24" in result.product_reference

    def test_could_i_afford(self, provider):
        result = parse_user_input("Could I afford the Nike Air Max?", provider)
        assert result.intent == "purchase_query"
        assert "Nike Air Max" in result.product_reference

    def test_is_it_worth_buying(self, provider):
        result = parse_user_input("Is it worth buying the iPad Air?", provider)
        assert result.intent == "purchase_query"
        assert "iPad Air" in result.product_reference

    def test_i_want_to_buy(self, provider):
        result = parse_user_input("I want to buy the PlayStation 5", provider)
        assert result.intent == "purchase_query"
        assert "PlayStation 5" in result.product_reference

    def test_should_i_get(self, provider):
        result = parse_user_input("Should I get the LG OLED TV?", provider)
        assert result.intent == "purchase_query"
        assert "LG OLED TV" in result.product_reference

    def test_product_with_title_variation(self, provider):
        """Product can be a name, title, or ID."""
        result = parse_user_input("Can I buy B0BX1234?", provider)
        assert result.intent == "purchase_query"
        assert "B0BX1234" in result.product_reference


class TestOutOfScopeDetection:
    def test_weather_query(self, provider):
        result = parse_user_input("What is the weather today?", provider)
        assert result.intent == "out_of_scope"
        assert result.product_reference is None

    def test_random_question(self, provider):
        result = parse_user_input("Tell me a joke", provider)
        assert result.intent == "out_of_scope"

    def test_empty_query(self, provider):
        result = parse_user_input("", provider)
        assert result.intent == "out_of_scope"

    def test_whitespace_only(self, provider):
        result = parse_user_input("   ", provider)
        assert result.intent == "out_of_scope"


class TestGeneralQuestionDetection:
    def test_financial_question(self, provider):
        result = parse_user_input("What is my savings balance?", provider)
        assert result.intent == "general_question"

    def test_budget_question(self, provider):
        result = parse_user_input("How is my budget looking?", provider)
        assert result.intent == "general_question"


class TestProductExtraction:
    def test_strips_trailing_punctuation(self, provider):
        result = parse_user_input("Can I buy the iPhone 15?", provider)
        assert result.product_reference is not None
        assert not result.product_reference.endswith("?")

    def test_strips_savvio_mention(self, provider):
        result = parse_user_input(
            "Can I buy the AirPods Pro, savvio?", provider
        )
        assert result.product_reference is not None
        assert "savvio" not in result.product_reference.lower()

    def test_multi_word_product(self, provider):
        result = parse_user_input(
            "Should I buy the Sony WH-1000XM5 Wireless Headphones?", provider
        )
        assert result.intent == "purchase_query"
        assert "Sony" in result.product_reference
        assert "Headphones" in result.product_reference


class TestConfidence:
    def test_exact_pattern_match_high_confidence(self, provider):
        result = parse_user_input("Can I buy the iPhone 15?", provider)
        assert result.confidence >= 0.9

    def test_out_of_scope_has_confidence(self, provider):
        result = parse_user_input("What is the weather?", provider)
        assert result.confidence > 0.0


class TestParsedIntentDataclass:
    def test_dataclass_fields(self):
        intent = ParsedIntent(
            intent="purchase_query",
            product_reference="iPhone",
            user_context="I have savings",
            confidence=0.95,
        )
        assert intent.intent == "purchase_query"
        assert intent.product_reference == "iPhone"
        assert intent.user_context == "I have savings"
        assert intent.confidence == 0.95
