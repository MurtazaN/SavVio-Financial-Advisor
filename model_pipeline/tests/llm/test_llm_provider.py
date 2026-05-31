"""Tests for the LLM provider abstraction."""

import pytest

from llm.llm_provider import MockProvider, get_provider, _provider_cache


@pytest.fixture(autouse=True)
def _clear_provider_cache():
    """Clear the singleton cache between tests."""
    _provider_cache.clear()
    yield
    _provider_cache.clear()


class TestMockProvider:
    def test_provider_name(self):
        provider = MockProvider()
        assert provider.provider_name == "mock"

    def test_generate_returns_string(self):
        provider = MockProvider()
        result = provider.generate("system", "hello world")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_includes_input(self):
        provider = MockProvider()
        result = provider.generate("system", "test input message")
        assert "test input message" in result

    def test_generate_structured_returns_dict(self):
        provider = MockProvider()
        result = provider.generate_structured(
            "system",
            "Can I buy the iPhone?",
            schema={"intent": "string"},
        )
        assert isinstance(result, dict)
        assert "intent" in result

    def test_generate_structured_purchase_intent(self):
        provider = MockProvider()
        result = provider.generate_structured(
            "system",
            "Sony headphones",
            schema={"intent": "string"},
        )
        assert result["intent"] == "purchase_query"
        assert result["product_reference"] == "Sony headphones"


class TestGetProvider:
    def test_default_is_mock(self):
        provider = get_provider()
        assert isinstance(provider, MockProvider)

    def test_explicit_mock(self):
        provider = get_provider("mock")
        assert isinstance(provider, MockProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_provider("nonexistent")

    def test_singleton_caching(self):
        p1 = get_provider("mock")
        p2 = get_provider("mock")
        assert p1 is p2

    def test_openai_requires_key(self):
        """OpenAI provider should fail without API key."""
        import os
        os.environ.pop("OPENAI_API_KEY", None)
        _provider_cache.clear()
        # Either ImportError (no package) or ValueError (no key)
        with pytest.raises((ImportError, ValueError)):
            get_provider("openai")

    def test_gemini_requires_key(self):
        """Gemini provider should fail without API key."""
        import os
        os.environ.pop("GEMINI_API_KEY", None)
        _provider_cache.clear()
        with pytest.raises((ImportError, ValueError)):
            get_provider("gemini")

    def test_claude_requires_key(self):
        """Claude provider should fail without API key."""
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        _provider_cache.clear()
        with pytest.raises((ImportError, ValueError)):
            get_provider("claude")
