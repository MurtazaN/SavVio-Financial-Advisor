"""
SavVio LLM Integration Package.

This package provides the complete LLM layer for SavVio — handling natural
language user input, product resolution, and conversational response generation.

Components:
    - llm_provider:       Provider abstraction (mock / OpenAI / Gemini / Claude)
    - intent_parser:      Role 1 — parse user queries, extract product references
    - product_resolver:   Resolve product text to product_id via pgvector search
    - response_generator: Role 2 — generate conversational recommendations
    - guardrails:         Output safety verification (6 code-level checks)
    - config:             LLM-specific configuration

Usage:
    from llm.intent_parser import parse_user_input
    from llm.product_resolver import resolve_product
    from llm.response_generator import generate_response, RecommendationContext
    from llm.guardrails import check_response
    from llm.llm_provider import get_provider

Legacy interface (backward compatible with run_pipeline.py):
    from llm.prompt_engin import apply_llm_guardrails
"""

from llm.llm_provider import get_provider, LLMProvider
from llm.intent_parser import parse_user_input, ParsedIntent
from llm.product_resolver import resolve_product, ProductMatch
from llm.response_generator import generate_response, RecommendationContext
from llm.guardrails import check_response, GuardrailResult

__all__ = [
    "get_provider",
    "LLMProvider",
    "parse_user_input",
    "ParsedIntent",
    "resolve_product",
    "ProductMatch",
    "generate_response",
    "RecommendationContext",
    "check_response",
    "GuardrailResult",
]
