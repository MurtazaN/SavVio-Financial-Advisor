"""
LLM Guardrails — Legacy Interface.

This module preserves backward compatibility with run_pipeline.py while
delegating to the new LLM integration components.

The original placeholder has been replaced by a full LLM integration stack:
    - llm_provider.py   — Provider abstraction (mock / OpenAI / Gemini / Claude)
    - intent_parser.py  — Role 1: natural language understanding
    - product_resolver.py — pgvector product search
    - response_generator.py — Role 2: conversational recommendations
    - guardrails.py     — Output safety verification
"""

from __future__ import annotations

import logging

import mlflow

from llm.config import LLMConfig
from llm.guardrails import check_response
from llm.llm_provider import get_provider
from llm.response_generator import RecommendationContext, generate_response

logger = logging.getLogger(__name__)


def apply_llm_guardrails(y_pred, user_data: dict, model_confidence: float):
    """
    Legacy interface — now delegates to the full LLM integration stack.

    Wraps the deterministic engine output + ML confidence into a natural
    language response and verifies it against safety guardrails.

    Args:
        y_pred: Model prediction label (GREEN/YELLOW/RED).
        user_data: Dict with user financial profile + product context.
        model_confidence: Float confidence from the ML model.

    Returns:
        Tuple of (is_safe: bool, explanation: str).
    """
    logger.info("Evaluating output via LLM guardrails (v2)...")

    # Log the prompt template version for lineage
    mlflow.log_param("llm_prompt_template", f"system_{LLMConfig.SYSTEM_PROMPT_VERSION}")
    mlflow.log_param("llm_provider", LLMConfig.PROVIDER)

    # Build recommendation context from legacy inputs
    color = str(y_pred).upper() if not isinstance(y_pred, str) else y_pred.upper()
    context = RecommendationContext(
        product_name=user_data.get("product_name", "the product"),
        product_price=float(user_data.get("product_price", 0.0)),
        recommendation_color=color,
        original_color=color,
        was_downgraded=bool(user_data.get("was_downgraded", False)),
        triggered_rules=user_data.get("triggered_rules", []),
        confidence_scores={color: model_confidence},
        affordability_score=float(user_data.get("affordability_score", 0.0)),
        savings_to_price_ratio=float(user_data.get("savings_to_price_ratio", 0.0)),
        emergency_fund_months=float(user_data.get("emergency_fund_months", 0.0)),
        debt_to_income_ratio=float(user_data.get("debt_to_income_ratio", 0.0)),
    )

    # Generate response
    provider = get_provider()
    explanation = generate_response(context, provider)

    # Run guardrail checks
    result = check_response(explanation, color, context)

    if not result.passed:
        logger.warning(
            "Guardrail violations detected: %s — using sanitized fallback",
            result.violations,
        )
        # Fall back to template response (always safe)
        from llm.response_generator import _generate_from_template
        explanation = _generate_from_template(context)
        # Re-check the fallback (should always pass)
        result = check_response(explanation, color, context)

    is_safe = result.passed
    logger.info("LLM guardrails result: is_safe=%s, provider=%s", is_safe, provider.provider_name)

    return is_safe, explanation
