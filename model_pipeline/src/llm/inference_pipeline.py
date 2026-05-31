"""
Inference Pipeline — End-to-end SavVio recommendation orchestrator.

Chains all LLM integration components together for a single user request:
    1. Intent parsing     — understand what the user is asking
    2. Product resolution — find the product via pgvector search
    3. Response generation — LLM-powered conversational explanation
    4. Guardrail verification — safety checks before response delivery

The financial evaluation (deterministic engine + ML model) is performed by the
calling layer (e.g., the FastAPI backend) and passed in as `recommendation_color`,
`triggered_rules`, etc. This module handles only the LLM-specific steps.

Usage:
    from llm.inference_pipeline import run_recommendation
    from llm.llm_provider import get_provider

    provider = get_provider()
    result = run_recommendation(
        user_query="Can I buy the Sony WH-1000XM5?",
        recommendation_color="GREEN",
        triggered_rules=["green:all_clear"],
        was_downgraded=False,
        model_confidence=0.87,
        financial_context={
            "affordability_score": 250.0,
            "savings_to_price_ratio": 6.2,
            "emergency_fund_months": 5.0,
            "debt_to_income_ratio": 0.18,
        },
        provider=provider,
        db_engine=engine,
    )
    print(result.response)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from llm.guardrails import check_response
from llm.intent_parser import ParsedIntent, parse_user_input
from llm.llm_provider import LLMProvider
from llm.product_resolver import ProductMatch, resolve_product
from llm.response_generator import RecommendationContext, generate_response
from llm.prompts.response_templates import OUT_OF_SCOPE_RESPONSE, PRODUCT_NOT_FOUND_RESPONSE

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Full result of one end-to-end recommendation request."""
    # Input parsing
    intent: str
    product_reference: str | None

    # Product resolution
    product_match: ProductMatch | None

    # Final recommendation
    recommendation_color: str
    response: str

    # Guardrail outcome
    guardrail_passed: bool
    guardrail_violations: list[str] = field(default_factory=list)

    # Metadata
    was_downgraded: bool = False
    triggered_rules: list[str] = field(default_factory=list)


def run_recommendation(
    user_query: str,
    recommendation_color: str,
    triggered_rules: list[str],
    was_downgraded: bool,
    model_confidence: float,
    financial_context: dict,
    provider: LLMProvider,
    db_engine=None,
    original_color: str = "",
) -> InferenceResult:
    """
    Orchestrate the full LLM recommendation flow for one user request.

    Steps:
        1. Parse the user query (intent + product reference).
        2. Resolve the product via pgvector similarity search (if db_engine provided).
        3. Build the recommendation context from deterministic engine output + ML confidence.
        4. Generate a natural language response via the LLM provider.
        5. Run guardrail checks; fall back to template if violations detected.

    Args:
        user_query:            Raw natural language query from the user.
        recommendation_color:  Authoritative label (GREEN/YELLOW/RED) from the deterministic engine.
        triggered_rules:       List of rule strings that fired (from financial_engine + downgrade_engine).
        was_downgraded:        True if Layer 2 downgraded the Layer 1 label.
        model_confidence:      ML model confidence for the recommendation_color class.
        financial_context:     Dict with affordability_score, savings_to_price_ratio,
                               emergency_fund_months, debt_to_income_ratio.
        provider:              LLMProvider instance (mock / openai / gemini / claude).
        db_engine:             SQLAlchemy engine for product resolution. None skips Step 2.
        original_color:        Pre-downgrade color (if was_downgraded is True).

    Returns:
        InferenceResult with the final response and all intermediate outputs.
    """
    # --- Step 1: Parse intent ---
    parsed: ParsedIntent = parse_user_input(user_query, provider)
    logger.info(
        "Intent parsed: intent=%s, product='%s', confidence=%.2f",
        parsed.intent, parsed.product_reference, parsed.confidence,
    )

    # Out-of-scope queries get a redirect response immediately.
    if parsed.intent == "out_of_scope":
        return InferenceResult(
            intent=parsed.intent,
            product_reference=None,
            product_match=None,
            recommendation_color=recommendation_color,
            response=OUT_OF_SCOPE_RESPONSE,
            guardrail_passed=True,
        )

    # --- Step 2: Resolve product via pgvector ---
    product_match: ProductMatch | None = None
    product_name = parsed.product_reference or "the product"
    product_price = float(financial_context.get("product_price", 0.0))

    if db_engine is not None and parsed.product_reference:
        product_match = resolve_product(parsed.product_reference, db_engine)
        if product_match:
            product_name = product_match.product_name
            product_price = product_match.price
            logger.info(
                "Product resolved: '%s' → '%s' ($%.2f)",
                parsed.product_reference, product_name, product_price,
            )
        else:
            logger.info("Product not found for reference: '%s'", parsed.product_reference)
            return InferenceResult(
                intent=parsed.intent,
                product_reference=parsed.product_reference,
                product_match=None,
                recommendation_color=recommendation_color,
                response=PRODUCT_NOT_FOUND_RESPONSE,
                guardrail_passed=True,
            )

    # --- Step 3: Build recommendation context ---
    context = RecommendationContext(
        product_name=product_name,
        product_price=product_price,
        recommendation_color=recommendation_color,
        original_color=original_color or recommendation_color,
        was_downgraded=was_downgraded,
        triggered_rules=triggered_rules,
        confidence_scores={recommendation_color: model_confidence},
        user_context=parsed.user_context,
        affordability_score=float(financial_context.get("affordability_score", 0.0)),
        savings_to_price_ratio=float(financial_context.get("savings_to_price_ratio", 0.0)),
        emergency_fund_months=float(financial_context.get("emergency_fund_months", 0.0)),
        debt_to_income_ratio=float(financial_context.get("debt_to_income_ratio", 0.0)),
    )

    # --- Step 4: Generate response ---
    response = generate_response(context, provider)

    # --- Step 5: Guardrail verification ---
    guardrail_result = check_response(response, recommendation_color, context)

    if not guardrail_result.passed:
        logger.warning(
            "Guardrail violations: %s — falling back to template",
            guardrail_result.violations,
        )
        from llm.response_generator import _generate_from_template
        response = _generate_from_template(context)
        guardrail_result = check_response(response, recommendation_color, context)

    return InferenceResult(
        intent=parsed.intent,
        product_reference=parsed.product_reference,
        product_match=product_match,
        recommendation_color=recommendation_color,
        response=response,
        guardrail_passed=guardrail_result.passed,
        guardrail_violations=guardrail_result.violations,
        was_downgraded=was_downgraded,
        triggered_rules=triggered_rules,
    )
