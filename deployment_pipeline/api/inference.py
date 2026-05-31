"""
Inference Orchestrator — Synchronous prediction pipeline stages:
1. DB Profile Loading
2. Intent Parsing / Product Matching
3. Product Metadata & Review Retrieval
4. ML Scoring (Layer 1)
5. Quality Downgrade Engine (Layer 2)
6. Response Generation & Guardrails
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import text

from deployment_pipeline.api.config import APIConfig
from deployment_pipeline.api.model_loader import _ensure_import_paths

_ensure_import_paths()

from llm.intent_parser import parse_user_input
from llm.product_resolver import resolve_product, resolve_product_by_id
from llm.response_generator import RecommendationContext, generate_response, _generate_from_template
from llm.guardrails import check_response
from llm.prompts.response_templates import (
    OUT_OF_SCOPE_RESPONSE,
    PRODUCT_NOT_FOUND_RESPONSE,
    USER_NOT_FOUND_RESPONSE,
)
from features.product_features import (
    ProductFeatures,
    compute_product_features,
    compute_category_stats,
)
from features.review_features import ReviewFeatures, compute_review_features
from deterministic_engine.downgrade_engine import DowngradeEngine

from deployment_pipeline.api.metrics import (
    record_inference,
    record_guardrail_failure,
    record_layer2_downgrade,
    observe_ml_confidence,
    observe_llm_latency,
)
from deployment_pipeline.api.schemas import (
    FinancialFeaturesView,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result Types
# ---------------------------------------------------------------------------

@dataclass
class ProductResolution:
    product_name: str | None = None
    product_price: float | None = None
    product_id: str | None = None
    evaluation_mode: str = "catalog"
    user_context: str | None = None
    parsed_intent: Any = None


@dataclass
class MLScore:
    confidence: float | None = None
    predicted_label: str | None = None
    unavailable_reason: str | None = None
    # Financial features returned by model for API response / LLM
    affordability_score: float = 0.0
    affordability_unreliable: bool = False
    savings_to_price_ratio: float = 0.0
    price_to_income_ratio: float = 0.0
    residual_utility_score: float = 0.0
    net_worth_indicator: float = 0.0
    credit_risk_indicator: float = 0.0


# ---------------------------------------------------------------------------
# Stage 1: Load User Financial Profile
# ---------------------------------------------------------------------------

def _load_user_financial_profile(user_id: str, db_engine) -> dict | None:
    sql = text("""
        SELECT user_id, monthly_income, monthly_expenses, savings_balance,
               has_loan, loan_amount, monthly_emi, loan_interest_rate,
               loan_term_months, credit_score, employment_status, region,
               liquid_savings, discretionary_income, debt_to_income_ratio,
               saving_to_income_ratio, monthly_expense_burden_ratio,
               emergency_fund_months
        FROM financial_profiles
        WHERE user_id = :user_id
    """)
    try:
        with db_engine.connect() as conn:
            row = conn.execute(sql, {"user_id": user_id}).fetchone()
    except Exception as e:
        logger.error("Failed to load financial profile for user_id=%s: %s", user_id, e)
        return None

    if row is None:
        return None

    columns = [
        "user_id", "monthly_income", "monthly_expenses", "savings_balance",
        "has_loan", "loan_amount", "monthly_emi", "loan_interest_rate",
        "loan_term_months", "credit_score", "employment_status", "region",
        "liquid_savings", "discretionary_income", "debt_to_income_ratio",
        "saving_to_income_ratio", "monthly_expense_burden_ratio",
        "emergency_fund_months",
    ]
    return dict(zip(columns, row))


# ---------------------------------------------------------------------------
# Stage 2: Resolve Product
# ---------------------------------------------------------------------------

def _resolve_product(request: PredictRequest, manager) -> ProductResolution | PredictResponse:
    if request.product_id:
        match = resolve_product_by_id(request.product_id, manager.db_engine)
        if match:
            return ProductResolution(
                product_name=match.product_name, product_price=match.price,
                product_id=match.product_id,
            )
        return PredictResponse(
            recommendation="YELLOW", confidence=None,
            explanation=PRODUCT_NOT_FOUND_RESPONSE, evaluation_mode="none",
        )

    _t0 = time.time()
    parsed = parse_user_input(request.user_query, manager.llm_provider)
    observe_llm_latency(
        provider=getattr(manager.llm_provider, "provider_name", "unknown"),
        operation="intent_parse",
        latency=time.time() - _t0,
    )

    if parsed.intent == "out_of_scope":
        return PredictResponse(
            recommendation="YELLOW", confidence=None,
            explanation=OUT_OF_SCOPE_RESPONSE, evaluation_mode="none",
        )

    price_hint = getattr(parsed, "price_hint", None)
    user_ctx = getattr(parsed, "user_context", None)
    match = None

    if parsed.product_reference:
        match = resolve_product(parsed.product_reference, manager.db_engine, price_hint=price_hint)

    if match:
        return ProductResolution(
            product_name=match.product_name, product_price=match.price,
            product_id=match.product_id, user_context=user_ctx, parsed_intent=parsed,
        )

    if price_hint is not None and parsed.product_reference:
        return ProductResolution(
            product_name=parsed.product_reference.strip(), product_price=float(price_hint),
            evaluation_mode="hypothetical", user_context=user_ctx, parsed_intent=parsed,
        )

    return PredictResponse(
        recommendation="YELLOW", confidence=None,
        explanation=PRODUCT_NOT_FOUND_RESPONSE, evaluation_mode="none",
    )


# ---------------------------------------------------------------------------
# Stage 3: Load Product Data
# ---------------------------------------------------------------------------

def _format_review_snippets(reviews_df) -> list:
    """Return up to 5 formatted review snippets sorted by helpfulness."""
    if reviews_df.empty:
        return []
    top = reviews_df.sort_values("helpful_vote", ascending=False).head(5)
    snippets = []
    for _, r in top.iterrows():
        rating = int(r.get("rating") or 0)
        title = str(r.get("review_title") or "").strip()
        text = str(r.get("review_text") or "").strip()
        snippet = f"[{rating}\u2605]"
        if title:
            snippet += f" {title}"
        if text:
            snippet += f": {text[:120]}{'...' if len(text) > 120 else ''}"
        if len(snippet) > 5:
            snippets.append(snippet)
    return snippets


def _load_product_data(product: ProductResolution, manager) -> tuple[dict, list]:
    """Retrieves product specs and formatted review snippets for scoring and generation."""
    empty = {
        "average_rating": 0, "rating_number": 0, "rating_variance": 0, "category": "unknown",
        "value_density": 0, "review_confidence": 0, "rating_polarization": 0,
        "quality_risk_score": 0, "cold_start_flag": 0, "price_category_rank": 0,
        "category_rating_deviation": 0, "verified_purchase_ratio": 0,
        "helpful_concentration": 0, "sentiment_spread": 0, "review_depth_score": 0,
        "reviewer_diversity": 0, "extreme_rating_ratio": 0,
    }

    if not product.product_id or not manager.db_engine:
        return empty, []

    # Load product row
    try:
        with manager.db_engine.connect() as conn:
            row = conn.execute(text(
                "SELECT product_id, product_name, price, average_rating, "
                "rating_number, rating_variance, category FROM products WHERE product_id = :pid"
            ), {"pid": product.product_id}).fetchone()
    except Exception as e:
        logger.error("Failed to load product %s: %s", product.product_id, e)
        return empty, []

    if row is None:
        return empty, []

    cols = ["product_id", "product_name", "price", "average_rating", "rating_number", "rating_variance", "category"]
    product_row = pd.Series(dict(zip(cols, row)))

    # Load reviews
    try:
        with manager.db_engine.connect() as conn:
            review_rows = conn.execute(text(
                "SELECT user_id, product_id, rating, review_title, review_text, "
                "verified_purchase, helpful_vote FROM reviews WHERE product_id = :pid"
            ), {"pid": product.product_id}).fetchall()
    except Exception as e:
        logger.error("Failed to load reviews for %s: %s", product.product_id, e)
        review_rows = []

    review_cols = ["user_id", "product_id", "rating", "review_title", "review_text", "verified_purchase", "helpful_vote"]
    reviews_df = pd.DataFrame(review_rows, columns=review_cols) if review_rows else pd.DataFrame()

    # Compute product + review features
    products_df = pd.read_sql("SELECT product_id, price, average_rating, rating_number, rating_variance, category FROM products", manager.db_engine)
    cat_stats = compute_category_stats(products_df)
    max_rn = float(products_df["rating_number"].max() or 0)

    pf = compute_product_features(product_row, cat_stats, max_rn)
    rf = compute_review_features(reviews_df)
    snippets = _format_review_snippets(reviews_df)

    return {
        "average_rating": float(product_row["average_rating"] or 0),
        "rating_number": float(product_row["rating_number"] or 0),
        "rating_variance": float(product_row["rating_variance"] or 0),
        "category": str(product_row["category"] or "unknown"),
        **pf.to_dict(), **rf.to_dict(),
    }, snippets


# ---------------------------------------------------------------------------
# Stage 4: ML Model Scoring (AUTHORITY)
# ---------------------------------------------------------------------------

def _score_ml_model(user_profile: dict, product: ProductResolution, product_data: dict, manager) -> MLScore:
    if manager.model is None:
        return MLScore(unavailable_reason="no_model")

    try:
        raw_row = {**user_profile, "product_price": float(product.product_price or 0), **product_data}
        result = manager.predict(pd.DataFrame([raw_row]))

        r = result.iloc[0]
        return MLScore(
            predicted_label=str(r["label"]).strip().upper(),
            confidence=float(r["confidence"]),
            affordability_score=float(r.get("affordability_score", 0)),
            affordability_unreliable=bool(r.get("affordability_score_unreliable", False)),
            savings_to_price_ratio=float(r.get("savings_to_price_ratio", 0)),
            price_to_income_ratio=float(r.get("price_to_income_ratio", 0)),
            residual_utility_score=float(r.get("residual_utility_score", 0)),
            net_worth_indicator=float(r.get("net_worth_indicator", 0)),
            credit_risk_indicator=float(r.get("credit_risk_indicator", 0)),
        )

    except Exception as e:
        logger.warning("ML model scoring failed: %s", e, exc_info=True)
        return MLScore(unavailable_reason="scoring_error")


# ---------------------------------------------------------------------------
# Stage 4b: Layer 2 Downgrade (product/review quality signals)
# ---------------------------------------------------------------------------

def _apply_layer2_downgrade(ml_label: str, product_data: dict):
    """Evaluates product/review quality signals to determine if a secondary downgrade is required."""
    pf = ProductFeatures(
        value_density=float(product_data.get("value_density", 0)),
        review_confidence=float(product_data.get("review_confidence", 0)),
        rating_polarization=float(product_data.get("rating_polarization", 0)),
        quality_risk_score=float(product_data.get("quality_risk_score", 0)),
        cold_start_flag=int(product_data.get("cold_start_flag", 0)),
        price_category_rank=float(product_data.get("price_category_rank", 0)),
        category_rating_deviation=float(product_data.get("category_rating_deviation", 0)),
    )
    rf = ReviewFeatures(
        verified_purchase_ratio=float(product_data.get("verified_purchase_ratio", 0)),
        helpful_concentration=float(product_data.get("helpful_concentration", 0)),
        sentiment_spread=float(product_data.get("sentiment_spread", 0)),
        review_depth_score=float(product_data.get("review_depth_score", 0)),
        reviewer_diversity=float(product_data.get("reviewer_diversity", 0)),
        extreme_rating_ratio=float(product_data.get("extreme_rating_ratio", 0)),
    )
    return DowngradeEngine().evaluate(ml_label, pf, rf)


# ---------------------------------------------------------------------------
# Stage 5: LLM Response Generation + Guardrails
# ---------------------------------------------------------------------------

def _generate_explanation(product: ProductResolution, ml: MLScore, user_profile: dict, manager, downgrade_result=None, review_snippets: list = None) -> str:
    """Invokes LLM for natural language reasoning and validates output against decision color."""
    final_color = (downgrade_result.final_label if downgrade_result else None) or ml.predicted_label or "YELLOW"
    original_color = (downgrade_result.original_label if downgrade_result else None) or final_color
    was_downgraded = downgrade_result.was_downgraded if downgrade_result else False
    triggered_rules = (
        (downgrade_result.product_triggers + downgrade_result.review_triggers)
        if downgrade_result and was_downgraded
        else []
    )

    context = RecommendationContext(
        product_name=product.product_name or "the product",
        product_price=product.product_price or 0.0,
        recommendation_color=final_color,
        original_color=original_color,
        was_downgraded=was_downgraded,
        triggered_rules=triggered_rules,
        confidence_scores={final_color: ml.confidence or 0.0},
        ml_confidence=ml.confidence,
        ml_unavailable_reason=ml.unavailable_reason,
        hypothetical_purchase=(product.evaluation_mode == "hypothetical"),
        user_context=product.user_context,
        affordability_score=ml.affordability_score,
        affordability_score_unreliable=ml.affordability_unreliable,
        savings_to_price_ratio=ml.savings_to_price_ratio,
        emergency_fund_months=float(user_profile.get("emergency_fund_months", 0) or 0),
        debt_to_income_ratio=float(user_profile.get("debt_to_income_ratio", 0) or 0),
        monthly_income=float(user_profile.get("monthly_income", 0) or 0),
        monthly_expense_burden_ratio=float(user_profile.get("monthly_expense_burden_ratio", 0) or 0),
        price_to_income_ratio=ml.price_to_income_ratio,
        credit_score=user_profile.get("credit_score"),
        review_snippets=review_snippets or [],
    )

    _t0 = time.time()
    response_text = generate_response(context, manager.llm_provider)
    observe_llm_latency(
        provider=getattr(manager.llm_provider, "provider_name", "unknown"),
        operation="response_gen",
        latency=time.time() - _t0,
    )
    guardrail = check_response(response_text, final_color, context)

    if not guardrail.passed:
        record_guardrail_failure()
        logger.warning("Guardrail violations: %s — falling back to template", guardrail.violations)
        response_text = _generate_from_template(context)

    return response_text


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_inference(request: PredictRequest, manager) -> PredictResponse:
    start_time = time.time()

    if manager.db_engine is None:
        return PredictResponse(
            recommendation="YELLOW", confidence=None,
            explanation="Service is temporarily unavailable.", evaluation_mode="none",
        )

    user_profile = _load_user_financial_profile(request.user_id, manager.db_engine)
    if user_profile is None:
        return PredictResponse(
            recommendation="YELLOW", confidence=None,
            explanation=USER_NOT_FOUND_RESPONSE, evaluation_mode="none",
        )

    resolution = _resolve_product(request, manager)
    if isinstance(resolution, PredictResponse):
        return resolution

    product_data, review_snippets = _load_product_data(resolution, manager)

    ml = _score_ml_model(user_profile, resolution, product_data, manager)

    # Layer 2 Quality Check (Down-ranking for problematic catalog items)
    downgrade_result = None
    if ml.predicted_label and resolution.product_id:
        try:
            downgrade_result = _apply_layer2_downgrade(ml.predicted_label, product_data)
            if downgrade_result.was_downgraded:
                record_layer2_downgrade(
                    from_color=downgrade_result.original_label,
                    to_color=downgrade_result.final_label,
                )
                logger.info(
                    "Layer 2 downgrade: %s → %s (triggers: %s)",
                    downgrade_result.original_label,
                    downgrade_result.final_label,
                    downgrade_result.product_triggers + downgrade_result.review_triggers,
                )
        except Exception as e:
            logger.warning("Layer 2 downgrade engine failed: %s", e)

    final_label = (downgrade_result.final_label if downgrade_result else None) or ml.predicted_label or "YELLOW"

    explanation = _generate_explanation(resolution, ml, user_profile, manager, downgrade_result, review_snippets)

    elapsed = time.time() - start_time
    logger.info("Inference complete: %s (%.3fs)", final_label, elapsed)

    # ------------------------------------------------------------------
    # Prometheus Metrics Recording
    # ------------------------------------------------------------------
    record_inference(
        color=final_label,
        evaluation_mode=resolution.evaluation_mode,
        latency=elapsed,
    )
    if ml.confidence is not None:
        observe_ml_confidence(ml.confidence)

    return PredictResponse(
        recommendation=final_label,
        confidence=ml.confidence,
        ml_unavailable_reason=ml.unavailable_reason,
        explanation=explanation,
        product_name=resolution.product_name,
        product_price=resolution.product_price,
        evaluation_mode=resolution.evaluation_mode,
        affordability_score=ml.affordability_score,
        affordability_score_unreliable=ml.affordability_unreliable,
        emergency_fund_months=float(user_profile.get("emergency_fund_months", 0) or 0),
        debt_to_income_ratio=float(user_profile.get("debt_to_income_ratio", 0) or 0),
        financial_features=FinancialFeaturesView(
            discretionary_income=float(user_profile.get("discretionary_income", 0) or 0),
            debt_to_income_ratio=float(user_profile.get("debt_to_income_ratio", 0) or 0),
            saving_to_income_ratio=float(user_profile.get("saving_to_income_ratio", 0) or 0),
            monthly_expense_burden_ratio=float(user_profile.get("monthly_expense_burden_ratio", 0) or 0),
            emergency_fund_months=float(user_profile.get("emergency_fund_months", 0) or 0),
            affordability_score=ml.affordability_score,
            price_to_income_ratio=ml.price_to_income_ratio,
            residual_utility_score=ml.residual_utility_score,
            savings_to_price_ratio=ml.savings_to_price_ratio,
            net_worth_indicator=ml.net_worth_indicator,
            credit_risk_indicator=ml.credit_risk_indicator,
        ),
        ml_predicted_label=ml.predicted_label,
        ml_model_name=APIConfig.ML_MODEL_DISPLAY_NAME,
    )
