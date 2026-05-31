"""
Generate natural-language explanations aligned with deterministic engine output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm.prompts.response_templates import COLOR_INTROS

logger = logging.getLogger(__name__)


@dataclass
class RecommendationContext:
    product_name: str
    product_price: float
    recommendation_color: str  # GREEN / YELLOW / RED
    original_color: str
    was_downgraded: bool
    triggered_rules: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    ml_confidence: Optional[float] = None  # None => classifier not used
    # When ml_confidence is None: why (clearer than "not loaded" for every case)
    ml_unavailable_reason: Optional[str] = None  # "no_model" | "no_pipeline" | "scoring_error"
    hypothetical_purchase: bool = False
    user_context: Optional[str] = None
    affordability_score: float = 0.0
    affordability_score_unreliable: bool = False
    savings_to_price_ratio: float = 0.0
    emergency_fund_months: float = 0.0
    debt_to_income_ratio: float = 0.0
    # Additional profile fields for richer LLM explanations
    monthly_income: float = 0.0
    monthly_expense_burden_ratio: float = 0.0  # MEB — expenses as % of income
    price_to_income_ratio: float = 0.0          # PIR — price as % of monthly income
    credit_score: Optional[int] = None          # Raw FICO score (300–850)
    # Up to 5 formatted review snippets — e.g. '[4★] Great sound: battery lasts forever...'
    # Empty list when product has no reviews or is a hypothetical evaluation.
    review_snippets: List[str] = field(default_factory=list)


def _color_key(color: str) -> str:
    c = (color or "YELLOW").upper()
    if c not in ("GREEN", "YELLOW", "RED"):
        return "YELLOW"
    return c


def _generate_from_template(context: RecommendationContext) -> str:
    """Non-LLM fallback — still aligned with signal."""
    ck = _color_key(context.recommendation_color)
    intro_tpl = COLOR_INTROS.get(ck, COLOR_INTROS["YELLOW"])
    intro = intro_tpl.format(
        product_name=context.product_name,
        product_price=context.product_price,
    )

    lines = [intro, ""]
    if context.hypothetical_purchase:
        lines.append(
            "- **Note:** No catalog match — evaluating affordability at **your stated price** "
            f"for **{context.product_name}**."
        )
    if context.ml_confidence is not None:
        # Two decimals: sub-1% scores (common on uncertain rows) otherwise show as 0.0%.
        lines.append(
            f"- **Signal:** {context.recommendation_color} "
            f"(ML confidence: {context.ml_confidence:.2%})."
        )
        if context.hypothetical_purchase and context.ml_confidence < 0.15:
            lines.append(
                "- **Note:** With **no catalog product row**, review/product features are neutralized for ML; "
                "the classifier probability can be **low or flat** even when the rules engine says GREEN."
            )
    else:
        reason = context.ml_unavailable_reason or ""
        if reason == "no_model":
            tail = (
                "no **trained model artifact** found (set `MODEL_ARTIFACT_DIR` or run the training pipeline). "
                "Recommendation is from the **rules engine**."
            )
        elif reason == "no_pipeline":
            tail = (
                "**Feature pipeline** (`FEATURE_PIPELINE_PATH` / `feature_pipeline.pkl`) missing — ML score skipped. "
                "Recommendation is from the **rules engine**."
            )
        elif reason == "scoring_error":
            tail = "ML scoring failed (see API logs); recommendation is from the **rules engine**."
        else:
            tail = (
                "ML score unavailable — recommendation is from the **deterministic rules engine**."
            )
        lines.append(f"- **Signal:** {context.recommendation_color} — {tail}")
    if context.affordability_score_unreliable:
        lines.append(
            "- **Affordability score:** calculation error (inputs out of safe range); "
            f"**Savings-to-price ratio:** {context.savings_to_price_ratio:.2f}."
        )
    else:
        lines.append(
            f"- **Affordability score:** {context.affordability_score:.2f}; "
            f"**Savings-to-price ratio:** {context.savings_to_price_ratio:.2f}."
        )
    lines.append(
        f"- **Emergency fund:** ~{context.emergency_fund_months:.1f} months; "
        f"**Debt-to-income:** {context.debt_to_income_ratio:.2%}."
    )
    if context.triggered_rules:
        lines.append("- **Rules considered:** " + "; ".join(context.triggered_rules[:8]))
    if context.was_downgraded:
        lines.append(
            "- **Note:** Product and review signals caused a **one-step downgrade** from the pure financial assessment."
        )
    if context.review_snippets:
        lines.append("")
        lines.append("**What customers say:**")
        for snippet in context.review_snippets:
            lines.append(f"  - {snippet}")
    if context.user_context:
        lines.append(f"- **Your note:** {context.user_context}")

    lines.append("")
    if ck == "GREEN":
        lines.append(
            "If this remains within your monthly plan, you can proceed — still track discretionary spend."
        )
    elif ck == "YELLOW":
        lines.append(
            "Consider waiting a pay cycle, comparing alternatives, or reducing other discretionary spend first."
        )
    else:
        lines.append(
            "Prioritize essentials, debt minimums, and emergency savings before this purchase."
        )

    return "\n".join(lines)


def generate_response(context: RecommendationContext, llm_provider: Any) -> str:
    """
    Ask the LLM to produce an explanation that respects the authoritative color.

    On failure, uses _generate_from_template.
    """
    ck = _color_key(context.recommendation_color)
    rules_txt = "; ".join(context.triggered_rules[:12]) if context.triggered_rules else "(none listed)"

    if context.review_snippets:
        reviews_block = "Customer voice (most helpful reviews):\n" + "\n".join(
            f"  {s}" for s in context.review_snippets
        )
        reviews_instruction = (
            "- Weave 1–2 specific details from the customer reviews into your explanation "
            "so the user understands what real buyers experienced with this product. "
            "If reviews are positive, note what customers praise. "
            "If reviews are mixed or negative, flag the concerns honestly."
        )
    else:
        reviews_block = "Customer voice: no reviews available for this product."
        reviews_instruction = "- Do not fabricate customer opinions; note that no reviews are available."

    afs_line = (
        "Affordability score: [calculation error — do not quote]"
        if context.affordability_score_unreliable
        else f"Affordability score (monthly discretionary minus price): {context.affordability_score:+.2f}"
    )
    dti_pct = context.debt_to_income_ratio * 100
    meb_pct = context.monthly_expense_burden_ratio * 100
    pir_pct = context.price_to_income_ratio * 100
    credit_line = (
        f"Credit score: {context.credit_score} (300–850 scale; below 580 = poor, 580–669 = fair, 670+ = good)"
        if context.credit_score is not None
        else "Credit score: not available"
    )

    downgrade_note = (
        f"\nDOWNGRADE: The ML model initially scored this as {context.original_color}, "
        f"but product/review quality signals downgraded it to {context.recommendation_color}. "
        f"Briefly acknowledge this — e.g. 'While your finances could support this purchase, "
        f"product and review signals raised enough concern to warrant caution.'"
        if context.was_downgraded else ""
    )

    prompt = f"""You are SavVio, a fiduciary financial advisor. Your job is to explain a purchase recommendation clearly, using the exact numbers provided.

DECISION: {context.recommendation_color} — do NOT contradict this.{downgrade_note}
{"Do NOT suggest buying, going ahead, or that this is safe." if context.recommendation_color == "RED" else ""}
{"Do NOT tell the user to avoid the purchase or that it is risky." if context.recommendation_color == "GREEN" else ""}

PRODUCT: {context.product_name}
PRICE: ${context.product_price:,.2f}{"  (hypothetical — no catalog match, price is user-stated)" if context.hypothetical_purchase else ""}

FINANCIAL PROFILE (use these exact values — never invent numbers):
- Monthly income: ${context.monthly_income:,.0f}
- This purchase = {pir_pct:.0f}% of monthly income
- Monthly expense burden: {meb_pct:.0f}% of income already committed to expenses
- {afs_line}
- Savings cover price: {context.savings_to_price_ratio:.1f}x
- Emergency fund: {context.emergency_fund_months:.1f} months (target: 3–6 months)
- Debt-to-income: {dti_pct:.0f}%
- {credit_line}
- Rules that triggered: {rules_txt}
{f"- User added context: {context.user_context}" if context.user_context else ""}

{reviews_block}

Write exactly 4 parts in order, as flowing prose. Do NOT include any numbers, labels, headers, or section names — just the text itself:

Part 1 (1–2 sentences): State the {context.recommendation_color} verdict and the single strongest reason, using at least one specific number.
Part 2 (2–3 sentences): Explain what 2–3 of the financial numbers above mean for this user in plain English. Make it feel personal — e.g. "On a $1,110 income, this $600 purchase would consume more than half your monthly earnings."
Part 3 (1–2 sentences): {"Reference what real buyers say — quote or paraphrase a specific detail from the customer reviews above." if context.review_snippets else "Note that no customer reviews are available for this product."}
Part 4 (1 sentence): One concrete next step the user can take right now.

Separate each part with a blank line. Use **bold** for key numbers. Write warmly but honestly. Do not use bullet points. Do not start any line with a number or label."""

    try:
        raw = llm_provider.generate(prompt, max_tokens=700, temperature=0.35)
        if raw and len(raw.strip()) > 20:
            return raw.strip()
    except Exception as e:
        logger.warning("LLM response generation failed: %s", e)

    return _generate_from_template(context)
