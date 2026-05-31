"""
Affordability & Financial Health Feature Computation.

Transforms raw financial profiles and product data into 6 model-ready
financial features that capture a user's ability to absorb a purchase.

Feature groups:
    Financial (6) — Measures of user affordability and resilience.
    No product review features — the decision engine is purely financial.

NOTE: This is NOT a batch pipeline script.  It does not read/write files.
Financial features are pre-computed and stored in PostgreSQL by the
data pipeline (financial_features.py).

This module combines them ON DEMAND for a specific user-product pair.

For batch scenario generation (ML training label generation), see:
    features/training_data_generator.py

Usage (single pair — Decision API / Deterministic Engine):
    from features.financial_features import compute_affordability

    result = compute_affordability(
        user_financial_profile={...},
        product_price=799.99,
    )
"""

import logging
from dataclasses import dataclass
from typing import Optional

from features.affordability import compute_affordability_values

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference-time feature computation
# ---------------------------------------------------------------------------

@dataclass
class AffordabilityResult:
    """Container for the 6 computed financial features returned by compute_affordability()."""

    # Financial features — how well can this user absorb the purchase?
    affordability_score: Optional[float]
    price_to_income_ratio: Optional[float]
    residual_utility_score: Optional[float]
    savings_to_price_ratio: Optional[float]
    net_worth_indicator: Optional[float]
    credit_risk_indicator: Optional[float]
    affordability_score_unreliable: bool = False

    def to_dict(self) -> dict:
        return {
            "affordability_score": self.affordability_score,
            "price_to_income_ratio": self.price_to_income_ratio,
            "residual_utility_score": self.residual_utility_score,
            "savings_to_price_ratio": self.savings_to_price_ratio,
            "net_worth_indicator": self.net_worth_indicator,
            "credit_risk_indicator": self.credit_risk_indicator,
            "affordability_score_unreliable": self.affordability_score_unreliable,
        }


def compute_affordability(
    user_financial_profile: dict,
    product_price: float,
    product_info: Optional[dict] = None,
    cumulative_spend: float = 0.0,
) -> AffordabilityResult:
    """
    Computes 6 financial features for a single user-product pair.

    Called at inference time by the Deterministic Financial Logic Engine
    when a user queries a specific product.

    When a user evaluates multiple products in a session, pass the total
    price of all previously approved purchases as ``cumulative_spend``
    so that savings and discretionary income reflect what the user
    actually has left.

    Args:
        user_financial_profile: Dict with pre-computed financial fields from DB.
        product_price: Price of the product being evaluated.
        product_info:  (Unused — kept for API compatibility.)
        cumulative_spend: Sum of prices for prior purchases in this
            session. Defaults to 0 (first / standalone purchase).

    Returns:
        AffordabilityResult with 6 financial features.
    """
    values = compute_affordability_values(
        user_financial_profile=user_financial_profile,
        product_price=product_price,
        cumulative_spend=cumulative_spend,
    )

    result = AffordabilityResult(
        affordability_score=values["affordability_score"],
        price_to_income_ratio=values["price_to_income_ratio"],
        residual_utility_score=values["residual_utility_score"],
        savings_to_price_ratio=values["savings_to_price_ratio"],
        net_worth_indicator=values["net_worth_indicator"],
        credit_risk_indicator=values["credit_risk_indicator"],
        affordability_score_unreliable=bool(values.get("affordability_score_unreliable", False)),
    )

    logger.info(
        "Affordability computed — price: %.2f, score: %.2f, RUS: %s",
        product_price,
        result.affordability_score,
        result.residual_utility_score,
    )

    return result
