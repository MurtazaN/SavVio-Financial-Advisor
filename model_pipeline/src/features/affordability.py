"""Canonical affordability feature computation shared by training and inference."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Clamp pathological affordability / RUS values so the engine and UI are not driven by overflow.
_AFFORDABILITY_ABS_MAX = 10_000.0
_RUS_ABS_MAX = 1_000.0


def _obligation_floor_for_residual(monthly_income: float) -> float:
    """Minimum denominator for RUS so tiny monthly obligations do not explode the ratio."""
    inc = float(monthly_income or 0.0)
    return max(1e-6, 0.01 * max(abs(inc), 1.0))


AFFORDABILITY_COLUMNS = [
    "affordability_score",
    "price_to_income_ratio",
    "residual_utility_score",
    "savings_to_price_ratio",
    "net_worth_indicator",
    "credit_risk_indicator",
]


def compute_affordability_values(
    user_financial_profile: dict,
    product_price: float,
    cumulative_spend: float = 0.0,
) -> Dict[str, Optional[float]]:
    """Compute the six financial affordability features for one user-product pair."""
    income = user_financial_profile.get("monthly_income", 0.0)
    discretionary = user_financial_profile.get("discretionary_income", 0.0)
    savings = user_financial_profile.get("liquid_savings", 0.0)
    expenses = user_financial_profile.get("monthly_expenses", 0.0)
    emi = user_financial_profile.get("monthly_emi", 0.0)
    loan_amount = user_financial_profile.get("loan_amount", 0.0)
    credit_score = user_financial_profile.get("credit_score")

    # Spend prior approved purchases from discretionary income first, then savings.
    di_used = min(max(discretionary, 0.0), cumulative_spend)
    savings_used = cumulative_spend - di_used
    discretionary = discretionary - di_used
    savings = max(savings - savings_used, 0.0)

    total_obligations = expenses + emi

    affordability_raw = discretionary - product_price
    affordability_unreliable = abs(affordability_raw) > _AFFORDABILITY_ABS_MAX
    if affordability_unreliable:
        logger.warning(
            "affordability_score raw %.2f outside ±%.0f — clamping (check discretionary_income / price)",
            affordability_raw,
            _AFFORDABILITY_ABS_MAX,
        )
    affordability_score = round(affordability_raw, 2)
    if affordability_unreliable:
        affordability_score = float(
            max(-_AFFORDABILITY_ABS_MAX, min(_AFFORDABILITY_ABS_MAX, affordability_score))
        )

    price_to_income = round(product_price / income, 4) if income > 0 else None

    residual_utility = None
    if total_obligations > 0:
        denom = max(total_obligations, _obligation_floor_for_residual(income))
        rus = (savings - product_price) / denom
        if abs(rus) > _RUS_ABS_MAX:
            logger.warning(
                "residual_utility_score magnitude %.4f exceeds %.0f — clamping (obligations=%.4f, denom=%.4f)",
                rus,
                _RUS_ABS_MAX,
                total_obligations,
                denom,
            )
            rus = max(-_RUS_ABS_MAX, min(_RUS_ABS_MAX, rus))
        residual_utility = round(rus, 4)

    savings_to_price = round(savings / product_price, 4) if product_price > 0 else None
    net_worth = round((savings - loan_amount) / income, 4) if income > 0 else None
    credit_risk = round((credit_score - 299) / 550, 4) if credit_score is not None else None

    return {
        "affordability_score": affordability_score,
        "price_to_income_ratio": price_to_income,
        "residual_utility_score": residual_utility,
        "savings_to_price_ratio": savings_to_price,
        "net_worth_indicator": net_worth,
        "credit_risk_indicator": credit_risk,
        "affordability_score_unreliable": affordability_unreliable,
    }


def add_affordability_features(
    scenarios: pd.DataFrame,
    product_price_col: str = "product_price",
) -> pd.DataFrame:
    """Add canonical affordability columns to a scenarios DataFrame."""

    def _per_row(row: pd.Series) -> pd.Series:
        values = compute_affordability_values(
            {
                "monthly_income": row.get("monthly_income", 0.0),
                "discretionary_income": row.get("discretionary_income", 0.0),
                "liquid_savings": row.get("liquid_savings", 0.0),
                "monthly_expenses": row.get("monthly_expenses", 0.0),
                "monthly_emi": row.get("monthly_emi", 0.0),
                "loan_amount": row.get("loan_amount", 0.0),
                "credit_score": row.get("credit_score"),
            },
            product_price=float(row.get(product_price_col, 0.0) or 0.0),
            cumulative_spend=0.0,
        )
        return pd.Series({k: values[k] for k in AFFORDABILITY_COLUMNS})

    enriched = scenarios.copy()
    enriched[AFFORDABILITY_COLUMNS] = enriched.apply(_per_row, axis=1)
    return enriched
