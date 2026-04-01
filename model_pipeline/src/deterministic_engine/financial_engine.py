"""
Deterministic Decision Engine for SavVio.

Rule-based labeling system that assigns GREEN / YELLOW / RED to
(user, product) pairs. This engine is AUTHORITATIVE — neither the
ML model nor the LLM wrapper may override its output.

The engine answers: "Should this user buy this product?"

    GREEN  — Purchase fits comfortably within the user's financial life.
    YELLOW — Genuine concern — the user should pause and think.
    RED    — Purchase would cause financial harm.

Design principle:
    Every rule combines signals from at least TWO independent correlation
    groups to avoid false triggers from a single underlying cause.

Correlation groups:
    Group 1 — Income capacity:   affordability_score, discretionary_income,
                                  price_to_income_ratio
    Group 2 — Savings depth:     saving_to_income_ratio, savings_to_price_ratio,
                                  emergency_fund_months, residual_utility_score
    Group 3 — Debt burden:       debt_to_income_ratio, monthly_expense_burden_ratio
    Group 4 — Independent:       credit_risk_indicator, net_worth_indicator

Features used — 11 total (5 DB + 6 computed), ALL financial:
    DB:       discretionary_income, debt_to_income_ratio, saving_to_income_ratio,
              monthly_expense_burden_ratio, emergency_fund_months
    Computed: affordability_score, price_to_income_ratio, residual_utility_score,
              savings_to_price_ratio, net_worth_indicator, credit_risk_indicator

Usage (batch — label generation for ML training):
    from deterministic_engine.financial_engine import DecisionEngine
    engine = DecisionEngine()
    df["label"] = df.apply(engine.decide_row, axis=1)

Usage (single pair — Decision API):
    result = engine.decide(financial_dict, product_dict)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    """Outcome of the deterministic engine for one (user, product) pair."""
    decision_category: str
    triggered_rules: List[str] = field(default_factory=list)
    confidence_downgrades: List[str] = field(default_factory=list)
    explanation: str = ""


class DecisionEngine:
    """
    Pure-financial multi-condition labeling engine.

    RED:    4 compound AND rules — each crosses at least 2 correlation groups
            and includes a price_to_income_ratio escape so trivial purchases
            never trigger RED.
    YELLOW: 5 compound AND rules — each crosses at least 2 groups.
            YELLOW triggers when >= 2 rules trigger.
    GREEN:  default when no RED triggers and fewer than 2 YELLOW rules trigger.
    """

    @staticmethod
    def _safe(val, feature_name: str):
        """Raise ValueError if val is None or NaN."""
        if val is None:
            raise ValueError(f"Missing required financial feature: {feature_name}")
        try:
            if math.isnan(val):
                raise ValueError(f"NaN passed for financial feature: {feature_name}")
        except TypeError:
            pass
        return val

    def decide(self, financial: dict, product: dict) -> DecisionResult:
        """
        Core labeling logic.

        Evaluates a user-product pair and returns GREEN / YELLOW / RED
        with the rules that triggered the decision.

        Args:
            financial: Dict with per-pair computed features + per-entity DB features.
            product:   Dict with product price (only price is used by the engine).

        Returns:
            DecisionResult with decision_category, triggered rules, and explanation.
        """
        # --- Per-pair computed features (from financial_features.py) ---
        affordability = self._safe(financial.get("affordability_score"), "affordability_score")
        pir = self._safe(financial.get("price_to_income_ratio"), "price_to_income_ratio")
        rus = self._safe(financial.get("residual_utility_score"), "residual_utility_score")
        spr = self._safe(financial.get("savings_to_price_ratio"), "savings_to_price_ratio")
        nwi = self._safe(financial.get("net_worth_indicator"), "net_worth_indicator")
        credit = self._safe(financial.get("credit_risk_indicator"), "credit_risk_indicator")

        # --- Per-entity DB features (from financial_features.py) ---
        dti = self._safe(financial.get("debt_to_income_ratio"), "debt_to_income_ratio")
        stir = self._safe(financial.get("saving_to_income_ratio"), "saving_to_income_ratio")
        meb = self._safe(financial.get("monthly_expense_burden_ratio"), "monthly_expense_burden_ratio")
        efm = self._safe(financial.get("emergency_fund_months"), "emergency_fund_months")

        # Evaluate RED Rules
        red_result = self._evaluate_red_rules(affordability, pir, rus, spr, nwi, meb, efm, dti)
        if red_result:
            return red_result

        # Evaluate YELLOW Rules
        yellow_result = self._evaluate_yellow_rules(affordability, pir, rus, spr, nwi, credit, dti, stir, efm, meb)
        if yellow_result:
            return yellow_result

        # GREEN: Purchase fits comfortably within the user's finances.
        return DecisionResult("GREEN", ["green:all_clear"], [],
            explanation="Affordable purchase with healthy financial position.")

    def _evaluate_red_rules(self, affordability, pir, rus, spr, nwi, meb, efm, dti) -> Optional[DecisionResult]:
        """
        Evaluate RED rules: Purchase would cause financial harm.
        All conditions within each rule must be true (AND logic).
        Every rule crosses at least 2 independent groups and includes
        a PIR escape hatch so trivial purchases never trigger RED.
        """
        triggered = []

        # RED Rule 1 — Groups 1 + 2 
        if affordability < 0 and spr < 1.5 and pir > 0.10:
            triggered.append("red:cant_afford_from_any_angle")
            return DecisionResult("RED", triggered,
                explanation="Income can't handle it, savings barely cover the price, and the purchase is non-trivial.")

        # RED Rule 2 — Groups 3 + 1 + 2
        if meb > 0.80 and pir > 0.20 and efm < 3.0:
            triggered.append("red:maxed_budget_significant_purchase")
            return DecisionResult("RED", triggered,
                explanation="Budget over 80% consumed, product is 20%+ of income, and thin emergency fund.")

        # RED Rule 3 — Groups 4 + 1 + 2
        if nwi < -2.0 and affordability < 0 and pir > 0.15 and efm < 3.0:
            triggered.append("red:underwater_no_surplus_significant")
            return DecisionResult("RED", triggered,
                explanation="Net worth negative, no surplus income, and a significant purchase.")

        # RED Rule 4 — Groups 2 + 3 + price awareness
        if efm < 1.0 and dti > 0.30 and pir > 0.10:
            triggered.append("red:paycheck_to_paycheck_wipes_net")
            return DecisionResult("RED", triggered,
                explanation="Less than 1 month emergency fund, DTI over 30%, and purchase is non-trivial.")

        return None

    def _evaluate_yellow_rules(self, affordability, pir, rus, spr, nwi, credit, dti, stir, efm, meb) -> Optional[DecisionResult]:
        """
        Evaluate YELLOW rules: Genuine concern — user should pause and think.
        YELLOW now triggers when >= 1 rule triggers.
        """
        yellow_triggers = []
        explanations = []

        # YELLOW Rule 1 — Groups 1 + 2
        if affordability < 0 and pir > 0.25 and spr < 5.0:
            yellow_triggers.append("yellow:income_pressure")
            explanations.append("Income doesn't cover purchase directly and savings cushion is moderate.")

        # YELLOW Rule 2 — Groups 2 + 1
        if spr < 5.0 and pir > 0.10:
            yellow_triggers.append("yellow:savings_strain")
            explanations.append("Savings can't comfortably absorb the purchase.")

        # YELLOW Rule 3 — Groups 3 + 1 + 2 
        if meb > 0.70 and pir > 0.10 and spr < 5.0:
            yellow_triggers.append("yellow:debt_stress")
            explanations.append("High debt obligations coupled with significant purchase and thin savings.")

        # YELLOW Rule 4 — Groups 2 + 1
        if efm < 3.0 and affordability < 0:
            yellow_triggers.append("yellow:low_resilience")
            explanations.append("Low emergency funds for an unaffordable immediate expense.")

        # YELLOW Rule 5 — Groups 4 + 1 + 2
        if credit < 0.35 and nwi < 1.0 and pir > 0.15 and spr < 5.0:
            yellow_triggers.append("yellow:weak_profile")
            explanations.append("Weak overall financial profile for a significant purchase.")

        # YELLOW triggers when at least one rule triggers.
        if len(yellow_triggers) >= 1:
            combined_explanation = " ".join(explanations)
            return DecisionResult("YELLOW", yellow_triggers, [],
                explanation=combined_explanation)

        return None

    def decide_row(self, row) -> str:
        """
        Wrapper for pandas .apply(). Extracts financial and product
        fields from a DataFrame row and returns the label string.
        """
        financial = {
            # Per-pair computed features.
            "affordability_score": row.get("affordability_score"),
            "price_to_income_ratio": row.get("price_to_income_ratio"),
            "residual_utility_score": row.get("residual_utility_score"),
            "savings_to_price_ratio": row.get("savings_to_price_ratio"),
            "net_worth_indicator": row.get("net_worth_indicator"),
            "credit_risk_indicator": row.get("credit_risk_indicator"),
            # Per-entity DB features.
            "debt_to_income_ratio": row.get("debt_to_income_ratio"),
            "saving_to_income_ratio": row.get("saving_to_income_ratio"),
            "monthly_expense_burden_ratio": row.get("monthly_expense_burden_ratio"),
            "emergency_fund_months": row.get("emergency_fund_months"),
        }
        product = {
            "price": row.get("price", row.get("product_price")),
        }
        return self.decide(financial, product).decision_category

