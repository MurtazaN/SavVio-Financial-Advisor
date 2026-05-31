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

    RED:    5 compound AND rules — each crosses at least 2 correlation groups
            and includes a price_to_income_ratio escape so trivial purchases
            never trigger RED.
    YELLOW: 5 compound AND rules — each crosses at least 2 groups.
            YELLOW triggers when >= 1 rule triggers.
    GREEN:  default when no RED triggers and no YELLOW rules trigger.
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
        spr = self._safe(financial.get("savings_to_price_ratio"), "savings_to_price_ratio")
        nwi = self._safe(financial.get("net_worth_indicator"), "net_worth_indicator")
        credit = self._safe(financial.get("credit_risk_indicator"), "credit_risk_indicator")

        # --- Per-entity DB features (from financial_features.py) ---
        dti = self._safe(financial.get("debt_to_income_ratio"), "debt_to_income_ratio")
        meb = self._safe(financial.get("monthly_expense_burden_ratio"), "monthly_expense_burden_ratio")
        efm = self._safe(financial.get("emergency_fund_months"), "emergency_fund_months")

        logger.info(
            "DecisionEngine inputs: affordability=%s pir=%s spr=%s nwi=%s "
            "credit_risk_indicator(CRI)=%s dti=%s meb=%s efm=%s",
            affordability, pir, spr, nwi, credit, dti, meb, efm,
        )

        # Evaluate RED Rules
        red_result = self._evaluate_red_rules(affordability, pir, spr, nwi, meb, efm, dti)
        if red_result:
            return red_result

        # Evaluate YELLOW Rules
        yellow_result = self._evaluate_yellow_rules(affordability, pir, spr, nwi, credit, efm, meb)
        if yellow_result:
            return yellow_result

        # GREEN: Purchase fits comfortably within the user's finances.
        return DecisionResult("GREEN", ["green:all_clear"], [],
            explanation="Affordable purchase with healthy financial position.")

    def _evaluate_red_rules(self, affordability, pir, spr, nwi, meb, efm, dti) -> Optional[DecisionResult]:
        """
        Evaluate RED rules: Purchase would cause financial harm.
        All conditions within each rule must be true (AND logic).
        Every rule crosses at least 2 independent groups and includes
        a PIR escape hatch so trivial purchases never trigger RED.
        """
        triggered = []

        # RED Rule 1 — Groups 1 + 2
        r1 = affordability < 0 and spr < 1.5 and pir > 0.10
        logger.info(
            "Rule RED-1 red:cant_afford_from_any_angle fired=%s (need aff<0, spr<1.5, pir>0.10): "
            "aff<0=%s spr<1.5=%s pir>0.10=%s",
            r1, affordability < 0, spr < 1.5, pir > 0.10,
        )
        if r1:
            triggered.append("red:cant_afford_from_any_angle")
            return DecisionResult("RED", triggered,
                explanation="Income can't handle it, savings barely cover the price, and the purchase is non-trivial.")

        # RED Rule 2 — Groups 3 + 1 + 2 (MEB > 80%)
        r2 = meb > 0.80 and pir > 0.20 and efm < 3.0
        logger.info(
            "Rule RED-2 red:maxed_budget_significant_purchase fired=%s (need meb>0.80, pir>0.20, efm<3): "
            "meb>0.80=%s pir>0.20=%s efm<3=%s",
            r2, meb > 0.80, pir > 0.20, efm < 3.0,
        )
        if r2:
            triggered.append("red:maxed_budget_significant_purchase")
            return DecisionResult("RED", triggered,
                explanation="Budget over 80% consumed, product is 20%+ of income, and thin emergency fund.")

        # RED Rule 3 — Groups 4 + 1 + 2
        r3 = nwi < -2.0 and affordability < 0 and pir > 0.15 and efm < 3.0
        logger.info(
            "Rule RED-3 red:underwater_no_surplus_significant fired=%s: nwi<-2=%s aff<0=%s pir>0.15=%s efm<3=%s",
            r3, nwi < -2.0, affordability < 0, pir > 0.15, efm < 3.0,
        )
        if r3:
            triggered.append("red:underwater_no_surplus_significant")
            return DecisionResult("RED", triggered,
                explanation="Net worth negative, no surplus income, and a significant purchase.")

        # RED Rule 4 — Groups 2 + 3 + price awareness
        r4 = efm < 1.0 and dti > 0.30 and pir > 0.10
        logger.info(
            "Rule RED-4 red:paycheck_to_paycheck_wipes_net fired=%s: efm<1=%s dti>0.30=%s pir>0.10=%s",
            r4, efm < 1.0, dti > 0.30, pir > 0.10,
        )
        if r4:
            triggered.append("red:paycheck_to_paycheck_wipes_net")
            return DecisionResult("RED", triggered,
                explanation="Less than 1 month emergency fund, DTI over 30%, and purchase is non-trivial.")

        # RED Rule 5 — Groups 1 + 3: flow stress
        # Catches the case where monthly discretionary cannot cover the price
        # AND the budget is already heavily consumed, even if liquid savings
        # could technically absorb the hit.  A fiduciary should flag this as
        # harmful because it forces a draw on savings that a tight budget
        # cannot replenish quickly.
        r5 = affordability < 0 and meb > 0.70 and pir > 0.15
        logger.info(
            "Rule RED-5 red:flow_stress_maxed_budget fired=%s (need aff<0, meb>0.70, pir>0.15): "
            "aff<0=%s meb>0.70=%s pir>0.15=%s",
            r5, affordability < 0, meb > 0.70, pir > 0.15,
        )
        if r5:
            triggered.append("red:flow_stress_maxed_budget")
            return DecisionResult("RED", triggered,
                explanation="Monthly income can't cover this purchase, and over 70% of income is already "
                            "consumed by obligations. Even if savings exist, this forces a draw that a "
                            "tight budget cannot replenish quickly.")

        return None

    def _evaluate_yellow_rules(self, affordability, pir, spr, nwi, credit, efm, meb) -> Optional[DecisionResult]:
        """
        Evaluate YELLOW rules: Genuine concern — user should pause and think.
        YELLOW now triggers when >= 1 rule triggers.
        """
        yellow_triggers = []
        explanations = []

        # YELLOW Rule 1 — Groups 1 + 2
        y1 = affordability < 0 and pir > 0.25 and spr < 5.0
        logger.info(
            "Rule YELLOW-1 yellow:income_pressure fired=%s (aff<0, pir>0.25, spr<5)",
            y1,
        )
        if y1:
            yellow_triggers.append("yellow:income_pressure")
            explanations.append("Income doesn't cover purchase directly and savings cushion is moderate.")

        # YELLOW Rule 2 — Groups 2 + 1
        y2 = spr < 5.0 and pir > 0.10
        logger.info(
            "Rule YELLOW-2 yellow:savings_strain fired=%s (spr<5, pir>0.10): spr<5=%s pir>0.10=%s",
            y2, spr < 5.0, pir > 0.10,
        )
        if y2:
            yellow_triggers.append("yellow:savings_strain")
            explanations.append("Savings can't comfortably absorb the purchase.")

        # YELLOW Rule 3 — Groups 3 + 1 + 2 (MEB > 70%)
        y3 = meb > 0.70 and pir > 0.10 and spr < 5.0
        logger.info(
            "Rule YELLOW-3 yellow:debt_stress fired=%s (meb>0.70, pir>0.10, spr<5): meb>0.70=%s pir>0.10=%s spr<5=%s",
            y3, meb > 0.70, pir > 0.10, spr < 5.0,
        )
        if y3:
            yellow_triggers.append("yellow:debt_stress")
            explanations.append("High debt obligations coupled with significant purchase and thin savings.")

        # YELLOW Rule 4 — Groups 2 + 1
        y4 = efm < 3.0 and affordability < 0
        logger.info(
            "Rule YELLOW-4 yellow:low_resilience fired=%s (efm<3, aff<0): efm<3=%s aff<0=%s",
            y4, efm < 3.0, affordability < 0,
        )
        if y4:
            yellow_triggers.append("yellow:low_resilience")
            explanations.append("Low emergency funds for an unaffordable immediate expense.")

        # YELLOW Rule 5 — Groups 4 + 1 + 2 (CRI = credit_risk_indicator from credit_score)
        y5 = credit < 0.35 and nwi < 1.0 and pir > 0.15 and spr < 5.0
        logger.info(
            "Rule YELLOW-5 yellow:weak_profile fired=%s (CRI<0.35, nwi<1, pir>0.15, spr<5): "
            "CRI<0.35=%s nwi<1=%s pir>0.15=%s spr<5=%s",
            y5, credit < 0.35, nwi < 1.0, pir > 0.15, spr < 5.0,
        )
        if y5:
            yellow_triggers.append("yellow:weak_profile")
            explanations.append("Weak overall financial profile for a significant purchase.")

        # YELLOW triggers when at least one rule triggers.
        if len(yellow_triggers) >= 1:
            combined_explanation = " ".join(explanations)
            return DecisionResult("YELLOW", yellow_triggers, [],
                explanation=combined_explanation)

        logger.info("YELLOW rules: none fired — continuing to GREEN.")
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

