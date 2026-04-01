"""
Layer 2 Downgrade Engine — Product & Review Based Caution.

This engine takes the authoritative Layer 1 financial label
from `financial_engine.DecisionEngine` and can only downgrade it
by at most one step based on product and review risk patterns:

    GREEN -> YELLOW
    YELLOW -> RED
    RED    -> RED  (never upgraded)

Downgrade requires BOTH:
    - at least one product rule trigger, AND
    - at least one review rule trigger.

This follows the same cross-group design principle used in the
financial engine: a downgrade requires corroboration from two
independent signal categories (product metadata + review data)
to avoid false triggers from a single source.

Product feature groups (by underlying driver):
    Group A — Rating quality (avg_rating):   value_density,
                                              category_rating_deviation
    Group B — Review volume (rating_number): review_confidence,
                                              cold_start_flag
    Group C — Review controversy (variance):  rating_polarization,
                                              quality_risk_score
    Group D — Price positioning (price/cat):  price_category_rank

Review feature groups (by underlying driver):
    Group E — Purchase verification:  verified_purchase_ratio
    Group F — Helpfulness dist.:      helpful_concentration
    Group G — Sentiment direction:    sentiment_spread
    Group H — Content quality:        review_depth_score
    Group I — Reviewer identity:      reviewer_diversity
    Group J — Rating extremeness:     extreme_rating_ratio
"""

from dataclasses import dataclass, field
from typing import List

from features.product_features import ProductFeatures
from features.review_features import ReviewFeatures


@dataclass
class DowngradeResult:
    final_label: str
    original_label: str
    was_downgraded: bool
    product_triggers: List[str] = field(default_factory=list)
    review_triggers: List[str] = field(default_factory=list)
    explanation: str = ""


class DowngradeEngine:
    """
    Rule-based downgrade engine combining product and review signals.

    Product rules (PR1–PR3) use the 7 product features.
    Review rules (RR1–RR3) use the 6 review features.
    """

    def evaluate(
        self,
        financial_label: str,
        product_features: ProductFeatures,
        review_features: ReviewFeatures,
    ) -> DowngradeResult:
        """
        Evaluate product and review rules and apply a one-step downgrade
        to the financial label if BOTH sides flag concerns.
        """
        product_triggers, product_expl = self._evaluate_product_rules(product_features)
        review_triggers, review_expl = self._evaluate_review_rules(review_features)

        if product_triggers and review_triggers:
            final_label = self._apply_downgrade(financial_label)
            was_downgraded = final_label != financial_label
        else:
            final_label = financial_label
            was_downgraded = False

        explanation_parts: List[str] = []
        if product_expl:
            explanation_parts.append("Product concerns: " + " ".join(product_expl))
        if review_expl:
            explanation_parts.append("Review concerns: " + " ".join(review_expl))

        return DowngradeResult(
            final_label=final_label,
            original_label=financial_label,
            was_downgraded=was_downgraded,
            product_triggers=product_triggers,
            review_triggers=review_triggers,
            explanation=" ".join(explanation_parts).strip(),
        )

    def _evaluate_product_rules(self, pf: ProductFeatures):
        triggers: List[str] = []
        explanations: List[str] = []

        # PR1: Below-category-average rating AND weak review evidence (A + B)
        if pf.category_rating_deviation < -0.5 and pf.review_confidence < 0.3:
            triggers.append("PR1:underrated_unverified")
            explanations.append(
                "Rates below category average with insufficient review evidence."
            )

        # PR2: Well below category average AND premium priced (A + D)
        if pf.category_rating_deviation < -0.8 and pf.price_category_rank > 0.7:
            triggers.append("PR2:worst_premium_in_category")
            explanations.append(
                "Among the worst-rated products in its category while priced as a premium option."
            )

        # PR3: Polarized AND cold-start AND expensive (C + B + D)
        if (
            pf.rating_polarization > 0.6
            and pf.cold_start_flag == 1
            and pf.price_category_rank > 0.5
        ):
            triggers.append("PR3:polarized_new_expensive")
            explanations.append(
                "Highly polarized early reviews on a relatively expensive, cold-start product."
            )

        return triggers, explanations

    def _evaluate_review_rules(self, rf: ReviewFeatures):
        triggers: List[str] = []
        explanations: List[str] = []

        # RR1: Fake review pattern (E + J + H)
        if (
            rf.verified_purchase_ratio < 0.3
            and rf.extreme_rating_ratio > 0.8
            and rf.review_depth_score < 0.2
        ):
            triggers.append("RR1:fake_review_pattern")
            explanations.append(
                "Many shallow, extreme reviews with very few verified purchases (possible astroturfing)."
            )

        # RR2: Unverified opinion domination (E + F + I)
        if (
            rf.verified_purchase_ratio < 0.4
            and rf.helpful_concentration > 0.7
            and rf.reviewer_diversity < 0.5
        ):
            triggers.append("RR2:unverified_opinion_domination")
            explanations.append(
                "Helpfulness votes concentrated on a small, unverified reviewer set."
            )

        # RR3: Negative, shallow, unverified (G + H + E)
        if (
            rf.sentiment_spread < -0.3
            and rf.review_depth_score < 0.3
            and rf.verified_purchase_ratio < 0.5
        ):
            triggers.append("RR3:negative_shallow_unverified")
            explanations.append(
                "Mostly negative, shallow reviews from largely unverified purchasers."
            )

        return triggers, explanations

    def _apply_downgrade(self, label: str) -> str:
        """
        Apply at most a single-step downgrade.
        """
        if label == "GREEN":
            return "YELLOW"
        if label == "YELLOW":
            return "RED"
        return "RED"

