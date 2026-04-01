import sys
import os

import pytest


from deterministic_engine.downgrade_engine import DowngradeEngine
from features.product_features import ProductFeatures
from features.review_features import ReviewFeatures


def _base_product_features() -> ProductFeatures:
    return ProductFeatures(
        value_density=1.0,
        review_confidence=0.5,
        rating_polarization=0.2,
        quality_risk_score=0.5,
        cold_start_flag=0,
        price_category_rank=0.5,
        category_rating_deviation=0.0,
    )


def _base_review_features() -> ReviewFeatures:
    return ReviewFeatures(
        verified_purchase_ratio=0.8,
        helpful_concentration=0.2,
        sentiment_spread=0.2,
        review_depth_score=0.5,
        reviewer_diversity=0.8,
        extreme_rating_ratio=0.2,
    )


class TestDowngradeRules:
    def test_product_rule1_triggers(self):
        engine = DowngradeEngine()
        pf = _base_product_features()
        pf.category_rating_deviation = -0.8
        pf.review_confidence = 0.2
        triggers, _ = engine._evaluate_product_rules(pf)
        assert any("PR1" in t for t in triggers)

    def test_product_rule2_triggers(self):
        engine = DowngradeEngine()
        pf = _base_product_features()
        pf.category_rating_deviation = -1.0
        pf.price_category_rank = 0.8
        triggers, _ = engine._evaluate_product_rules(pf)
        assert any("PR2" in t for t in triggers)

    def test_product_rule3_triggers(self):
        engine = DowngradeEngine()
        pf = _base_product_features()
        pf.rating_polarization = 0.7
        pf.cold_start_flag = 1
        pf.price_category_rank = 0.6
        triggers, _ = engine._evaluate_product_rules(pf)
        assert any("PR3" in t for t in triggers)

    def test_review_rule1_triggers(self):
        engine = DowngradeEngine()
        rf = _base_review_features()
        rf.verified_purchase_ratio = 0.1
        rf.extreme_rating_ratio = 0.9
        rf.review_depth_score = 0.1
        triggers, _ = engine._evaluate_review_rules(rf)
        assert any("RR1" in t for t in triggers)

    def test_review_rule2_triggers(self):
        engine = DowngradeEngine()
        rf = _base_review_features()
        rf.verified_purchase_ratio = 0.3
        rf.helpful_concentration = 0.8
        rf.reviewer_diversity = 0.4
        triggers, _ = engine._evaluate_review_rules(rf)
        assert any("RR2" in t for t in triggers)

    def test_review_rule3_triggers(self):
        engine = DowngradeEngine()
        rf = _base_review_features()
        rf.sentiment_spread = -0.4
        rf.review_depth_score = 0.2
        rf.verified_purchase_ratio = 0.4
        triggers, _ = engine._evaluate_review_rules(rf)
        assert any("RR3" in t for t in triggers)


class TestCombinedLogic:
    def test_product_only_does_not_downgrade(self):
        """Product trigger alone is insufficient — need both sides."""
        engine = DowngradeEngine()
        pf = _base_product_features()
        rf = _base_review_features()
        pf.category_rating_deviation = -0.8
        pf.review_confidence = 0.2
        result = engine.evaluate("GREEN", pf, rf)
        assert not result.was_downgraded
        assert result.final_label == "GREEN"

    def test_review_only_does_not_downgrade(self):
        """Review trigger alone is insufficient — need both sides."""
        engine = DowngradeEngine()
        pf = _base_product_features()
        rf = _base_review_features()
        rf.verified_purchase_ratio = 0.1
        rf.extreme_rating_ratio = 0.9
        rf.review_depth_score = 0.1
        result = engine.evaluate("GREEN", pf, rf)
        assert not result.was_downgraded
        assert result.final_label == "GREEN"

    def test_both_sides_downgrade_one_step(self):
        """Product AND review triggers together → one-step downgrade."""
        engine = DowngradeEngine()
        pf = _base_product_features()
        rf = _base_review_features()
        pf.category_rating_deviation = -0.8
        pf.review_confidence = 0.2
        rf.verified_purchase_ratio = 0.1
        rf.extreme_rating_ratio = 0.9
        rf.review_depth_score = 0.1

        result = engine.evaluate("GREEN", pf, rf)
        assert result.was_downgraded
        assert result.final_label == "YELLOW"
        assert result.original_label == "GREEN"

    def test_yellow_downgrades_to_red(self):
        engine = DowngradeEngine()
        pf = _base_product_features()
        rf = _base_review_features()
        pf.category_rating_deviation = -0.8
        pf.review_confidence = 0.2
        rf.verified_purchase_ratio = 0.1
        rf.extreme_rating_ratio = 0.9
        rf.review_depth_score = 0.1

        result = engine.evaluate("YELLOW", pf, rf)
        assert result.final_label == "RED"

    def test_red_stays_red(self):
        engine = DowngradeEngine()
        pf = _base_product_features()
        rf = _base_review_features()
        pf.category_rating_deviation = -0.8
        pf.review_confidence = 0.2
        rf.verified_purchase_ratio = 0.1
        rf.extreme_rating_ratio = 0.9
        rf.review_depth_score = 0.1

        result = engine.evaluate("RED", pf, rf)
        assert result.final_label == "RED"

