"""Unit tests for deterministic labeling orchestration."""

from __future__ import annotations
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import deterministic_engine.labeling_pipeline as lp

MODULE = "deterministic_engine.labeling_pipeline"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _downgrade_result(label: str) -> MagicMock:
    r = MagicMock()
    r.final_label = label
    return r


def _fake_decision_engine(label: str):
    """Return a real class whose decide_row() always returns `label`.
    MagicMock breaks pandas.DataFrame.apply(), so we use a real class.
    """
    class FakeDecisionEngine:
        def decide_row(self, row):
            return label
    return FakeDecisionEngine()


def _fake_decision_engine_sequence(labels: list[str]):
    """Return a real class whose decide_row() iterates through `labels`."""
    it = iter(labels)
    class FakeDecisionEngine:
        def decide_row(self, row):
            return next(it)
    return FakeDecisionEngine()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_scenarios():
    return pd.DataFrame({
        "product_id": ["p1", "p2"],
        "price": [10.0, 50.0],
    })


@pytest.fixture
def base_products():
    return pd.DataFrame({"product_id": ["p1", "p2"], "category": ["A", "B"]})


@pytest.fixture
def base_reviews():
    return pd.DataFrame({"product_id": ["p1", "p2"], "rating": [4.0, 3.5]})


@pytest.fixture
def product_feats_df():
    return pd.DataFrame({
        "product_id":                ["p1",  "p2"],
        "value_density":             [0.8,   0.4],
        "review_confidence":         [0.9,   0.6],
        "rating_polarization":       [0.1,   0.3],
        "quality_risk_score":        [0.05,  0.2],
        "cold_start_flag":           [0,     1],
        "price_category_rank":       [0.7,   0.5],
        "category_rating_deviation": [0.02,  0.1],
    })


@pytest.fixture
def review_feats_df():
    return pd.DataFrame({
        "product_id":               ["p1",  "p2"],
        "verified_purchase_ratio":  [0.9,   0.5],
        "helpful_concentration":    [0.3,   0.6],
        "sentiment_spread":         [0.2,   0.4],
        "review_depth_score":       [0.7,   0.5],
        "reviewer_diversity":       [0.8,   0.4],
        "extreme_rating_ratio":     [0.05,  0.15],
    }).set_index("product_id")


# ---------------------------------------------------------------------------
# apply_deterministic_labels — Layer 1 only (no reviews)
# ---------------------------------------------------------------------------

class TestApplyDeterministicLabelsLayer1Only:

    def test_final_recommendation_column_present(self, base_scenarios, base_products):
        with patch(f"{MODULE}.DecisionEngine", return_value=_fake_decision_engine("BUY")):
            result = lp.apply_deterministic_labels(base_scenarios, base_products)
        assert "final_recommendation" in result.columns

    def test_labels_match_engine_output(self, base_scenarios, base_products):
        with patch(f"{MODULE}.DecisionEngine", return_value=_fake_decision_engine_sequence(["BUY", "HOLD"])):
            result = lp.apply_deterministic_labels(base_scenarios, base_products)
        assert list(result["final_recommendation"]) == ["BUY", "HOLD"]

    def test_downgraded_column_is_all_zero(self, base_scenarios, base_products):
        with patch(f"{MODULE}.DecisionEngine", return_value=_fake_decision_engine("BUY")):
            result = lp.apply_deterministic_labels(base_scenarios, base_products)
        assert (result["downgraded"] == 0).all()

    def test_internal_l1_label_not_in_output(self, base_scenarios, base_products):
        with patch(f"{MODULE}.DecisionEngine", return_value=_fake_decision_engine("BUY")):
            result = lp.apply_deterministic_labels(base_scenarios, base_products)
        assert "_l1_label" not in result.columns

    def test_input_dataframe_not_mutated(self, base_scenarios, base_products):
        original = base_scenarios.copy()
        with patch(f"{MODULE}.DecisionEngine", return_value=_fake_decision_engine("BUY")):
            lp.apply_deterministic_labels(base_scenarios, base_products)
        pd.testing.assert_frame_equal(base_scenarios, original)


# ---------------------------------------------------------------------------
# apply_deterministic_labels — Layer 1 + Layer 2 (with reviews)
# ---------------------------------------------------------------------------

class TestApplyDeterministicLabelsLayer2:

    def test_downgrade_applied_when_label_changes(
        self, base_scenarios, base_products, base_reviews, product_feats_df, review_feats_df,
    ):
        mock_downgrade = MagicMock()
        mock_downgrade.evaluate.side_effect = [
            _downgrade_result("HOLD"),
            _downgrade_result("BUY"),
        ]
        with patch(f"{MODULE}.DecisionEngine", return_value=_fake_decision_engine("BUY")), \
             patch(f"{MODULE}.DowngradeEngine", return_value=mock_downgrade), \
             patch(f"{MODULE}.compute_product_features_batch", return_value=product_feats_df), \
             patch(f"{MODULE}.compute_review_features_batch", return_value=review_feats_df):
            result = lp.apply_deterministic_labels(base_scenarios, base_products, base_reviews)
        assert result.loc[result["product_id"] == "p1", "final_recommendation"].iloc[0] == "HOLD"
        assert result.loc[result["product_id"] == "p1", "downgraded"].iloc[0] == 1
        assert result.loc[result["product_id"] == "p2", "downgraded"].iloc[0] == 0

    def test_no_downgrade_when_label_unchanged(
        self, base_scenarios, base_products, base_reviews, product_feats_df, review_feats_df,
    ):
        mock_downgrade = MagicMock()
        mock_downgrade.evaluate.return_value = _downgrade_result("BUY")
        with patch(f"{MODULE}.DecisionEngine", return_value=_fake_decision_engine("BUY")), \
             patch(f"{MODULE}.DowngradeEngine", return_value=mock_downgrade), \
             patch(f"{MODULE}.compute_product_features_batch", return_value=product_feats_df), \
             patch(f"{MODULE}.compute_review_features_batch", return_value=review_feats_df):
            result = lp.apply_deterministic_labels(base_scenarios, base_products, base_reviews)
        assert (result["downgraded"] == 0).all()

    def test_l1_label_not_in_output(
        self, base_scenarios, base_products, base_reviews, product_feats_df, review_feats_df,
    ):
        mock_downgrade = MagicMock()
        mock_downgrade.evaluate.return_value = _downgrade_result("BUY")
        with patch(f"{MODULE}.DecisionEngine", return_value=_fake_decision_engine("BUY")), \
             patch(f"{MODULE}.DowngradeEngine", return_value=mock_downgrade), \
             patch(f"{MODULE}.compute_product_features_batch", return_value=product_feats_df), \
             patch(f"{MODULE}.compute_review_features_batch", return_value=review_feats_df):
            result = lp.apply_deterministic_labels(base_scenarios, base_products, base_reviews)
        assert "_l1_label" not in result.columns


# ---------------------------------------------------------------------------
# _attach_layer2_features
# ---------------------------------------------------------------------------

class TestAttachLayer2Features:

    def test_product_feature_columns_attached(
        self, base_scenarios, base_products, base_reviews, product_feats_df, review_feats_df,
    ):
        with patch(f"{MODULE}.compute_product_features_batch", return_value=product_feats_df), \
             patch(f"{MODULE}.compute_review_features_batch", return_value=review_feats_df):
            result = lp._attach_layer2_features(base_scenarios, base_products, base_reviews)
        for col in ["value_density", "review_confidence", "cold_start_flag"]:
            assert col in result.columns

    def test_review_feature_columns_attached(
        self, base_scenarios, base_products, base_reviews, product_feats_df, review_feats_df,
    ):
        with patch(f"{MODULE}.compute_product_features_batch", return_value=product_feats_df), \
             patch(f"{MODULE}.compute_review_features_batch", return_value=review_feats_df):
            result = lp._attach_layer2_features(base_scenarios, base_products, base_reviews)
        for col in ["verified_purchase_ratio", "sentiment_spread", "reviewer_diversity"]:
            assert col in result.columns

    def test_missing_reviews_filled_with_zero(self, base_scenarios, base_products):
        prod_df = pd.DataFrame({
            "product_id":                ["p1", "p2"],
            "value_density":             [0.5,  0.5],
            "review_confidence":         [0.5,  0.5],
            "rating_polarization":       [0.1,  0.1],
            "quality_risk_score":        [0.1,  0.1],
            "cold_start_flag":           [0,    0],
            "price_category_rank":       [0.5,  0.5],
            "category_rating_deviation": [0.0,  0.0],
        })
        empty_rev = pd.DataFrame(columns=[
            "product_id", "verified_purchase_ratio", "helpful_concentration",
            "sentiment_spread", "review_depth_score", "reviewer_diversity",
            "extreme_rating_ratio",
        ]).set_index("product_id")
        with patch(f"{MODULE}.compute_product_features_batch", return_value=prod_df), \
             patch(f"{MODULE}.compute_review_features_batch", return_value=empty_rev):
            result = lp._attach_layer2_features(
                base_scenarios, base_products,
                pd.DataFrame(columns=["product_id", "rating"]),
            )
        assert (result["verified_purchase_ratio"] == 0.0).all()
        assert result["verified_purchase_ratio"].isna().sum() == 0

    def test_duplicate_products_deduplicated(
        self, base_scenarios, base_reviews, product_feats_df, review_feats_df,
    ):
        duped = pd.DataFrame({
            "product_id": ["p1", "p1", "p2"],
            "category":   ["A",  "A",  "B"],
        })
        with patch(f"{MODULE}.compute_product_features_batch", return_value=product_feats_df) as mock_prod, \
             patch(f"{MODULE}.compute_review_features_batch", return_value=review_feats_df):
            lp._attach_layer2_features(base_scenarios, duped, base_reviews)
        called_df = mock_prod.call_args[0][0]
        assert called_df["product_id"].duplicated().sum() == 0


# ---------------------------------------------------------------------------
# _apply_layer2_downgrade
# ---------------------------------------------------------------------------

class TestApplyLayer2Downgrade:

    def _scenarios(self):
        return pd.DataFrame({
            "product_id":                ["p1",  "p2"],
            "_l1_label":                 ["BUY", "BUY"],
            "value_density":             [0.8,   0.4],
            "review_confidence":         [0.9,   0.6],
            "rating_polarization":       [0.1,   0.3],
            "quality_risk_score":        [0.05,  0.2],
            "cold_start_flag":           [0,     1],
            "price_category_rank":       [0.7,   0.5],
            "category_rating_deviation": [0.02,  0.1],
            "verified_purchase_ratio":   [0.9,   0.5],
            "helpful_concentration":     [0.3,   0.6],
            "sentiment_spread":          [0.2,   0.4],
            "review_depth_score":        [0.7,   0.5],
            "reviewer_diversity":        [0.8,   0.4],
            "extreme_rating_ratio":      [0.05,  0.15],
        })

    def test_downgraded_flag_when_label_changes(self):
        mock_downgrade = MagicMock()
        mock_downgrade.evaluate.side_effect = [
            _downgrade_result("HOLD"),
            _downgrade_result("BUY"),
        ]
        with patch(f"{MODULE}.DowngradeEngine", return_value=mock_downgrade):
            result = lp._apply_layer2_downgrade(self._scenarios())
        assert result.loc[0, "downgraded"] == 1
        assert result.loc[1, "downgraded"] == 0

    def test_final_recommendation_reflects_downgrade(self):
        mock_downgrade = MagicMock()
        mock_downgrade.evaluate.side_effect = [
            _downgrade_result("HOLD"),
            _downgrade_result("SELL"),
        ]
        with patch(f"{MODULE}.DowngradeEngine", return_value=mock_downgrade):
            result = lp._apply_layer2_downgrade(self._scenarios())
        assert list(result["final_recommendation"]) == ["HOLD", "SELL"]

    def test_evaluate_called_once_per_row(self):
        mock_downgrade = MagicMock()
        mock_downgrade.evaluate.return_value = _downgrade_result("BUY")
        with patch(f"{MODULE}.DowngradeEngine", return_value=mock_downgrade):
            scenarios = self._scenarios()
            lp._apply_layer2_downgrade(scenarios)
        assert mock_downgrade.evaluate.call_count == len(scenarios)