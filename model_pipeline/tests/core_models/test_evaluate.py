"""Unit tests for the model evaluation module (evaluate.py).

All MLflow calls and file I/O are mocked so tests run without
a real MLflow server or filesystem artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from core_models.evaluate import (
    evaluate_model,
    _plot_confusion_matrix,
    _plot_roc_curves,
    _plot_precision_recall_curves,
    _plot_calibration_curves,
    _log_classification_report,
    _log_per_class_metrics,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def label_names():
    return ["GREEN", "YELLOW", "RED"]


@pytest.fixture
def binary_label_names():
    return ["GREEN", "RED"]


@pytest.fixture
def y_true_3class():
    """Balanced 3-class ground truth."""
    return np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])


@pytest.fixture
def y_pred_3class():
    return np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])


@pytest.fixture
def y_prob_3class():
    rng = np.random.default_rng(42)
    probs = rng.dirichlet(alpha=[1, 1, 1], size=9)
    return probs


@pytest.fixture
def mock_model(y_pred_3class, y_prob_3class):
    m = MagicMock()
    m.predict.return_value = y_pred_3class
    m.predict_proba.return_value = y_prob_3class
    return m


@pytest.fixture
def mock_model_no_proba(y_pred_3class):
    """Model without predict_proba (e.g. SVM without probability=True)."""
    m = MagicMock(spec=["predict"])
    m.predict.return_value = y_pred_3class
    return m


# ---------------------------------------------------------------------------
# evaluate_model — return value and MLflow logging
# ---------------------------------------------------------------------------

class TestEvaluateModel:

    @patch("core_models.evaluate.mlflow")
    def test_returns_dict_with_required_keys(
        self, mock_mlflow, mock_model, y_true_3class, label_names
    ):
        result = evaluate_model(mock_model, np.zeros((9, 3)), y_true_3class, label_names)
        for key in ("accuracy", "f1_score", "roc_auc", "pr_auc"):
            assert key in result, f"Missing key: {key}"

    @patch("core_models.evaluate.mlflow")
    def test_accuracy_is_between_0_and_1(
        self, mock_mlflow, mock_model, y_true_3class, label_names
    ):
        result = evaluate_model(mock_model, np.zeros((9, 3)), y_true_3class, label_names)
        assert 0.0 <= result["accuracy"] <= 1.0

    @patch("core_models.evaluate.mlflow")
    def test_f1_is_between_0_and_1(
        self, mock_mlflow, mock_model, y_true_3class, label_names
    ):
        result = evaluate_model(mock_model, np.zeros((9, 3)), y_true_3class, label_names)
        assert 0.0 <= result["f1_score"] <= 1.0

    @patch("core_models.evaluate.mlflow")
    def test_roc_auc_is_between_0_and_1(
        self, mock_mlflow, mock_model, y_true_3class, label_names
    ):
        result = evaluate_model(mock_model, np.zeros((9, 3)), y_true_3class, label_names)
        assert 0.0 <= result["roc_auc"] <= 1.0

    @patch("core_models.evaluate.mlflow")
    def test_pr_auc_is_between_0_and_1(
        self, mock_mlflow, mock_model, y_true_3class, label_names
    ):
        result = evaluate_model(mock_model, np.zeros((9, 3)), y_true_3class, label_names)
        assert 0.0 <= result["pr_auc"] <= 1.0

    @patch("core_models.evaluate.mlflow")
    def test_metrics_logged_to_mlflow(
        self, mock_mlflow, mock_model, y_true_3class, label_names
    ):
        evaluate_model(mock_model, np.zeros((9, 3)), y_true_3class, label_names)
        mock_mlflow.log_metrics.assert_called_once()

    @patch("core_models.evaluate.mlflow")
    def test_artifacts_logged_to_mlflow(
        self, mock_mlflow, mock_model, y_true_3class, label_names
    ):
        evaluate_model(mock_model, np.zeros((9, 3)), y_true_3class, label_names)
        assert mock_mlflow.log_artifact.call_count >= 1

    @patch("core_models.evaluate.mlflow")
    def test_per_class_metrics_logged(
        self, mock_mlflow, mock_model, y_true_3class, label_names
    ):
        evaluate_model(mock_model, np.zeros((9, 3)), y_true_3class, label_names)
        logged_keys = [c[0][0] for c in mock_mlflow.log_metric.call_args_list]
        for cls in label_names:
            assert any(cls in k for k in logged_keys), f"No metrics logged for class {cls}"

    @patch("core_models.evaluate.mlflow")
    def test_model_without_predict_proba(
        self, mock_mlflow, mock_model_no_proba, y_true_3class, label_names
    ):
        """evaluate_model should not crash when model has no predict_proba."""
        result = evaluate_model(
            mock_model_no_proba, np.zeros((9, 3)), y_true_3class, label_names
        )
        assert "accuracy" in result
        assert "roc_auc" not in result
        assert "pr_auc" not in result

    @patch("core_models.evaluate.mlflow")
    def test_default_label_names_used_when_none(
        self, mock_mlflow, mock_model, y_true_3class
    ):
        """Should not raise when label_names=None."""
        result = evaluate_model(mock_model, np.zeros((9, 3)), y_true_3class, None)
        assert "accuracy" in result

    @patch("core_models.evaluate.mlflow")
    def test_metrics_are_rounded_to_4_decimals(
        self, mock_mlflow, mock_model, y_true_3class, label_names
    ):
        result = evaluate_model(mock_model, np.zeros((9, 3)), y_true_3class, label_names)
        for v in result.values():
            assert round(v, 4) == v


# ---------------------------------------------------------------------------
# _log_per_class_metrics
# ---------------------------------------------------------------------------

class TestLogPerClassMetrics:

    @patch("core_models.evaluate.mlflow")
    def test_logs_precision_recall_f1_for_each_class(
        self, mock_mlflow, y_true_3class, y_pred_3class, label_names
    ):
        _log_per_class_metrics(y_true_3class, y_pred_3class, label_names)
        logged_keys = [c[0][0] for c in mock_mlflow.log_metric.call_args_list]
        for cls in label_names:
            assert any(f"{cls}_precision" in k for k in logged_keys)
            assert any(f"{cls}_recall" in k for k in logged_keys)
            assert any(f"{cls}_f1" in k for k in logged_keys)

    @patch("core_models.evaluate.mlflow")
    def test_logs_support_for_each_class(
        self, mock_mlflow, y_true_3class, y_pred_3class, label_names
    ):
        _log_per_class_metrics(y_true_3class, y_pred_3class, label_names)
        logged_keys = [c[0][0] for c in mock_mlflow.log_metric.call_args_list]
        for cls in label_names:
            assert any(f"{cls}_support" in k for k in logged_keys)


# ---------------------------------------------------------------------------
# Visualization helpers — just verify they save and log artifacts
# ---------------------------------------------------------------------------

class TestPlotHelpers:

    @patch("core_models.evaluate.mlflow")
    def test_plot_confusion_matrix_logs_artifact(
        self, mock_mlflow, tmp_path, y_true_3class, y_pred_3class, label_names
    ):
        _plot_confusion_matrix(y_true_3class, y_pred_3class, label_names, str(tmp_path))
        mock_mlflow.log_artifact.assert_called_once()
        assert (tmp_path / "confusion_matrix.png").exists()

    @patch("core_models.evaluate.mlflow")
    def test_plot_roc_curves_logs_artifact(
        self, mock_mlflow, tmp_path, y_true_3class, y_prob_3class, label_names
    ):
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true_3class, classes=[0, 1, 2])
        _plot_roc_curves(y_bin, y_prob_3class, label_names, str(tmp_path))
        mock_mlflow.log_artifact.assert_called_once()
        assert (tmp_path / "roc_curves.png").exists()

    @patch("core_models.evaluate.mlflow")
    def test_plot_pr_curves_logs_artifact(
        self, mock_mlflow, tmp_path, y_true_3class, y_prob_3class, label_names
    ):
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true_3class, classes=[0, 1, 2])
        _plot_precision_recall_curves(y_bin, y_prob_3class, label_names, str(tmp_path))
        mock_mlflow.log_artifact.assert_called_once()
        assert (tmp_path / "pr_curves.png").exists()

    @patch("core_models.evaluate.mlflow")
    def test_plot_calibration_curves_logs_artifact(
        self, mock_mlflow, tmp_path, y_true_3class, y_prob_3class, label_names
    ):
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true_3class, classes=[0, 1, 2])
        _plot_calibration_curves(y_bin, y_prob_3class, label_names, str(tmp_path))
        mock_mlflow.log_artifact.assert_called_once()
        assert (tmp_path / "calibration_curves.png").exists()

    @patch("core_models.evaluate.mlflow")
    def test_log_classification_report_saves_file(
        self, mock_mlflow, tmp_path, y_true_3class, y_pred_3class, label_names
    ):
        _log_classification_report(y_true_3class, y_pred_3class, label_names, str(tmp_path))
        mock_mlflow.log_artifact.assert_called_once()
        assert (tmp_path / "classification_report.txt").exists()

    @patch("core_models.evaluate.mlflow")
    def test_classification_report_contains_class_names(
        self, mock_mlflow, tmp_path, y_true_3class, y_pred_3class, label_names
    ):
        _log_classification_report(y_true_3class, y_pred_3class, label_names, str(tmp_path))
        content = (tmp_path / "classification_report.txt").read_text()
        for cls in label_names:
            assert cls in content