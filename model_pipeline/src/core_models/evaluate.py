"""
Model Evaluation Module.

Evaluates a trained model on a held-out set, computes metrics, generates
all required visualizations, and logs everything to MLflow.

Visualizations generated:
    - Confusion matrix (3x3 for GREEN/YELLOW/RED)
    - ROC curves (per-class, one-vs-rest)
    - Precision-Recall curves (per-class)
    - Calibration / reliability curves (per-class)

Metrics logged:
    - Aggregate: accuracy, weighted F1, weighted ROC-AUC, weighted PR-AUC
    - Per-class: precision, recall, F1 for each of GREEN/YELLOW/RED

Changes from v1:
    - All required visualizations generated and logged as MLflow artifacts.
    - Per-class metrics logged individually (not just weighted averages).
    - Classification report logged as text artifact.
    - PR-AUC added for imbalanced class detection.
    - Calibration curves added for probability reliability.
"""

import os
import logging
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments.
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.calibration import CalibrationDisplay
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(y_true, y_pred, label_names, save_dir):
    """Generate and save confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=label_names,
        ax=ax, cmap="Blues",
    )
    ax.set_title("Confusion Matrix")
    path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path)
    logger.info("Confusion matrix saved and logged.")


def _plot_roc_curves(y_true_bin, y_prob, label_names, save_dir):
    """Generate per-class ROC curves on a single figure (one-vs-rest)."""
    n_classes = len(label_names)
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(n_classes):
        RocCurveDisplay.from_predictions(
            y_true_bin[:, i], y_prob[:, i],
            name=label_names[i], ax=ax,
        )

    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    path = os.path.join(save_dir, "roc_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path)
    logger.info("ROC curves saved and logged.")


def _plot_precision_recall_curves(y_true_bin, y_prob, label_names, save_dir):
    """Generate per-class Precision-Recall curves on a single figure."""
    n_classes = len(label_names)
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(n_classes):
        PrecisionRecallDisplay.from_predictions(
            y_true_bin[:, i], y_prob[:, i],
            name=label_names[i], ax=ax,
        )

    ax.set_title("Precision-Recall Curves (One-vs-Rest)")
    ax.legend(loc="lower left")
    path = os.path.join(save_dir, "pr_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path)
    logger.info("PR curves saved and logged.")


def _plot_calibration_curves(y_true_bin, y_prob, label_names, save_dir):
    """Generate per-class calibration / reliability curves."""
    n_classes = len(label_names)
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(n_classes):
        CalibrationDisplay.from_predictions(
            y_true_bin[:, i], y_prob[:, i],
            n_bins=10, name=label_names[i], ax=ax,
        )

    ax.set_title("Calibration Curves (Reliability Diagram)")
    path = os.path.join(save_dir, "calibration_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path)
    logger.info("Calibration curves saved and logged.")


def _log_classification_report(y_true, y_pred, label_names, save_dir):
    """Save classification report as a text artifact in MLflow."""
    report_str = classification_report(y_true, y_pred, target_names=label_names)
    print(f"\nClassification Report:\n{report_str}")

    path = os.path.join(save_dir, "classification_report.txt")
    with open(path, "w") as f:
        f.write(report_str)
    mlflow.log_artifact(path)
    logger.info("Classification report saved and logged.")


def _log_per_class_metrics(y_true, y_pred, label_names):
    """Log precision, recall, F1 for each class as individual MLflow metrics."""
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)

    for cls_name in label_names:
        if cls_name in report:
            cls_metrics = report[cls_name]
            mlflow.log_metric(f"{cls_name}_precision", round(cls_metrics["precision"], 4))
            mlflow.log_metric(f"{cls_name}_recall", round(cls_metrics["recall"], 4))
            mlflow.log_metric(f"{cls_name}_f1", round(cls_metrics["f1-score"], 4))
            mlflow.log_metric(f"{cls_name}_support", cls_metrics["support"])

    logger.info("Per-class metrics logged for: %s", label_names)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test,
    y_test,
    label_names=None,
    sensitive_df: Optional[pd.DataFrame] = None,
):
    """
    Evaluate the model on the provided dataset.

    Computes aggregate and per-class metrics, generates all visualizations,
    and logs everything to the active MLflow run.

    Args:
        model:        Trained model with .predict() and optionally .predict_proba().
        X_test:       Evaluation features.
        y_test:       Evaluation labels (integer-encoded).
        label_names:  List of class names (e.g., ["GREEN", "RED", "YELLOW"]).
                      If None, uses integer labels.
        sensitive_df: Row-aligned sensitive features for Fairlearn
                      ``ThresholdOptimizer`` (post bias mitigation).

    Returns:
        Dict of aggregate metrics (accuracy, f1_score, roc_auc, pr_auc).
    """
    try:
        from guards.bias_detection import (
            is_fairlearn_threshold_optimizer,
            predict_with_sensitive_features,
        )
    except ImportError:

        def predict_with_sensitive_features(m, X, s=None):
            return m.predict(X)

        def is_fairlearn_threshold_optimizer(m):
            return False

    y_pred = predict_with_sensitive_features(model, X_test, sensitive_df)
    if hasattr(model, "predict_proba") and not is_fairlearn_threshold_optimizer(model):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = None

    n_classes = len(np.unique(y_test))
    if label_names is None:
        label_names = [str(i) for i in range(n_classes)]

    # -- Aggregate metrics --

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    metrics = {
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
    }

    # ROC-AUC (requires probabilities).
    if y_prob is not None:
        if n_classes == 2:
            roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        metrics["roc_auc"] = round(roc_auc, 4)

    # PR-AUC (requires binarized labels + probabilities).
    if y_prob is not None:
        y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
        # Handle binary edge case where label_binarize returns (n, 1) shape.
        if y_test_bin.shape[1] == 1:
            y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

        per_class_pr_auc = average_precision_score(y_test_bin, y_prob, average=None)
        weighted_pr_auc = average_precision_score(y_test_bin, y_prob, average="weighted")
        metrics["pr_auc"] = round(weighted_pr_auc, 4)

        # Log per-class PR-AUC.
        for i, name in enumerate(label_names):
            mlflow.log_metric(f"{name}_pr_auc", round(per_class_pr_auc[i], 4))

    # Log aggregate metrics.
    print("Evaluation Results:")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")
    mlflow.log_metrics(metrics)

    # -- Per-class metrics --

    _log_per_class_metrics(y_test, y_pred, label_names)

    # -- Visualizations --
    # All plots go to a temp directory, are logged to MLflow, then cleaned up.

    with tempfile.TemporaryDirectory() as tmp_dir:

        # Confusion matrix.
        _plot_confusion_matrix(y_test, y_pred, label_names, tmp_dir)

        # Classification report as text artifact.
        _log_classification_report(y_test, y_pred, label_names, tmp_dir)

        # The remaining plots require predicted probabilities.
        if y_prob is not None:
            y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
            if y_test_bin.shape[1] == 1:
                y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

            # ROC curves (per-class, one-vs-rest).
            _plot_roc_curves(y_test_bin, y_prob, label_names, tmp_dir)

            # Precision-Recall curves (per-class).
            _plot_precision_recall_curves(y_test_bin, y_prob, label_names, tmp_dir)

            # Calibration / reliability curves (per-class).
            _plot_calibration_curves(y_test_bin, y_prob, label_names, tmp_dir)

    return metrics
