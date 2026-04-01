import pandas as pd
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score
import mlflow

def evaluate_bias(y_test, y_pred, sensitive_features: pd.DataFrame):
    """
    Uses Fairlearn to detect bias in the model based on sensitive attributes
    (e.g., gender, region). Logs unfairness metrics directly to MLflow.
    """
    print("\nRunning Bias & Fairness Detection...")

    # MLflow metrics dictionary for tracking
    fairness_metrics = {}

    for feature_name in sensitive_features.columns:
        sf_col = sensitive_features[feature_name]

        # Demographic Parity: Are predictions independent of the sensitive feature?
        dpd = demographic_parity_difference((y_test==0).astype(int), (y_pred==0).astype(int), sensitive_features=sf_col)

        # Equalized Odds: Do different groups have the same True Positive / False Positive Rates?
        eod = equalized_odds_difference((y_test==0).astype(int), (y_pred==0).astype(int), sensitive_features=sf_col)

        print(f"  Feature: {feature_name}")
        print(f"    - Demographic Parity Diff: {dpd:.4f} (Ideal: 0)")
        print(f"    - Equalized Odds Diff:     {eod:.4f} (Ideal: 0)")

        # Record into MLflow with the feature name attached
        fairness_metrics[f"bias_dpd_{feature_name}"] = dpd
        fairness_metrics[f"bias_eod_{feature_name}"] = eod

    # Log it upstream
    mlflow.log_metrics(fairness_metrics)

    return fairness_metrics
