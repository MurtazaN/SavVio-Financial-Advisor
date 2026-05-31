"""
Tests for model_pipeline/src/guards/bias_detection.py

Covers:
  - _primary_mitigation_axis: flag-driven vs fallback axis selection
  - evaluate_bias: correct GREEN class targeting, returns 3-tuple, flag logic
  - detect_and_mitigate: passes green_class_idx through; no mitigation when gate passes
"""
import sys
from unittest.mock import MagicMock
sys.modules['sqlalchemy'] = MagicMock()
import os
import types

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal stubs so the module can be imported without real MLflow / Fairlearn
# ---------------------------------------------------------------------------

# Stub mlflow
mlflow_stub = types.ModuleType("mlflow")
mlflow_stub.log_metrics = lambda *a, **kw: None
mlflow_stub.log_metric = lambda *a, **kw: None
mlflow_stub.set_tag = lambda *a, **kw: None
mlflow_stub.log_artifact = lambda *a, **kw: None
sys.modules.setdefault("mlflow", mlflow_stub)

# Stub fairlearn.metrics
fl_metrics = types.ModuleType("fairlearn.metrics")
fl_metrics.demographic_parity_difference = lambda *a, **kw: 0.0
fl_metrics.equalized_odds_difference = lambda *a, **kw: 0.0
sys.modules.setdefault("fairlearn", types.ModuleType("fairlearn"))
sys.modules.setdefault("fairlearn.metrics", fl_metrics)

# Stub fairlearn.postprocessing
fl_post = types.ModuleType("fairlearn.postprocessing")
class _DummyTO:
    def fit(self, *a, **kw): return self
    def predict(self, X, sensitive_features=None): return np.zeros(len(X), dtype=int)
fl_post.ThresholdOptimizer = _DummyTO
sys.modules.setdefault("fairlearn.postprocessing", fl_post)

# Make sure the src directory is on the path
SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Stub config
config_stub = types.ModuleType("config")
class _Config:
    BIAS_DISPARITY_THRESHOLD = 0.10
config_stub.Config = _Config
sys.modules.setdefault("config", config_stub)

from guards.bias_detection import (
    _primary_mitigation_axis,
    evaluate_bias,
    detect_and_mitigate,
    DEMOGRAPHIC_SLICES,
    FINANCIAL_SLICES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sensitive(n=100, employment=True, region=True):
    data = {}
    if employment:
        data["employment_status"] = np.random.choice(["Employed", "Unemployed"], n)
    if region:
        data["region"] = np.random.choice(["North", "South", "East"], n)
    return pd.DataFrame(data)


def _make_labels(n=100, green_idx=0, n_classes=3):
    """Balanced multiclass labels where green_idx is GREEN."""
    labels = np.tile(np.arange(n_classes), n // n_classes + 1)[:n]
    np.random.shuffle(labels)
    return labels


# ---------------------------------------------------------------------------
# Tests: _primary_mitigation_axis
# ---------------------------------------------------------------------------

class TestPrimaryMitigationAxis:
    def test_fallback_prefers_savings_band(self):
        df = pd.DataFrame({"savings_band": ["a"], "employment_status": ["b"], "region": ["c"]})
        assert _primary_mitigation_axis(df) == "savings_band"

    def test_fallback_prefers_employment_when_no_savings(self):
        df = pd.DataFrame({"employment_status": ["b"], "region": ["c"]})
        assert _primary_mitigation_axis(df) == "employment_status"

    def test_fallback_uses_first_col_when_nothing_known(self):
        df = pd.DataFrame({"income_band": ["a"]})
        assert _primary_mitigation_axis(df) == "income_band"

    def test_flag_drives_demographic_preference(self):
        """If employment_status is flagged it should be preferred over savings_band."""
        df = pd.DataFrame({"savings_band": ["x"], "employment_status": ["y"]})
        flags = ["DPD_employment_status=0.15", "EOD_income_band=0.20"]
        assert _primary_mitigation_axis(df, dpd_eod_flags=flags) == "employment_status"

    def test_flag_falls_back_when_col_not_in_sensitive(self):
        """Flagged column not in sensitive_df → fall back to priority order."""
        df = pd.DataFrame({"savings_band": ["x"]})
        flags = ["DPD_region=0.15"]          # region not in df
        assert _primary_mitigation_axis(df, dpd_eod_flags=flags) == "savings_band"

    def test_flag_eod_prefix_parsed_correctly(self):
        df = pd.DataFrame({"region": ["a"], "savings_band": ["b"]})
        flags = ["EOD_region=0.12"]
        # region is a demographic slice → should be chosen
        assert _primary_mitigation_axis(df, dpd_eod_flags=flags) == "region"

    def test_non_demographic_flag_returned_when_no_demographic_flag(self):
        df = pd.DataFrame({"income_band": ["a"], "savings_band": ["b"]})
        flags = ["EOD_income_band=0.20"]
        # income_band is financial, not demographic — still chosen over fallback
        assert _primary_mitigation_axis(df, dpd_eod_flags=flags) == "income_band"


# ---------------------------------------------------------------------------
# Tests: evaluate_bias
# ---------------------------------------------------------------------------

class TestEvaluateBias:
    def _run(self, n=200, green_idx=0, flip_fraction=0.0):
        """
        Build a minimal bias evaluation scenario.
        ``flip_fraction`` controls how many GREEN true-labels are predicted wrong
        to produce disparity.
        """
        np.random.seed(42)
        y_true = _make_labels(n, green_idx=green_idx)
        y_pred = y_true.copy()
        if flip_fraction > 0:
            green_pos = np.where(y_true == green_idx)[0]
            n_flip = int(len(green_pos) * flip_fraction)
            y_pred[green_pos[:n_flip]] = (green_idx + 1) % 3
        sens = _make_sensitive(n)
        return evaluate_bias(y_true, y_pred, sens, green_class_idx=green_idx)

    def test_returns_three_tuple(self):
        result = self._run()
        assert len(result) == 3, "evaluate_bias must return (metrics, bias_passed, flags)"

    def test_perfect_predictions_pass_gate(self):
        _, bias_passed, flags = self._run(flip_fraction=0.0)
        assert bias_passed is True
        assert flags == []

    def test_metrics_dict_has_aggregate_keys(self):
        metrics, _, _ = self._run()
        assert "aggregate_accuracy" in metrics
        assert "aggregate_f1" in metrics
        assert "aggregate_green_rate" in metrics

    def test_green_class_idx_respected(self):
        """
        When green_class_idx=2 the GREEN rate should reflect predictions==2,
        not predictions==0.
        """
        np.random.seed(0)
        n = 200
        y_true = np.tile([0, 1, 2], n // 3 + 1)[:n]
        y_pred = y_true.copy()
        sens = _make_sensitive(n)
        metrics_g0, _, _ = evaluate_bias(y_true, y_pred, sens, green_class_idx=0)
        metrics_g2, _, _ = evaluate_bias(y_true, y_pred, sens, green_class_idx=2)
        # green_rate for g0 should equal fraction where pred==0
        assert abs(metrics_g0["aggregate_green_rate"] - (y_pred == 0).mean()) < 1e-6
        assert abs(metrics_g2["aggregate_green_rate"] - (y_pred == 2).mean()) < 1e-6

    def test_flags_list_contains_string_identifiers(self):
        """Introduce a per-group F1 drop large enough to generate a flag."""
        np.random.seed(7)
        n = 300
        y_true = np.tile([0, 1, 2], n // 3)
        y_pred = y_true.copy()
        # Make model terrible only for "Unemployed" group
        sens = pd.DataFrame({"employment_status": np.where(np.arange(n) % 3 == 0, "Unemployed", "Employed")})
        # Flip all GREEN predictions for Unemployed
        unemployed_mask = sens["employment_status"] == "Unemployed"
        y_pred[unemployed_mask & (y_true == 0)] = 1
        _, _, flags = evaluate_bias(y_true, y_pred, sens, green_class_idx=0)
        assert isinstance(flags, list)
        # At least one flag should reference employment_status or a low-F1 issue
        assert any(isinstance(f, str) for f in flags)

    def test_no_scenarios_raw_still_works(self):
        """evaluate_bias should run with only demographic slices when scenarios_raw is None."""
        np.random.seed(1)
        n = 100
        y_true = _make_labels(n)
        y_pred = y_true.copy()
        sens = _make_sensitive(n)
        metrics, bias_passed, flags = evaluate_bias(y_true, y_pred, sens, green_class_idx=0)
        assert "aggregate_accuracy" in metrics


# ---------------------------------------------------------------------------
# Tests: detect_and_mitigate
# ---------------------------------------------------------------------------

class TestDetectAndMitigate:
    def _dummy_model(self, y_pred, n_classes=3):
        model = MagicMock()
        model.predict.return_value = y_pred
        proba = np.zeros((len(y_pred), n_classes))
        for i, c in enumerate(y_pred):
            proba[i, c] = 1.0
        model.predict_proba.return_value = proba
        return model

    def test_no_mitigation_when_gate_passes(self):
        """When bias_passed=True, the original model is returned unchanged."""
        np.random.seed(42)
        n = 200
        y = _make_labels(n)
        sens = _make_sensitive(n)
        model = self._dummy_model(y)

        result_model, bias_passed, metrics = detect_and_mitigate(
            model, MagicMock(), y, MagicMock(), y, y,
            sens, sens,
            green_class_idx=0,
        )
        assert bias_passed is True
        assert result_model is model

    def test_green_class_idx_propagated_to_evaluate_bias(self):
        """green_class_idx=2 must be threaded through without reverting to classes.min()."""
        np.random.seed(5)
        n = 150
        y = _make_labels(n, green_idx=2)
        sens = _make_sensitive(n)
        model = self._dummy_model(y)

        with patch("guards.bias_detection.evaluate_bias", wraps=__import__(
            "guards.bias_detection", fromlist=["evaluate_bias"]
        ).evaluate_bias) as mock_eval:
            detect_and_mitigate(
                model, MagicMock(), y, MagicMock(), y, y,
                sens, sens,
                green_class_idx=2,
            )
            call_kwargs = mock_eval.call_args
            assert call_kwargs.kwargs.get("green_class_idx") == 2
