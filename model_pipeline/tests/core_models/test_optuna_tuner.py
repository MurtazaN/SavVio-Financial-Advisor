"""Unit tests for hyperparameter tuning module (optuna_tuner.py).

All Optuna studies, MLflow runs, and model training are mocked so tests
run fast without real training or a tracking server.
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, ANY

from core_models.optuna_tuner import (
    tune_model,
    tune_best_candidate,
    _xgboost_objective,
    _lightgbm_objective,
    _xgb_linear_objective,
    _OBJECTIVES,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 5))
    y = rng.integers(0, 3, 30)
    return X[:20], y[:20], X[20:], y[20:]


@pytest.fixture
def candidates():
    return [
        {"name": "xgboost",    "metrics": {"f1_score": 0.72}, "bias_passed": True},
        {"name": "lightgbm",   "metrics": {"f1_score": 0.68}, "bias_passed": True},
        {"name": "xgb_linear", "metrics": {"f1_score": 0.65}, "bias_passed": False},
    ]


@pytest.fixture
def mock_study():
    study = MagicMock()
    study.best_value = 0.80
    study.best_params = {"max_depth": 4, "learning_rate": 0.1}
    study.best_trial.number = 3
    study.study_name = "xgboost_tuning"
    return study


# ---------------------------------------------------------------------------
# _OBJECTIVES registry
# ---------------------------------------------------------------------------

class TestObjectivesRegistry:

    def test_all_three_model_types_registered(self):
        assert "xgboost"    in _OBJECTIVES
        assert "lightgbm"   in _OBJECTIVES
        assert "xgb_linear" in _OBJECTIVES

    def test_all_values_are_callable(self):
        for name, fn in _OBJECTIVES.items():
            assert callable(fn), f"{name} objective is not callable"


# ---------------------------------------------------------------------------
# tune_model — now returns (params, score, study)
# ---------------------------------------------------------------------------

class TestTuneModel:

    @patch("core_models.optuna_tuner.mlflow")
    @patch("core_models.optuna_tuner.optuna.create_study")
    def test_returns_tuple_of_params_and_score(
        self, mock_create_study, mock_mlflow, small_dataset, mock_study
    ):
        mock_create_study.return_value = mock_study
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        X_tr, y_tr, X_val, y_val = small_dataset
        params, score, study = tune_model("xgboost", X_tr, y_tr, X_val, y_val, n_trials=2, timeout=10)

        assert isinstance(params, dict)
        assert isinstance(score, float)

    @patch("core_models.optuna_tuner.mlflow")
    @patch("core_models.optuna_tuner.optuna.create_study")
    def test_returns_best_params_from_study(
        self, mock_create_study, mock_mlflow, small_dataset, mock_study
    ):
        mock_create_study.return_value = mock_study
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        X_tr, y_tr, X_val, y_val = small_dataset
        params, score, study = tune_model("xgboost", X_tr, y_tr, X_val, y_val, n_trials=2, timeout=10)

        assert params == mock_study.best_params
        assert score == mock_study.best_value

    @patch("core_models.optuna_tuner.mlflow")
    @patch("core_models.optuna_tuner.optuna.create_study")
    def test_unsupported_model_type_raises(
        self, mock_create_study, mock_mlflow, small_dataset
    ):
        X_tr, y_tr, X_val, y_val = small_dataset
        with pytest.raises(ValueError, match="Tuning not supported"):
            tune_model("random_forest", X_tr, y_tr, X_val, y_val)

    @patch("core_models.optuna_tuner.mlflow")
    @patch("core_models.optuna_tuner.optuna.create_study")
    def test_study_created_with_maximize_direction(
        self, mock_create_study, mock_mlflow, small_dataset, mock_study
    ):
        mock_create_study.return_value = mock_study
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        X_tr, y_tr, X_val, y_val = small_dataset
        tune_model("xgboost", X_tr, y_tr, X_val, y_val, n_trials=2, timeout=10)

        _, kwargs = mock_create_study.call_args
        assert kwargs["direction"] == "maximize"

    @patch("core_models.optuna_tuner.mlflow")
    @patch("core_models.optuna_tuner.optuna.create_study")
    def test_mlflow_best_score_logged(
        self, mock_create_study, mock_mlflow, small_dataset, mock_study
    ):
        mock_create_study.return_value = mock_study
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        X_tr, y_tr, X_val, y_val = small_dataset
        tune_model("xgboost", X_tr, y_tr, X_val, y_val, n_trials=2, timeout=10)

        mock_mlflow.log_metric.assert_any_call("best_val_f1_weighted", mock_study.best_value)

    @patch("core_models.optuna_tuner.mlflow")
    @patch("core_models.optuna_tuner.optuna.create_study")
    def test_all_three_model_types_accepted(
        self, mock_create_study, mock_mlflow, small_dataset, mock_study
    ):
        mock_create_study.return_value = mock_study
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        X_tr, y_tr, X_val, y_val = small_dataset
        for model_type in ("xgboost", "lightgbm", "xgb_linear"):
            params, score, study = tune_model(model_type, X_tr, y_tr, X_val, y_val, n_trials=1, timeout=5)
            assert isinstance(params, dict)


# ---------------------------------------------------------------------------
# tune_best_candidate — now returns (model_type, params, score, study)
# ---------------------------------------------------------------------------

class TestTuneBestCandidate:

    @patch("core_models.optuna_tuner.tune_model")
    @patch("core_models.optuna_tuner.Config")
    def test_returns_none_when_tuning_disabled(
        self, mock_config, mock_tune_model, small_dataset, candidates
    ):
        mock_config.TUNING_BACKEND = "none"
        X_tr, y_tr, X_val, y_val = small_dataset
        result = tune_best_candidate(candidates, X_tr, y_tr, X_val, y_val)
        assert result is None
        mock_tune_model.assert_not_called()

    @patch("core_models.optuna_tuner.tune_model")
    @patch("core_models.optuna_tuner.Config")
    def test_returns_none_when_no_tunable_candidates(
        self, mock_config, mock_tune_model, small_dataset
    ):
        mock_config.TUNING_BACKEND = "optuna"
        non_tunable = [{"name": "random_forest", "metrics": {"f1_score": 0.75}, "bias_passed": True}]
        X_tr, y_tr, X_val, y_val = small_dataset
        result = tune_best_candidate(non_tunable, X_tr, y_tr, X_val, y_val)
        assert result is None

    @patch("core_models.optuna_tuner.tune_model")
    @patch("core_models.optuna_tuner.Config")
    def test_tunes_highest_f1_candidate(
        self, mock_config, mock_tune_model, small_dataset, candidates
    ):
        mock_config.TUNING_BACKEND = "optuna"
        mock_tune_model.return_value = ({"max_depth": 4}, 0.80, MagicMock())

        X_tr, y_tr, X_val, y_val = small_dataset
        tune_best_candidate(candidates, X_tr, y_tr, X_val, y_val)

        args = mock_tune_model.call_args[0]
        assert args[0] == "xgboost"

    @patch("core_models.optuna_tuner.tune_model")
    @patch("core_models.optuna_tuner.Config")
    def test_returns_model_type_params_and_score(
        self, mock_config, mock_tune_model, small_dataset, candidates
    ):
        mock_config.TUNING_BACKEND = "optuna"
        mock_tune_model.return_value = ({"max_depth": 4}, 0.80, MagicMock())

        X_tr, y_tr, X_val, y_val = small_dataset
        result = tune_best_candidate(candidates, X_tr, y_tr, X_val, y_val)

        assert result is not None
        model_type, params, score, study = result
        assert model_type == "xgboost"
        assert params == {"max_depth": 4}
        assert score == 0.80

    @patch("core_models.optuna_tuner.tune_model")
    @patch("core_models.optuna_tuner.Config")
    def test_skips_non_tunable_candidates(
        self, mock_config, mock_tune_model, small_dataset
    ):
        mock_config.TUNING_BACKEND = "optuna"
        mock_tune_model.return_value = ({"max_depth": 3}, 0.75, MagicMock())

        mixed = [
            {"name": "random_forest", "metrics": {"f1_score": 0.90}, "bias_passed": True},
            {"name": "xgboost",       "metrics": {"f1_score": 0.72}, "bias_passed": True},
        ]
        X_tr, y_tr, X_val, y_val = small_dataset
        result = tune_best_candidate(mixed, X_tr, y_tr, X_val, y_val)

        # random_forest is not tunable so xgboost should be picked
        model_type, _, _, _ = result
        assert model_type == "xgboost"