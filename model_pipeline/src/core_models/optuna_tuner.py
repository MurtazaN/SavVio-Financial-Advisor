"""
Hyperparameter Tuning with Optuna.

Bayesian optimization for XGBoost, LightGBM, and XGB-Linear using
Optuna's TPE sampler with MedianPruner for early trial termination.

Each trial is logged as an MLflow nested child run.
The best hyperparameters are returned for final model training.

Integration with run_pipeline.py:
    After baseline training, call tune_best_candidate() with the
    best-performing model type. It returns optimized params that you
    use to train a final tuned model and add it to the candidate list.

Usage:
    from tuning.optuna_tuner import tune_best_candidate

    best_params, best_score = tune_best_candidate(
        model_type="xgboost",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
    )
"""

import logging
from typing import Any, Dict, Optional, Tuple

import optuna
import mlflow
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import Config

logger = logging.getLogger(__name__)

# Suppress Optuna's per-trial INFO logs — summary is logged at the end.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Search space definitions (one per model type)
# ---------------------------------------------------------------------------

def _xgboost_objective(trial, X_train, y_train, X_val, y_val):
    """Objective function for XGBoost tree booster."""
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    model = XGBClassifier(
        **params,
        random_state=Config.RANDOM_STATE,
        verbosity=0,
        eval_metric="mlogloss",
        early_stopping_rounds=10,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average="weighted")


def _lightgbm_objective(trial, X_train, y_train, X_val, y_val):
    """Objective function for LightGBM."""
    import lightgbm as lgb

    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    model = LGBMClassifier(
        **params,
        random_state=Config.RANDOM_STATE,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(10, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average="weighted")


def _xgb_linear_objective(trial, X_train, y_train, X_val, y_val):
    """Objective function for XGBoost linear booster."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    model = XGBClassifier(
        booster="gblinear",
        **params,
        random_state=Config.RANDOM_STATE,
        verbosity=0,
        eval_metric="mlogloss",
        early_stopping_rounds=10,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average="weighted")


# Map model type to its objective function.
_OBJECTIVES = {
    "xgboost": _xgboost_objective,
    "lightgbm": _lightgbm_objective,
    "xgb_linear": _xgb_linear_objective,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tune_model(
    model_type: str,
    X_train, y_train,
    X_val, y_val,
    n_trials: int = None,
    timeout: int = None,
) -> Tuple[Dict[str, Any], float, optuna.study.Study]:
    """
    Run Optuna hyperparameter optimization for a single model type.

    Creates a study with TPE sampler (Bayesian) and MedianPruner
    (kills trials performing below the median of completed trials).
    Each trial is logged to MLflow as a nested run under a tuning parent run.

    Args:
        model_type: One of 'xgboost', 'lightgbm', 'xgb_linear'.
        X_train:    Training features.
        y_train:    Training labels.
        X_val:      Validation features.
        y_val:      Validation labels.
        n_trials:   Max number of trials. Defaults to Config.N_TUNING_TRIALS.
        timeout:    Max seconds for the study. Defaults to Config.TUNING_TIMEOUT_SECONDS.

    Returns:
        Tuple of (best_params dict, best_f1_score float, optuna study).

    Raises:
        ValueError: If model_type is not supported for tuning.
    """
    if model_type not in _OBJECTIVES:
        supported = list(_OBJECTIVES.keys())
        raise ValueError(
            f"Tuning not supported for '{model_type}'. Supported: {supported}"
        )

    n_trials = n_trials or Config.N_TUNING_TRIALS
    timeout = timeout or Config.TUNING_TIMEOUT_SECONDS

    objective_fn = _OBJECTIVES[model_type]

    # Wrap the objective so Optuna can call it with just (trial).
    # We explicitly create nested MLflow runs here (parent run around the
    # study, child run per trial) instead of relying on callback internals.
    def objective(trial):
        with mlflow.start_run(
            run_name=f"{model_type}_trial_{trial.number:03d}", nested=True,
        ):
            mlflow.log_param("model_type", model_type)
            score = objective_fn(trial, X_train, y_train, X_val, y_val)
            mlflow.log_params(trial.params)
            mlflow.log_metric("val_f1_weighted", score)
            mlflow.set_tag("optuna_trial_number", trial.number)
            return score

    study = optuna.create_study(
        study_name=f"{model_type}_tuning",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=Config.RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    logger.info(
        "Starting Optuna study for %s — %d trials, %ds timeout",
        model_type, n_trials, timeout,
    )

    with mlflow.start_run(run_name=f"{model_type}_tuning"):
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        mlflow.log_param("study_name", study.study_name)
        mlflow.log_param("n_trials_requested", n_trials)
        mlflow.log_param("timeout_seconds", timeout)
        mlflow.log_metric("best_val_f1_weighted", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})

    logger.info(
        "Optuna study complete for %s — best F1: %.4f in trial %d",
        model_type, study.best_value, study.best_trial.number,
    )
    logger.info("Best params: %s", study.best_params)

    return study.best_params, study.best_value, study


def tune_best_candidate(
    candidates: list,
    X_train, y_train,
    X_val, y_val,
) -> Optional[Tuple[str, Dict[str, Any], float, optuna.study.Study]]:
    """
    Identify the best baseline candidate that supports tuning,
    then optimize its hyperparameters.

    Skips candidates that don't support tuning.

    Args:
        candidates: List of candidate dicts from train_candidates()
                    (each has 'name', 'metrics', 'bias_passed').
        X_train:    Training features.
        y_train:    Training labels.
        X_val:      Validation features.
        y_val:      Validation labels.

    Returns:
        Tuple of (model_type, best_params, best_f1, study) or None if tuning is
        disabled or no tunable candidate exists.
    """
    if Config.TUNING_BACKEND == "none":
        logger.info("Tuning disabled (TUNING_BACKEND='none').")
        return None

    # Filter to candidates that have an Optuna objective defined.
    tunable = [c for c in candidates if c["name"] in _OBJECTIVES]
    if not tunable:
        logger.warning("No tunable candidates found. Skipping tuning.")
        return None

    # Pick the best tunable candidate by validation F1.
    best_baseline = max(tunable, key=lambda c: c["metrics"]["f1_score"])
    model_type = best_baseline["name"]

    logger.info(
        "Tuning best baseline: %s (baseline F1: %.4f)",
        model_type, best_baseline["metrics"]["f1_score"],
    )

    best_params, best_f1, study = tune_model(
        model_type, X_train, y_train, X_val, y_val,
    )

    improvement = best_f1 - best_baseline["metrics"]["f1_score"]
    logger.info(
        "Tuning result — F1: %.4f (improvement: %+.4f over baseline)",
        best_f1, improvement,
    )

    return model_type, best_params, best_f1, study
