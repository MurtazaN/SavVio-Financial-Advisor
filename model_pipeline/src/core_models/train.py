"""
Model Training Module.

Trains a model of the specified type using provided hyperparameters.
Supports: xgboost, lightgbm, xgb_linear.

Changes from v1:
    - Parameter filtering per model type (invalid params are dropped with a warning).
    - eval_set + early stopping support (pass X_val/y_val to prevent overfitting).
    - Verbose suppression for LightGBM and XGBoost.
    - Random state sourced from Config instead of hardcoded.
"""

import logging

import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.sklearn
import pandas as pd
from typing import Any, Dict, Optional, Set
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Valid parameters per model type — anything not listed here is filtered out.
# ---------------------------------------------------------------------------

VALID_PARAMS: Dict[str, Set[str]] = {
    "xgboost": {
        "max_depth", "learning_rate", "n_estimators", "subsample",
        "colsample_bytree", "reg_alpha", "reg_lambda", "min_child_weight",
        "gamma", "scale_pos_weight",
    },
    "xgb_linear": {
        "learning_rate", "n_estimators", "reg_alpha", "reg_lambda",
    },
    "lightgbm": {
        "max_depth", "learning_rate", "n_estimators", "num_leaves",
        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
        "min_child_samples",
    },
}


def _filter_params(model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only parameters valid for the given model type.
    Logs a warning for every dropped parameter so config mistakes are visible.
    """
    valid = VALID_PARAMS.get(model_type, set())
    filtered = {}
    for k, v in params.items():
        if k in valid:
            filtered[k] = v
        else:
            logger.warning("Dropped invalid param '%s' for model type '%s'", k, model_type)
    return filtered


def train_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    random_state: int = Config.RANDOM_STATE,
):
    """
    Train a model of the specified type.

    Args:
        model_type:   One of 'xgboost', 'lightgbm', 'xgb_linear'.
        X_train:      Training features.
        y_train:      Training labels.
        params:       Hyperparameters (invalid ones are filtered automatically).
        X_val:        Validation features (enables early stopping for tree models).
        y_val:        Validation labels.
        random_state: Seed for reproducibility.

    Returns:
        Trained model instance.
    """
    logger.info("Training %s...", model_type)
    filtered = _filter_params(model_type, params)

    # Whether we can use early stopping (need a validation set + tree model).
    use_early_stop = X_val is not None and y_val is not None

    if model_type == "xgboost":
        model = XGBClassifier(
            **filtered,
            random_state=random_state,
            verbosity=0,
            eval_metric="mlogloss",
        )
        fit_kwargs = {}
        if use_early_stop:
            model.set_params(early_stopping_rounds=10)
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
        model.fit(X_train, y_train, **fit_kwargs)

    elif model_type == "xgb_linear":
        model = XGBClassifier(
            booster="gblinear",
            **filtered,
            random_state=random_state,
            verbosity=0,
            eval_metric="mlogloss",
        )
        fit_kwargs = {}
        if use_early_stop:
            model.set_params(early_stopping_rounds=10)
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
        model.fit(X_train, y_train, **fit_kwargs)

    elif model_type == "lightgbm":
        model = LGBMClassifier(
            **filtered,
            random_state=random_state,
            verbose=-1,
        )
        fit_kwargs = {}
        if use_early_stop:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["eval_metric"] = "multi_logloss"
            fit_kwargs["callbacks"] = [
                __import__("lightgbm").early_stopping(10, verbose=False),
                __import__("lightgbm").log_evaluation(period=0),
            ]
        model.fit(X_train, y_train, **fit_kwargs)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    logger.info("Training complete for %s.", model_type)
    return model


def log_model_to_mlflow(model, model_type: str, signature):
    """
    Log the trained model to the active MLflow run.
    Uses the appropriate MLflow flavor based on model type.
    """
    try:
        if model_type in ("xgboost", "xgb_linear"):
            mlflow.xgboost.log_model(model, "model", signature=signature)
        elif model_type == "lightgbm":
            mlflow.lightgbm.log_model(model, "model", signature=signature)
        else:
            mlflow.sklearn.log_model(model, "model", signature=signature)
        logger.info("Model logged to MLflow: %s", model_type)
    except Exception as e:
        logger.error("Failed to log model to MLflow: %s", e, exc_info=True)