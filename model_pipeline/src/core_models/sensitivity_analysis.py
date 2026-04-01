"""Utilities for Optuna-based hyperparameter sensitivity analysis."""

import json
import logging
import os
from typing import Any, Dict, List

import optuna

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def analyze_optuna_sensitivity(
    study: optuna.study.Study,
    model_name: str,
    output_dir: str,
    min_completed_trials: int = 10,
    top_k_params: int = 5,
) -> Dict[str, Any]:
    """
    Analyze Optuna study results for hyperparameter sensitivity.

    Produces:
    1) Ranked hyperparameter importance JSON.
    2) Importance bar plot.
    3) Parameter-vs-objective scatter plot for top-K numeric parameters.
    """
    if study is None:
        return {
            "status": "skipped",
            "reason": "missing_study",
            "trial_count": 0,
            "top_importances": [],
            "artifacts": [],
        }

    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(completed_trials) < min_completed_trials:
        logger.warning(
            "Skipping sensitivity analysis for %s: completed trials (%d) < min (%d)",
            model_name,
            len(completed_trials),
            min_completed_trials,
        )
        return {
            "status": "skipped",
            "reason": "insufficient_trials",
            "trial_count": len(completed_trials),
            "top_importances": [],
            "artifacts": [],
        }

    os.makedirs(output_dir, exist_ok=True)

    importances = optuna.importance.get_param_importances(study)
    top_items = list(importances.items())[:top_k_params]

    if not top_items:
        return {
            "status": "skipped",
            "reason": "empty_importance",
            "trial_count": len(completed_trials),
            "top_importances": [],
            "artifacts": [],
        }

    top_importances = [
        {"param": param_name, "importance": float(importance)}
        for param_name, importance in top_items
    ]

    safe_model_name = model_name.replace("/", "_")
    artifacts: List[str] = []

    json_path = os.path.join(output_dir, f"{safe_model_name}_param_importance.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": model_name,
                "study_name": study.study_name,
                "completed_trials": len(completed_trials),
                "top_importances": top_importances,
            },
            handle,
            indent=2,
        )
    artifacts.append(json_path)

    importance_plot_path = os.path.join(
        output_dir, f"{safe_model_name}_param_importance.png"
    )
    _plot_importances(top_items, importance_plot_path, model_name)
    artifacts.append(importance_plot_path)

    sensitivity_plot_path = os.path.join(
        output_dir, f"{safe_model_name}_param_sensitivity.png"
    )
    plotted = _plot_param_sensitivity(
        completed_trials,
        [param for param, _ in top_items],
        sensitivity_plot_path,
        model_name,
    )
    if plotted:
        artifacts.append(sensitivity_plot_path)

    return {
        "status": "ok",
        "reason": "completed",
        "trial_count": len(completed_trials),
        "top_importances": top_importances,
        "artifacts": artifacts,
        "study_name": study.study_name,
    }


def _plot_importances(top_items, output_path: str, model_name: str) -> None:
    labels = [param_name for param_name, _ in top_items]
    values = [float(score) for _, score in top_items]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels[::-1], values[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Hyperparameter Importance ({model_name})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_param_sensitivity(
    completed_trials: list,
    params: List[str],
    output_path: str,
    model_name: str,
) -> bool:
    series = []
    for param_name in params:
        xs = []
        ys = []
        for trial in completed_trials:
            value = trial.params.get(param_name)
            if isinstance(value, (int, float)) and trial.value is not None:
                xs.append(float(value))
                ys.append(float(trial.value))
        if xs:
            series.append((param_name, xs, ys))

    if not series:
        return False

    fig, axes = plt.subplots(len(series), 1, figsize=(8, 3 * len(series)))
    if len(series) == 1:
        axes = [axes]

    for ax, (param_name, xs, ys) in zip(axes, series):
        ax.scatter(xs, ys, alpha=0.65, s=22)
        ax.set_xlabel(param_name)
        ax.set_ylabel("Validation F1")
        ax.set_title(f"{model_name}: F1 vs {param_name}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True
