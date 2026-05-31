"""
Model Pipeline — End-to-End Entrypoint.

Orchestrates the full model training flow:
    1. Initialization   — Seeds, MLflow config.
    2. Data Preparation  — Feature engineering, label encoding, 3-way split.
    3. Baseline Training — Train XGBoost, LightGBM, XGB-Linear, Logistic Regression.
    4. Model Selection   — Best F1 + bias gate (when bias module is ready).
    5. Final Evaluation  — Held-out test set, all visualizations, registry prep.

Usage:
    python run_pipeline.py
"""

import os
import logging
import joblib
import numpy as np
from datetime import datetime

import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from config import Config
from features.feature_engineering import build_training_data
from core_models.train import train_model, log_model_to_mlflow
from core_models.evaluate import evaluate_model
from core_models.optuna_tuner import tune_best_candidate
from core_models.sensitivity_analysis import analyze_optuna_sensitivity
from llm.prompt_engin import apply_llm_guardrails
from llm.config import LLMConfig

logger = logging.getLogger(__name__)


def write_evaluation_summary_md(candidates, best, final_metrics, output_path, sensitivity_summary=None):
    """Write a simple markdown summary for 3 baseline models + champion."""
    baseline_names = {"xgboost", "lightgbm", "xgb_linear"}
    baseline_rows = [c for c in candidates if c.get("name") in baseline_names]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        "# Evaluation Summary",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Baseline Models (Validation Metrics)",
        "",
    ]

    if not baseline_rows:
        lines.append("No baseline model metrics found.")
        lines.append("")
    else:
        for row in baseline_rows:
            lines.append(f"### {row['name']}")
            lines.append(f"- run_id: {row['run_id']}")
            for metric_name, metric_value in row["metrics"].items():
                lines.append(f"- {metric_name}: {metric_value}")
            lines.append("")

    lines.extend([
        "## Champion Model (Final Test Metrics)",
        "",
    ])

    if best is None or final_metrics is None:
        lines.append("Champion metrics are unavailable (model not selected or final evaluation failed).")
    else:
        lines.append(f"- model_name: {best['name']}")
        lines.append(f"- source_run_id: {best['run_id']}")
        for metric_name, metric_value in final_metrics.items():
            lines.append(f"- {metric_name}: {metric_value}")

    lines.extend([
        "",
        "## Hyperparameter Sensitivity (Tuned Champion)",
        "",
    ])

    if not sensitivity_summary:
        lines.append("Sensitivity analysis was not executed.")
    elif sensitivity_summary.get("status") != "ok":
        lines.append(
            f"Sensitivity analysis skipped: {sensitivity_summary.get('reason', 'unknown_reason')}"
        )
        lines.append(f"- completed_trials: {sensitivity_summary.get('trial_count', 0)}")
    else:
        lines.append(f"- study_name: {sensitivity_summary.get('study_name', 'unknown')}")
        lines.append(f"- completed_trials: {sensitivity_summary.get('trial_count', 0)}")
        lines.append("- top_hyperparameters:")
        for item in sensitivity_summary.get("top_importances", []):
            lines.append(f"  - {item['param']}: {item['importance']:.6f}")
        if sensitivity_summary.get("artifacts"):
            lines.append("- artifact_paths:")
            for artifact_path in sensitivity_summary["artifacts"]:
                lines.append(f"  - {artifact_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Saved evaluation summary markdown: %s", output_path)


# ---------------------------------------------------------------------------
# Bias detection — placeholder until module is complete.
# ---------------------------------------------------------------------------
try:
    from guards.bias_detection import (
        attach_savings_band_to_sensitive,
        detect_and_mitigate,
        predict_with_sensitive_features,
    )
    BIAS_AVAILABLE = True
except ImportError:
    logger.warning("Bias detection module not available — skipping bias checks.")
    BIAS_AVAILABLE = False

    def attach_savings_band_to_sensitive(sens_df, scenarios_df):
        return sens_df

    def predict_with_sensitive_features(model, X, sensitive=None):
        return model.predict(X)

# Temporarily force-disable bias checks until bias module contract is finalized.
BIAS_AVAILABLE = True
# ---------------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------------

def initialize():
    """Set global seeds and configure MLflow."""
    os.environ["PYTHONHASHSEED"] = str(Config.RANDOM_STATE)
    np.random.seed(Config.RANDOM_STATE)

    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s", Config.MLFLOW_TRACKING_URI)
    logger.info("MLflow experiment: %s", Config.EXPERIMENT_NAME)


# ---------------------------------------------------------------------------
# 2. Data Preparation
# ---------------------------------------------------------------------------


def prepare_data():
    """
    Build feature matrix, encode labels, and perform 3-way stratified split.

    Returns:
        dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                        sens_train, sens_val, sens_test, label_encoder, scenarios_raw,
                        scenarios_val (same rows as X_val / sens_val for bias checks)
    """
    # Feature engineering + deterministic labeling (GREEN/YELLOW/RED).
    X, y_raw, scenarios_raw = build_training_data(is_training=True)

    # Remove synthetic rows (injected by augment_near_zero_savings) before splitting.
    # The docstring for augment_near_zero_savings requires these be excluded from
    # any held-out evaluation set; without this filter they contaminate val/test
    # bias metrics for the savings_band slice.
    if "synthetic_flag" in scenarios_raw.columns:
        real_mask = (scenarios_raw["synthetic_flag"] != 1).values
        n_synthetic = int((~real_mask).sum())
        if n_synthetic > 0:
            logger.info("Removing %d synthetic rows before train/val/test split.", n_synthetic)
            scenarios_raw = scenarios_raw[real_mask].reset_index(drop=True)
            if hasattr(X, "reset_index"):
                X = X[real_mask].reset_index(drop=True)
            else:
                X = X[real_mask]
            y_raw = y_raw[real_mask].reset_index(drop=True)

    # Encode string labels into integers for model training.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    logger.info("Label encoding: %s", dict(zip(
        label_encoder.classes_, label_encoder.transform(label_encoder.classes_)
    )))
    logger.info("Label distribution:\n%s", y_raw.value_counts().to_string())

    # Extract sensitive features for bias detection.
    sensitive_cols = [c for c in Config.SENSITIVE_FEATURES if c in scenarios_raw.columns]
    sensitive_features = scenarios_raw[sensitive_cols] if sensitive_cols else None

    # --- 3-way stratified split: train (60%) / val (20%) / test (20%) ---
    # Split scenarios_raw (and sensitive cols) in the SAME calls as X, y so row
    # alignment is guaranteed for post-training bias detection on the val set.
    # First split: separate test set (20%).
    if sensitive_features is not None:
        X_temp, X_test, y_temp, y_test, scenarios_temp, scenarios_test, sens_temp, sens_test = train_test_split(
            X, y, scenarios_raw, sensitive_features,
            test_size=0.2,
            random_state=Config.RANDOM_STATE,
            stratify=y,
        )
    else:
        X_temp, X_test, y_temp, y_test, scenarios_temp, scenarios_test = train_test_split(
            X, y, scenarios_raw,
            test_size=0.2,
            random_state=Config.RANDOM_STATE,
            stratify=y,
        )
        sens_temp, sens_test = None, None

    # Second split: separate validation from training (25% of remaining = 20% of total).
    if sens_temp is not None:
        X_train, X_val, y_train, y_val, scenarios_train, scenarios_val, sens_train, sens_val = train_test_split(
            X_temp, y_temp, scenarios_temp, sens_temp,
            test_size=0.25,
            random_state=Config.RANDOM_STATE,
            stratify=y_temp,
        )
    else:
        X_train, X_val, y_train, y_val, scenarios_train, scenarios_val = train_test_split(
            X_temp, y_temp, scenarios_temp,
            test_size=0.25,
            random_state=Config.RANDOM_STATE,
            stratify=y_temp,
        )
        sens_train, sens_val = None, None

    # Derive savings_band from scenario financial columns (for ThresholdOptimizer axis).
    sens_train = attach_savings_band_to_sensitive(sens_train, scenarios_train)
    sens_val = attach_savings_band_to_sensitive(sens_val, scenarios_val)
    sens_test = attach_savings_band_to_sensitive(sens_test, scenarios_test)

    logger.info("Split sizes — train: %d, val: %d, test: %d", len(y_train), len(y_val), len(y_test))

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "sens_train": sens_train, "sens_val": sens_val, "sens_test": sens_test,
        "label_encoder": label_encoder,
        "scenarios_raw": scenarios_raw,
        "scenarios_val": scenarios_val,
    }


# ---------------------------------------------------------------------------
# 3. Train Candidates
# ---------------------------------------------------------------------------

def train_candidates(
    X_train, y_train, X_val, y_val,
    sens_train, sens_val, label_encoder,
    scenarios_val=None,
):
    """
    Train all model candidates, evaluate on validation set, run bias detection
    and optional ThresholdOptimizer mitigation (detect_and_mitigate).

    Args:
        scenarios_val: Raw scenario rows aligned with X_val / y_val (for slice derivation).

    Returns:
        List of dicts, each with: name, model, run_id, metrics, bias_passed.
    """
    models_to_train = [
        {"name": "xgboost",     "params": {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100}},
        {"name": "lightgbm",    "params": {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100}},
        {"name": "xgb_linear",  "params": {"learning_rate": 0.1, "n_estimators": 100}},
    ]

    candidates = []

    for model_config in models_to_train:
        model_type = model_config["name"]
        params = model_config["params"]

        logger.info("=" * 50)
        logger.info("Training candidate: %s", model_type)
        logger.info("=" * 50)

        try:
            with mlflow.start_run(run_name=f"{model_type}_baseline"):

                # Log metadata.
                mlflow.log_param("model_type", model_type)
                mlflow.log_params(params)
                mlflow.log_param("n_scenarios", Config.N_SCENARIOS)
                mlflow.log_param("label_type", "deterministic_engine_GYR")
                mlflow.log_param("num_classes", len(label_encoder.classes_))

                # Train.
                model = train_model(
                    model_type, X_train, y_train, params,
                    X_val=X_val, y_val=y_val,
                )

                # Evaluate on VALIDATION set (not test).
                metrics = evaluate_model(
                    model, X_val, y_val,
                    label_names=list(label_encoder.classes_),
                )

                # Bias detection — placeholder until module is complete.
                
                bias_passed = True
                if BIAS_AVAILABLE and sens_val is not None and sens_train is not None:
                    y_pred_val = model.predict(X_val)
                    y_prob_val = (
                        model.predict_proba(X_val)
                        if hasattr(model, "predict_proba")
                        else None
                    )
                    green_class_idx = int(label_encoder.transform(["GREEN"])[0])
                    model, bias_passed, fairness_metrics = detect_and_mitigate(
                        model,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        y_pred_val,
                        sens_train,
                        sens_val,
                        scenarios_raw=scenarios_val,
                        y_prob_val=y_prob_val,
                        green_class_idx=green_class_idx,
                    )
                    mlflow.log_metric("bias_gate_passed", int(bias_passed))

                # Log model artifact (possibly mitigated).
                signature = infer_signature(
                    X_train,
                    predict_with_sensitive_features(model, X_train, sens_train),
                )
                log_model_to_mlflow(model, model_type, signature)

                # Log scenario artifact for reproducibility.
                if os.path.exists(Config.SCENARIO_OUTPUT_PATH):
                    mlflow.log_artifact(Config.SCENARIO_OUTPUT_PATH, "data")

                run_id = mlflow.active_run().info.run_id

                candidates.append({
                    "name": model_type,
                    "model": model,
                    "run_id": run_id,
                    "metrics": metrics,
                    "bias_passed": bias_passed,
                    "fairness_metrics": fairness_metrics if BIAS_AVAILABLE else {},
                })
                logger.info("%s — val F1: %.4f, bias passed: %s",
                            model_type, metrics["f1_score"], bias_passed)

        except Exception as e:
            logger.error("Candidate %s failed: %s", model_type, e, exc_info=True)
            continue

    return candidates

# ---------------------------------------------------------------------------
# 3b. Hyperparameter Tuning
# ---------------------------------------------------------------------------

def tune_candidate(candidates, data):
    # 3b. Hyperparameter tuning on best baseline.
    tuning_context = None
    tuning_result = tune_best_candidate(
        candidates,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
    )
    if tuning_result:
        model_type, tuned_params, _, tuning_study = tuning_result
        tuning_context = {"model_type": model_type, "study": tuning_study}
        try:
            with mlflow.start_run(run_name=f"{model_type}_tuned"):
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("tuning_source", "optuna")
                mlflow.log_params(tuned_params)

                tuned_model = train_model(
                    model_type, data["X_train"], data["y_train"], tuned_params,
                    X_val=data["X_val"], y_val=data["y_val"],
                )
                tuned_metrics = evaluate_model(
                    tuned_model, data["X_val"], data["y_val"],
                    label_names=list(data["label_encoder"].classes_),
                )

                bias_passed = True
                if BIAS_AVAILABLE and data.get("sens_val") is not None and data.get(
                    "sens_train"
                ) is not None:
                    y_pred_val = tuned_model.predict(data["X_val"])
                    y_prob_val = (
                        tuned_model.predict_proba(data["X_val"])
                        if hasattr(tuned_model, "predict_proba")
                        else None
                    )
                    green_class_idx = int(data["label_encoder"].transform(["GREEN"])[0])
                    tuned_model, bias_passed, fairness_metrics = detect_and_mitigate(
                        tuned_model,
                        data["X_train"],
                        data["y_train"],
                        data["X_val"],
                        data["y_val"],
                        y_pred_val,
                        data["sens_train"],
                        data["sens_val"],
                        scenarios_raw=data.get("scenarios_val"),
                        y_prob_val=y_prob_val,
                        green_class_idx=green_class_idx,
                    )
                    mlflow.log_metric("bias_gate_passed", int(bias_passed))

                signature = infer_signature(
                    data["X_train"],
                    predict_with_sensitive_features(
                        tuned_model, data["X_train"], data["sens_train"]
                    ),
                )
                log_model_to_mlflow(tuned_model, model_type, signature)

                candidates.append({
                    "name": f"{model_type}_tuned",
                    "model": tuned_model,
                    "run_id": mlflow.active_run().info.run_id,
                    "metrics": tuned_metrics,
                    "bias_passed": bias_passed,
                    "tuning_study": tuning_study,
                    "fairness_metrics": fairness_metrics if BIAS_AVAILABLE else {},
                })
        except Exception as e:
            logger.error("Tuned model training failed: %s", e, exc_info=True)

    return candidates, tuning_context


def run_sensitivity_analysis(best, tuning_context):
    """Run Optuna-based sensitivity analysis for the tuned champion only."""
    if best is None:
        return {"status": "skipped", "reason": "no_champion", "trial_count": 0}

    if not Config.SENSITIVITY_ANALYSIS_ENABLED:
        return {"status": "skipped", "reason": "disabled_in_config", "trial_count": 0}

    if not best["name"].endswith("_tuned"):
        return {
            "status": "skipped",
            "reason": "champion_is_not_tuned",
            "trial_count": 0,
        }

    if not tuning_context or tuning_context.get("study") is None:
        return {"status": "skipped", "reason": "missing_tuning_study", "trial_count": 0}

    if not best["name"].startswith(tuning_context.get("model_type", "")):
        return {
            "status": "skipped",
            "reason": "study_champion_mismatch",
            "trial_count": 0,
        }

    output_dir = os.path.join(Config.BASE_DIR, "reports", "sensitivity")
    try:
        return analyze_optuna_sensitivity(
            study=tuning_context["study"],
            model_name=best["name"],
            output_dir=output_dir,
            min_completed_trials=Config.SENSITIVITY_MIN_COMPLETED_TRIALS,
            top_k_params=Config.SENSITIVITY_TOP_K_PARAMS,
        )
    except Exception as exc:
        logger.warning("Sensitivity analysis failed: %s", exc, exc_info=True)
        return {"status": "skipped", "reason": "runtime_error", "trial_count": 0}

# ---------------------------------------------------------------------------
# 4. Model Selection
# ---------------------------------------------------------------------------

def select_best_model(candidates):
    """
    Select the best model: must pass bias gate, then rank by F1.

    Returns:
        The winning candidate dict, or None if no candidates pass.
    """
    if not candidates:
        logger.error("No candidates to select from.")
        return None

    # Filter by bias gate.
    eligible = [c for c in candidates if c["bias_passed"]]
    if not eligible:
        logger.warning("No candidate passed the bias gate. "
                       "Falling back to best F1 regardless of bias.")
        eligible = candidates

    best = max(eligible, key=lambda c: c["metrics"]["f1_score"])
    logger.info("Selected best model: %s (F1: %.4f, run: %s)",
                best["name"], best["metrics"]["f1_score"], best["run_id"])
    return best


# ---------------------------------------------------------------------------
# 5. Final Evaluation on Held-Out Test Set
# ---------------------------------------------------------------------------

def final_evaluation(best, X_test, y_test, label_encoder, sens_test=None):
    """
    Run the best model on the held-out test set exactly once.
    Log final metrics and all visualizations to a dedicated MLflow run.

    ``sens_test`` is required when ``best['model']`` is a Fairlearn
    ThresholdOptimizer (post bias mitigation).
    """
    if best is None:
        logger.error("No best model — skipping final evaluation.")
        return None, None

    model = best["model"]
    model_type = best["name"]

    with mlflow.start_run(run_name=f"FINAL_{model_type}"):
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("evaluation_set", "test")
        mlflow.log_param("source_run_id", best["run_id"])

        metrics = evaluate_model(
            model, X_test, y_test,
            label_names=list(label_encoder.classes_),
            sensitive_df=sens_test,
        )

        logger.info("Final test metrics: %s", metrics)

        # Register best run model, then tag it as champion via alias.
        best_model_uri = f"runs:/{best['run_id']}/model"
        registered_model_version = mlflow.register_model(best_model_uri, Config.REGISTERED_MODEL_NAME)
        mlflow_client = MlflowClient(tracking_uri=Config.MLFLOW_TRACKING_URI)
        mlflow_client.set_registered_model_alias(
            name=Config.REGISTERED_MODEL_NAME,
            alias="champion",
            version=registered_model_version.version,
        )

        final_run_id = mlflow.active_run().info.run_id

    return metrics, final_run_id

# ---------------------------------------------------------------------------
# 5b. Save Best Model + Label Encoder Locally
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 5b. Save Best Model + Label Encoder Locally
# ---------------------------------------------------------------------------
def save_best_model_local(best, label_encoder):
    """Bundle champion model + feature pipeline + label encoder into a single MLflow pyfunc artifact at models/model/."""
    if not best: return
    import shutil
    from core_models.savvio_model_wrapper import SavVioModelWrapper

    model_dir = os.path.join(Config.BASE_DIR, "models")
    artifact_path = os.path.join(model_dir, "model")

    # mlflow.pyfunc.save_model needs file paths on disk
    joblib.dump(best["model"], os.path.join(model_dir, "classifier.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))

    # mlflow errors if target dir exists — remove previous run's artifact
    if os.path.exists(artifact_path):
        shutil.rmtree(artifact_path)

    mlflow.pyfunc.save_model(
        path=artifact_path,
        python_model=SavVioModelWrapper(),
        artifacts={
            "classifier": os.path.join(model_dir, "classifier.pkl"),
            "feature_pipeline": os.path.join(Config.MODEL_SAVE_DIR, "feature_pipeline.pkl"),
            "label_encoder": os.path.join(model_dir, "label_encoder.pkl"),
        },
    )
    logger.info("Saved bundled pyfunc artifact at %s", artifact_path)

# ---------------------------------------------------------------------------
# 6. LLM Layer Validation
# ---------------------------------------------------------------------------

def validate_llm_layer(best, data):
    """
    Smoke-test the LLM guardrail layer against the champion model.

    Samples one prediction per class (GREEN/YELLOW/RED) from the validation set,
    runs each through apply_llm_guardrails, and logs prompt versions + pass rate
    to a dedicated MLflow run for lineage tracking.
    """
    if best is None:
        logger.warning("LLM validation skipped — no champion model.")
        return

    model = best["model"]
    label_encoder = data["label_encoder"]
    X_val = data["X_val"]

    # Sample one index per class from the validation set.
    y_val_decoded = label_encoder.inverse_transform(data["y_val"])
    sample_indices = []
    for color in ["GREEN", "YELLOW", "RED"]:
        matches = np.where(y_val_decoded == color)[0]
        if len(matches) > 0:
            sample_indices.append(int(matches[0]))

    if not sample_indices:
        logger.warning("LLM validation: no validation samples found.")
        return

    guardrail_results = []
    with mlflow.start_run(run_name="llm_layer_validation"):
        mlflow.log_param("llm_provider",              LLMConfig.PROVIDER)
        mlflow.log_param("system_prompt_version",     LLMConfig.SYSTEM_PROMPT_VERSION)
        mlflow.log_param("intent_prompt_version",     LLMConfig.INTENT_PROMPT_VERSION)
        mlflow.log_param("response_prompt_version",   LLMConfig.RESPONSE_PROMPT_VERSION)
        mlflow.log_param("champion_model",            best["name"])

        for i in sample_indices:
            try:
                row = X_val[i:i+1] if not hasattr(X_val, "iloc") else X_val.iloc[i:i+1]
                y_pred = label_encoder.inverse_transform([model.predict(row)[0]])[0]
                proba = model.predict_proba(row)[0]
                confidence = float(proba.max())

                # Generic financial context — sufficient for a guardrail smoke test.
                user_data = {
                    "product_name": "Sample Product",
                    "product_price": 299.99,
                    "affordability_score": 0.0,
                    "savings_to_price_ratio": 3.0,
                    "emergency_fund_months": 4.0,
                    "debt_to_income_ratio": 0.25,
                    "triggered_rules": [],
                    "was_downgraded": False,
                }
                is_safe, _ = apply_llm_guardrails(y_pred, user_data, confidence)
                guardrail_results.append(int(is_safe))
                logger.info("LLM validation sample — label=%s, guardrail_passed=%s", y_pred, is_safe)

            except Exception as e:
                logger.warning("LLM validation sample %d failed: %s", i, e)

        if guardrail_results:
            pass_rate = sum(guardrail_results) / len(guardrail_results)
            mlflow.log_metric("guardrail_pass_rate", pass_rate)
            mlflow.log_metric("guardrail_samples_tested", len(guardrail_results))
            logger.info(
                "LLM layer validation complete — pass_rate=%.2f (%d/%d samples)",
                pass_rate, sum(guardrail_results), len(guardrail_results),
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_ci_metrics_files(final_metrics, best_candidate):
    """Write metrics.txt and bias_metrics.txt for CI gate jobs."""
    # metrics.txt — validation thresholds gate reads f1_score and roc_auc
    with open("metrics.txt", "w") as f:
        for key in ("f1_score", "roc_auc", "accuracy", "pr_auc"):
            if key in (final_metrics or {}):
                f.write(f"{key}={final_metrics[key]}\n")
    logger.info("Wrote metrics.txt")

    # bias_metrics.txt — CI gate reads bias_gate_passed (0 or 1).
    # We use the bias_passed bool that detect_and_mitigate() already computed
    # with category-aware thresholds (financial DPD is intentionally not checked,
    # product DPD threshold is 0.30, etc.). Writing raw DPD/EOD values and applying
    # a flat threshold in CI would incorrectly block on expected financial disparities.
    bias_passed = int((best_candidate or {}).get("bias_passed", True))
    with open("bias_metrics.txt", "w") as f:
        f.write(f"bias_gate_passed={bias_passed}\n")
    logger.info("Wrote bias_metrics.txt (bias_gate_passed=%d)", bias_passed)


def main():
    print("Starting End-to-End ML Pipeline...\n")
    print(f"[0/6] MLflow tracking URI: {Config.MLFLOW_TRACKING_URI}")
    print(f"[0/6] MLflow experiment: {Config.EXPERIMENT_NAME}")

    # 1. Initialize seeds and MLflow.
    print("[1/6] Initializing runtime + MLflow...")
    initialize()
    print("[1/6] Initialization complete.")

    # 2. Build features, encode labels, 3-way split.
    print("[2/6] Preparing data (load/encode/split)...")
    data = prepare_data()
    print(
        "[2/6] Data ready: "
        f"train={len(data['y_train'])}, val={len(data['y_val'])}, test={len(data['y_test'])}"
    )

    # 3. Train all candidates, evaluate on validation set.
    print("[3/6] Training baseline candidates...")
    candidates = train_candidates(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        data["sens_train"], data["sens_val"], data["label_encoder"],
        data.get("scenarios_val"),
    )
    print(f"[3/6] Baseline training complete. candidates={len(candidates)}")

    # 3b. Hyperparameter tuning on best baseline.
    print("[4/6] Running hyperparameter tuning on best baseline...")
    candidates, tuning_context = tune_candidate(candidates, data)
    print(f"[4/6] Tuning stage complete. candidates={len(candidates)}")

    # 4. Select best model (F1 + bias gate).
    print("[5/6] Selecting best model (F1 + bias gate)...")
    best = select_best_model(candidates)
    if best:
        print(
            "[5/6] Selected: "
            f"{best['name']} (val_f1={best['metrics']['f1_score']:.4f}, run_id={best['run_id']})"
        )
    else:
        print("[5/6] No model selected.")

    # 5. Final evaluation on held-out test set.
    print("[6/6] Running final evaluation on held-out test set...")
    final_metrics, final_run_id = final_evaluation(
        best,
        data["X_test"],
        data["y_test"],
        data["label_encoder"],
        sens_test=data.get("sens_test"),
    )
    if final_metrics is not None:
        print(f"[6/6] Final test metrics: {final_metrics}")
    else:
        print("[6/6] Final evaluation skipped or failed.")

    sensitivity_summary = run_sensitivity_analysis(best, tuning_context)

    # Write CI gate metric files (metrics.txt, bias_metrics.txt).
    best_candidate = next((c for c in candidates if best and c["name"] == best["name"]), None)
    write_ci_metrics_files(final_metrics, best_candidate)

    # 6. Save best model and label encoder locally under artifacts and preprocessing.
    save_best_model_local(best, data["label_encoder"])

    # 6b. Validate LLM guardrail layer — smoke test + log prompt versions to MLflow.
    print("[6b] Validating LLM guardrail layer...")
    validate_llm_layer(best, data)
    print("[6b] LLM layer validation complete.")

    # 7. Save a simple markdown summary for 3 baseline models + champion final metrics.
    summary_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_run_id = final_run_id or (best["run_id"] if best else "no_run_id")
    report_filename = f"evaluation_summary_{summary_timestamp}_{summary_run_id}.md"
    report_path = os.path.join(Config.BASE_DIR, "reports", report_filename)
    write_evaluation_summary_md(
        candidates,
        best,
        final_metrics,
        report_path,
        sensitivity_summary=sensitivity_summary,
    )

    if final_run_id and os.path.exists(report_path):
        with mlflow.start_run(run_id=final_run_id):
            mlflow.log_artifact(report_path, "reports")
    # Generate bias analysis report
    try:
        from guards.bias_report import generate_bias_report
        if best and final_metrics:
            best_candidate = next(
                (c for c in candidates if c["name"] == best["name"]), None
            )
            if best_candidate:
                generate_bias_report(
                    fairness_metrics=best_candidate.get("fairness_metrics", {}),
                    model_name=best["name"],
                    bias_passed=best_candidate.get("bias_passed", False),
                    all_flags=best_candidate.get("bias_flags", []),
                    mitigation_applied=bool(
                        best_candidate.get("mitigation_applied", False)
                    ),
                    mitigation_successful=bool(
                        best_candidate.get("mitigation_successful", False)
                    ),
                    final_metrics=final_metrics,
                    output_dir=os.path.join(Config.BASE_DIR, "reports"),
                )
    except Exception as e:
        logger.warning("Bias report generation failed: %s", e)

    # Summary.
    if best:
        print(f"\nPipeline Complete!")
        print(f"Best Model: {best['name']} — Val F1: {best['metrics']['f1_score']:.4f}")
        print(f"MLflow Run ID: {best['run_id']}")
    else:
        print("\nPipeline Complete — no valid model selected.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
