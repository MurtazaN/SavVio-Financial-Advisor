"""Run validation stages for the data pipeline.

Stages:
    - raw: schema and rule checks for raw inputs
    - raw_anomalies: tier-1 anomaly scan for raw inputs (monitoring only)
    - processed: schema and rule checks for processed outputs
    - features: schema and rule checks for engineered features
    - anomalies: tier-2 anomaly checks on final DB-load datasets

Stage outcome is derived from `ValidationReport.summary["pipeline_action"]`:
    - CONTINUE: no warning/critical failures
    - ALERT: warning failures present
    - HALT: critical failures present
"""

import logging
import sys
import os
from pathlib import Path

# Resolve local imports from the validation package.
current_script_path = Path(__file__).resolve()
validation_dir = current_script_path.parent          # .../dags/src/validation/


def _find_pipeline_root(start: Path) -> Path:
    """Find data_pipeline root by looking for data/ and config/ directories."""
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    return current_script_path.parents[2]  # fallback: .../dags/


pipeline_root = _find_pipeline_root(current_script_path.parent)

# Ensure running from data_pipeline root so relative data paths work
if os.getcwd() != str(pipeline_root):
    print(f"Changing working directory to: {pipeline_root}")
    os.chdir(pipeline_root)

from savviocore.validation.validation_config import ValidationReport
from src.validation.validate.raw_validator import run_raw_validation
from src.validation.validate.processed_validator import run_processed_validation
from savviocore.validation.feature_validator import run_feature_validation
from src.validation.anomaly.anomaly_validator import run_anomaly_validation, run_raw_anomaly_validation

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Airflow-compatible callables
# ═══════════════════════════════════════════════════════════════════════════

def validate_raw(**kwargs) -> dict:
    """
    Airflow PythonOperator callable for raw validation.
    Raises RuntimeError on CRITICAL failure to halt the DAG.
    """
    report = run_raw_validation()
    _handle_report(report)
    return report.summary


def validate_processed(**kwargs) -> dict:
    """
    Airflow PythonOperator callable for processed validation.
    Raises RuntimeError on CRITICAL failure to halt the DAG.
    """
    report = run_processed_validation()
    _handle_report(report)
    return report.summary


def validate_features(**kwargs) -> dict:
    """
    Airflow PythonOperator callable for feature validation.
    Raises RuntimeError on CRITICAL failure to halt the DAG.
    """
    report = run_feature_validation()
    _handle_report(report)
    return report.summary


def validate_raw_anomalies(**kwargs) -> dict:
    """
    Tier 1: Light anomaly scan on raw financial data. INFO-only — never halts.
    """
    report = run_raw_anomaly_validation()
    _handle_report(report)
    return report.summary


def validate_anomalies(**kwargs) -> dict:
    """
    Tier 2: Full anomaly detection on featured financial data.
    WARNING/CRITICAL thresholds gate the DB load.
    """
    report = run_anomaly_validation()
    _handle_report(report)
    return report.summary


# ═══════════════════════════════════════════════════════════════════════════
# Report handling and alerts
# ═══════════════════════════════════════════════════════════════════════════

def _handle_report(report: ValidationReport) -> None:
    """
    Central handler for validation reports.
    - HALT   → raise exception (stops Airflow task)
    - ALERT  → send notification (email/Slack placeholder)
    - CONTINUE → log and proceed
    """
    action = report.summary["pipeline_action"]

    if action == "HALT":
        msg = (
            f"VALIDATION FAILED [{report.stage}]: "
            f"{report.summary['failed_critical']} CRITICAL failures. "
            f"Pipeline halted."
        )
        logger.critical(msg)
        raise RuntimeError(msg)

    elif action == "ALERT":
        logger.warning(
            "Validation [%s] passed with %d warnings — triggering alerts",
            report.stage, report.summary["failed_warning"],
        )
        _send_alert(report)

    else:
        logger.info("Validation [%s] passed — continuing pipeline", report.stage)


def _send_alert(report: ValidationReport) -> None:
    """
    Placeholder for alert dispatch.
    In production, wire up to Airflow EmailOperator or SlackWebhookOperator.
    """
    summary = report.summary
    failed_checks = [r for r in report.results if not r.passed]

    subject = f"SavVio Validation Warning — {summary['stage'].upper()}"
    body_lines = [
        f"Stage: {summary['stage']}",
        f"Timestamp: {summary['timestamp']}",
        f"Warnings: {summary['failed_warning']}",
        f"Info: {summary['failed_info']}",
        "",
        "Failed checks:",
    ]
    for c in failed_checks:
        body_lines.append(f"  [{c.severity.name}] {c.check_name}: {c.details}")

    body = "\n".join(body_lines)
    logger.warning("ALERT:\n%s\n%s", subject, body)

    # TODO: Replace with actual alert
    # from airflow.operators.email import EmailOperator
    # EmailOperator(to="team@savvio.dev", subject=subject, html_content=body)


# ═══════════════════════════════════════════════════════════════════════════
# CLI — run all stages or a specific one
# ═══════════════════════════════════════════════════════════════════════════

# Dependency chain for "all" mode.
# Each stage depends on the previous one passing (no CRITICAL failures).
# raw_anomalies is independent (INFO-only, never halts) and runs alongside raw.
#
#   raw ──────→ processed ──→ features ──→ anomalies
#    ↘ raw_anomalies (independent, INFO-only)
#
VALIDATION_PIPELINE = [
    # (stage_name, depends_on)
    ("raw",            None),
    ("raw_anomalies",  None),       # independent — INFO-only, never halts
    ("processed",      "raw"),      # skip if raw HALTed
    ("features",       "processed"),# skip if processed HALTed
    ("anomalies",      "features"), # skip if features HALTed
]

STAGE_MAP = {
    "raw":            validate_raw,
    "raw_anomalies":  validate_raw_anomalies,
    "processed":      validate_processed,
    "features":       validate_features,
    "anomalies":      validate_anomalies,
}


def main():
    import argparse

    from src.utils import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(description="SavVio Data Validation Runner")
    parser.add_argument(
        "stage",
        choices=["raw", "processed", "features", "raw_anomalies", "anomalies", "all"],
        help="Which validation stage to run",
    )
    args = parser.parse_args()

    # ── Single-stage mode: run and exit ───────────────────────────────────
    if args.stage != "all":
        validator_func = STAGE_MAP[args.stage]
        try:
            validator_func()
        except RuntimeError:
            logger.error(f"Validation for stage '{args.stage}' failed critically.")
            sys.exit(1)
        except Exception as e:
            logger.exception(f"Unexpected error during '{args.stage}': {e}")
            sys.exit(1)
        sys.exit(0)

    # ── All-stages mode: follow dependency chain, fail-fast ───────────────
    halted_stages: set[str] = set()   # stages that HALTed
    exit_code = 0

    for stage_name, depends_on in VALIDATION_PIPELINE:
        # Skip if upstream dependency HALTed
        if depends_on and depends_on in halted_stages:
            logger.warning(
                "SKIPPING [%s] — upstream stage '%s' failed critically. "
                "Fix upstream issues first.",
                stage_name, depends_on,
            )
            halted_stages.add(stage_name)  # propagate halt downstream
            exit_code = 1
            continue

        validator_func = STAGE_MAP[stage_name]
        try:
            logger.info(f"Running validation for stage: {stage_name}")
            validator_func()
        except RuntimeError:
            exit_code = 1
            halted_stages.add(stage_name)
            logger.error(
                "Stage '%s' HALTED — downstream stages will be skipped.",
                stage_name,
            )
        except Exception as e:
            exit_code = 1
            halted_stages.add(stage_name)
            logger.exception(
                "Unexpected error in stage '%s': %s — treating as HALT.",
                stage_name, e,
            )

    # ── Final summary ─────────────────────────────────────────────────────
    if halted_stages:
        logger.critical(
            "Pipeline finished with HALTED stages: %s",
            ", ".join(sorted(halted_stages)),
        )
    else:
        logger.info("All validation stages completed successfully.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
