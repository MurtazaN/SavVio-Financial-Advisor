"""Anomaly checks used by the validation pipeline.

Tier 1 (`raw_anomaly`): lightweight checks on raw financial inputs for visibility.
Tier 2 (`anomaly`): pre-load checks on featured financial data used for DB import.

Anomaly detection is scoped to financial data only — this is where outliers have
the highest downstream impact on the agent's purchase guardrail decisions.
Product and review data quality is already covered by raw/processed validators.
"""

import logging
import sys
import os
import pandas as pd
from pathlib import Path

from src.validation.anomaly.detectors import AnomalyDetector
from savviocore.validation.validation_config import CheckResult, Severity, ValidationReport
from src.utils import setup_logging

current_file_path = Path(__file__).resolve()
validation_dir = current_file_path.parent.parent


def _find_pipeline_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    return current_file_path.parents[4]


pipeline_root = _find_pipeline_root(current_file_path.parent)
if os.getcwd() != str(pipeline_root):
    os.chdir(pipeline_root)


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _report_outliers(column: str, method: str, outliers: list,
                     dataset: str, severity: Severity = Severity.WARNING,
                     stage: str = "anomaly") -> CheckResult:
    count = len(outliers)
    passed = count == 0
    details = f"Found {count} outliers in '{column}' using {method}"
    if not passed:
        details += f" (indices: {outliers[:5]}...)"
    return CheckResult(
        check_name=f"anomaly_{column}_{method}",
        passed=passed, severity=severity,
        dataset=dataset, stage=stage,
        details=details, metric_value=count,
    )


def _quarantine_records(df: pd.DataFrame, indices: list, dataset_name: str,
                        quarantine_dir: str = "data/quarantine"):
    if not indices:
        return
    try:
        Path(quarantine_dir).mkdir(parents=True, exist_ok=True)
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        path = Path(quarantine_dir) / f"{dataset_name}_anomalies_{ts}.json"
        df.loc[indices].to_json(path, orient="records", lines=True)
        logger.warning(f"Quarantined {len(indices)} suspicious records to {path}")
    except Exception as e:
        logger.error(f"Failed to quarantine records: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 — Raw financial anomaly checks (light, INFO-only, never halts)
# ═══════════════════════════════════════════════════════════════════════════

def _raw_financial_anomalies(path: str) -> list[CheckResult]:
    results, ds = [], "financial_raw"
    if not Path(path).exists():
        return [CheckResult(f"load_{ds}", True, Severity.INFO, ds, "raw_anomaly", "File not found — skipped", 0)]
    try:
        df = _load_csv(path)
        detector = AnomalyDetector(df)
        outliers_income = detector.check_iqr("monthly_income_usd", multiplier=3.0)
        results.append(_report_outliers("monthly_income_usd", "iqr_3.0", outliers_income, ds, Severity.INFO, "raw_anomaly"))
        outliers_savings = detector.check_iqr("savings_usd", multiplier=3.0)
        results.append(_report_outliers("savings_usd", "iqr_3.0", outliers_savings, ds, Severity.INFO, "raw_anomaly"))
        outliers_expenses = detector.check_z_score("monthly_expenses_usd", threshold=4.0)
        results.append(_report_outliers("monthly_expenses_usd", "zscore_4.0", outliers_expenses, ds, Severity.INFO, "raw_anomaly"))
        all_outliers = set(outliers_income + outliers_savings + outliers_expenses)
        if all_outliers:
            _quarantine_records(df, list(all_outliers), "raw_financial")
    except Exception as e:
        logger.error(f"Raw financial anomaly check failed: {e}")
        results.append(CheckResult(f"anomaly_check_{ds}", True, Severity.INFO, ds, "raw_anomaly", str(e), 0))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 — Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_raw_anomaly_validation(
    financial_path: str = "data/raw/financial_data.csv",
) -> ValidationReport:
    """Tier 1: Light anomaly scan on raw financial data. INFO-only — never halts the pipeline."""
    report = ValidationReport(stage="raw_anomaly")

    logger.info("═" * 50)
    logger.info("  RAW ANOMALY DETECTION (Tier 1 — Financial Only)")
    logger.info("═" * 50)

    for r in _raw_financial_anomalies(financial_path):
        report.add(r)

    report.print_summary()
    report.save()
    return report


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2 — Featured financial anomaly checks (full, WARNING/CRITICAL, gates DB load)
# ═══════════════════════════════════════════════════════════════════════════

def _featured_financial_anomalies(path: str) -> list[CheckResult]:
    """Statistical anomaly detection on financial_featured.csv (goes to DB)."""
    results, ds = [], "financial_featured"
    if not Path(path).exists():
        return [CheckResult(f"load_{ds}", False, Severity.CRITICAL, ds, "anomaly", "File not found", 0)]
    try:
        df = _load_csv(path)
        detector = AnomalyDetector(df)

        for col, mult in [("monthly_income", 3.0), ("savings_balance", 4.5)]:
            if col in df.columns:
                outliers = detector.check_iqr(col, multiplier=mult)
                results.append(_report_outliers(col, f"iqr_{mult}", outliers, ds, Severity.WARNING))
        if "monthly_expenses" in df.columns:
            outliers = detector.check_z_score("monthly_expenses", threshold=6.0)
            results.append(_report_outliers("monthly_expenses", "zscore_6.0", outliers, ds, Severity.WARNING))

        for col in ["discretionary_income", "debt_to_income_ratio", "savings_rate",
                     "expense_burden_ratio", "emergency_fund_months"]:
            if col in df.columns:
                outliers = detector.check_iqr(col, multiplier=4.0)
                results.append(_report_outliers(col, "iqr_4.0", outliers, ds, Severity.WARNING))

        if "monthly_income" in df.columns:
            zero_income = detector.check_rule("monthly_income", lambda x: x > 0, "income_positive")
            results.append(_report_outliers("monthly_income", "zero_check", zero_income, ds, Severity.WARNING))

        if "monthly_income" in df.columns and "monthly_expenses" in df.columns:
            mask = df["monthly_expenses"] > (2 * df["monthly_income"])
            outliers = df[mask].index.tolist()
            results.append(_report_outliers("expense_income_ratio", "expenses_2x_income", outliers, ds, Severity.WARNING))

    except Exception as e:
        logger.error(f"Featured financial anomaly check failed: {e}")
        results.append(CheckResult(f"anomaly_check_{ds}", False, Severity.CRITICAL, ds, "anomaly", str(e), 0))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2 — Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_anomaly_validation(
    financial_path: str = "data/features/financial_featured.csv",
) -> ValidationReport:
    """Tier 2: Full anomaly detection on featured financial data going into DB."""
    report = ValidationReport(stage="anomaly")

    logger.info("═" * 50)
    logger.info("  ANOMALY DETECTION (Tier 2 — Financial Only, Pre-DB)")
    logger.info("═" * 50)

    logger.info(f"── Financial featured: {financial_path} ──")
    for r in _featured_financial_anomalies(financial_path):
        report.add(r)

    report.print_summary()
    report.save()

    if not report.passed:
        logger.critical("ANOMALY DETECTION FAILED — pipeline should HALT (block DB load)")
    elif report.has_warnings:
        logger.warning("ANOMALY DETECTION found outliers — sending alerts")

    return report


# ═══════════════════════════════════════════════════════════════════════════
# CLI entrypoint
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    setup_logging()

    parser = argparse.ArgumentParser(description="Run Anomaly Detection (Financial Only)")
    parser.add_argument("tier", choices=["raw", "featured"], help="Which tier to run")
    args = parser.parse_args()

    if args.tier == "raw":
        run_raw_anomaly_validation()
    else:
        run_anomaly_validation()
