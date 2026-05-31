"""
Drift Detection — Baseline comparison using Evidently AI.
Monitors product and financial feature distributions for statistical shifts.
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

import pandas as pd
import requests
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset

# Suppress logging on library import
logging.getLogger(__name__).addHandler(logging.NullHandler())

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load environment variables and config
# ---------------------------------------------------------------------------

# Load .env from monitoring/ (one level up — shared with the monitoring stack)
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_PATH)

# Load alert_config.yaml from the same directory as this file
_CONFIG_PATH = Path(__file__).parent / "alert_config.yaml"
with open(_CONFIG_PATH) as f:
    _CONFIG = yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Configuration — loaded from alert_config.yaml
# ---------------------------------------------------------------------------

MONITOR_COLS = (
    _CONFIG["monitored_features"]["financial"]
    + _CONFIG["monitored_features"]["product"]
)

COLUMN_THRESHOLDS: Dict[str, float] = _CONFIG["column_thresholds"]["overrides"]
DEFAULT_STATTEST_THRESHOLD: float = _CONFIG["column_thresholds"]["default"]

DRIFT_THRESHOLDS = _CONFIG["drift_thresholds"]
OUTPUT_SHIFT_THRESHOLD: float = _CONFIG["output_shift_threshold"]

# Notification config
_SLACK_CFG = _CONFIG["notification"]["slack"]
_EMAIL_CFG = _CONFIG["notification"]["email"]


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------

def send_slack_alert(summary: Dict) -> None:
    """Send a Slack alert if severity is in the notify_on list."""
    if not _SLACK_CFG.get("enabled"):
        return

    severity = summary["severity"]
    if severity not in _SLACK_CFG.get("notify_on", []):
        return

    webhook_url = os.environ.get(_SLACK_CFG["webhook_env_var"])
    if not webhook_url:
        logger.warning("Slack webhook URL not set — skipping Slack alert.")
        return

    # Build severity indicator and color based on severity
    emoji = {"GREEN": "[OK]", "YELLOW": "[WARNING]", "RED": "[CRITICAL]"}.get(severity, "[INFO]")

    drifted = summary.get("drifted_columns", {})
    drifted_list = (
        "\n".join(
            f"• `{col}` — score: {info.get('drift_score', 'N/A'):.4f}"
            for col, info in drifted.items()
            if info.get("drift_detected")
        )
        or "None"
    )

    message = {
        "text": (
            f"{emoji} *SavVio Drift Alert — {severity}*\n"
            f"*Action:* {summary['action']}\n"
            f"*Columns drifted:* {summary['n_drifted_cols']}/{summary['n_total_cols']} "
            f"({summary['share_drifted']*100:.1f}%)\n"
            f"*Drifted columns:*\n{drifted_list}\n"
            f"*Time:* {summary['timestamp']}"
        )
    }

    try:
        resp = requests.post(webhook_url, json=message, timeout=10)
        if resp.status_code == 200:
            logger.info("Slack alert sent — severity: %s", severity)
        else:
            logger.warning("Slack alert failed — status %d: %s", resp.status_code, resp.text)
    except Exception as e:
        logger.warning("Slack alert error: %s", e)


# ---------------------------------------------------------------------------
# Core detection function
# ---------------------------------------------------------------------------

def run_drift_detection(
    baseline_path: str,
    current_data,
    output_dir: str = str(Path(__file__).parent.parent / "reports"),
) -> Dict:
    """Computes Wasserstein distance for monitored columns vs baseline."""
    logger.info("Loading baseline and current data...")
    baseline = pd.read_csv(baseline_path)
    current  = current_data if isinstance(current_data, pd.DataFrame) else pd.read_csv(current_data)

    # Only monitor columns present in both datasets; warn about any missing ones.
    cols = [c for c in MONITOR_COLS if c in baseline.columns and c in current.columns]
    missing = set(MONITOR_COLS) - set(cols)
    if missing:
        logger.warning(
            "Skipping columns not found in both datasets: %s. "
            "Update MONITOR_COLS or regenerate the baseline to include them.",
            sorted(missing),
        )

    baseline = baseline[cols]
    current  = current[cols]

    logger.info("Running Evidently drift report on %d columns...", len(cols))

    per_column_threshold = {
        col: COLUMN_THRESHOLDS.get(col, DEFAULT_STATTEST_THRESHOLD)
        for col in cols
    }

    report   = Report(metrics=[DataDriftPreset(
        num_method="wasserstein",
        per_column_threshold=per_column_threshold,
    )])
    snapshot = report.run(reference_data=baseline, current_data=current)

    # Extract summary counts from metric_results (Evidently 0.7.x API).
    share_drifted = 0.0
    n_drifted     = 0
    drifted_cols  = {}
    for val in snapshot.metric_results.values():
        vtype = type(val).__name__
        if vtype == "CountValue":                   # DriftedColumnsCount
            n_drifted     = int(val.count.value)
            share_drifted = float(val.share.value)
        elif vtype == "SingleValue" and "Value drift for" in val.display_name:
            col_name = val.display_name.replace("Value drift for ", "")
            params   = val.metric_value_location.metric.params
            drifted_cols[col_name] = {
                "drift_score":    round(float(val.value), 4),
                "drift_detected": float(val.value) >= params.get("threshold", DEFAULT_STATTEST_THRESHOLD),
                "method":         params.get("method", "wasserstein"),
            }

    n_total = len(cols)

    # Determine severity.
    if share_drifted < DRIFT_THRESHOLDS["green"]:
        severity = "GREEN"
        action   = "No action needed — distributions stable."
    elif share_drifted < DRIFT_THRESHOLDS["yellow"]:
        severity = "YELLOW"
        action   = "Monitor closely — minor drift detected."
    else:
        severity = "RED"
        action   = "Alert — significant drift detected. Consider retraining."

    summary = {
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "severity":        severity,
        "action":          action,
        "share_drifted":   round(share_drifted, 4),
        "n_drifted_cols":  n_drifted,
        "n_total_cols":    n_total,
        "drifted_columns": drifted_cols,
    }

    # Persist outputs.
    os.makedirs(output_dir, exist_ok=True)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path  = os.path.join(output_dir, f"drift_report_{timestamp}.html")
    summary_path = os.path.join(output_dir, f"drift_summary_{timestamp}.json")

    snapshot.save_html(report_path)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Drift severity: %s — %s", severity, action)
    logger.info("Drift report saved to %s", report_path)
    logger.info("Drift summary saved to %s", summary_path)

    if severity == "RED":
        logger.warning(
            "DRIFT ALERT — %d/%d columns drifted (%.1f%%). Action: %s",
            n_drifted, n_total, share_drifted * 100, action,
        )

    # Send Slack alert
    send_slack_alert(summary)

    return summary


# ---------------------------------------------------------------------------
# Output distribution drift
# ---------------------------------------------------------------------------

def check_output_drift(
    baseline_output_dist: Dict[str, float],
    current_output_dist: Dict[str, float],
) -> Dict:
    """
    Compare GREEN/YELLOW/RED prediction ratios against baseline.

    Args:
        baseline_output_dist: e.g. {"GREEN": 0.55, "YELLOW": 0.25, "RED": 0.20}
        current_output_dist:  e.g. {"GREEN": 0.70, "YELLOW": 0.15, "RED": 0.15}

    Returns:
        Dict with per-label drift status, shift magnitude, and drifted flag.
    """
    results = {}
    for label in ["GREEN", "YELLOW", "RED"]:
        baseline_share = baseline_output_dist.get(label, 0.0)
        current_share  = current_output_dist.get(label, 0.0)
        shift          = abs(current_share - baseline_share)
        drifted        = shift > OUTPUT_SHIFT_THRESHOLD

        results[label] = {
            "baseline_share": round(baseline_share, 4),
            "current_share":  round(current_share, 4),
            "shift":          round(shift, 4),
            "drifted":        drifted,
        }

        if drifted:
            logger.warning(
                "OUTPUT DRIFT: %s label shifted by %.1f%% "
                "(baseline=%.1f%%, current=%.1f%%)",
                label, shift * 100,
                baseline_share * 100,
                current_share * 100,
            )

    return results


# ---------------------------------------------------------------------------
# Entry point for scheduled runs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess as _sp
    import sys as _sys
    import tempfile as _tempfile

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "savviocore" / "src"))

    _GCS_BASELINE = "gs://savvio-dev-mlflow-artifacts/monitoring/baseline_data.csv"

    # Download baseline from GCS into a temp file.
    logger.info("Downloading baseline from GCS: %s", _GCS_BASELINE)
    _tmp_baseline = _tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    _tmp_baseline.close()
    _dl = _sp.run(
        ["gcloud", "storage", "cp", _GCS_BASELINE, _tmp_baseline.name],
        capture_output=True, text=True,
    )
    if _dl.returncode != 0:
        print(f"Baseline not found in GCS — skipping drift detection.")
        print(f"(Run generate_baseline.py after model training to upload the baseline.)")
        print(f"Details: {_dl.stderr.strip()}")
        raise SystemExit(0)

    # Always pull current data live from the DB.
    logger.info("Collecting current production data from DB...")
    try:
        from collect_production_data import collect_df
        current_df = collect_df()
    except Exception as _e:
        print(f"Could not collect production data from DB: {_e}")
        print("Skipping drift detection — DB unavailable or no data yet.")
        raise SystemExit(0)

    summary = run_drift_detection(
        baseline_path=_tmp_baseline.name,
        current_data=current_df,
    )
    Path(_tmp_baseline.name).unlink(missing_ok=True)

    print(f"\nDrift Detection Complete")
    print(f"Severity:        {summary['severity']}")
    print(f"Action:          {summary['action']}")
    print(f"Columns drifted: {summary['n_drifted_cols']}/{summary['n_total_cols']}")

    # for label, info in output_results.items():
    #     print(f"  {label}: shift={info['shift']:.3f}, drifted={info['drifted']}")