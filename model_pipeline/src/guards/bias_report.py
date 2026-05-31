"""
Bias Analysis Report Generator for SavVio Model Pipeline.

Generates a clean markdown report after every pipeline run covering:
  - Pre-training bias mitigation results (before/after numbers)
  - Post-training bias detection results (all 17 slices)
  - Flags raised and whether the gate passed
  - Final model metrics

File: model_pipeline/src/guards/bias_report.py

Called from run_pipeline.py after final evaluation:

    from guards.bias_report import generate_bias_report
    generate_bias_report(
        fairness_metrics=fairness_metrics,
        model_name=best_model_name,
        bias_passed=bias_passed,
        all_flags=all_flags,
        mitigation_applied=mitigation_applied,
        mitigation_successful=mitigation_successful,
        final_metrics=final_metrics,
        output_dir="reports/",
    )

Output:
    model_pipeline/reports/bias_report_{model_name}_{timestamp}.md
    model_pipeline/reports/bias_report_latest.md  (always overwritten)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMOGRAPHIC_SLICES = {"region", "employment_status"}
FINANCIAL_SLICES = {
    "income_band", "dti_band", "savings_band", "emergency_fund_band",
    "discretionary_income_band", "saving_to_income_band",
    "expense_burden_band", "affordability_band", "has_loan", "over_leveraged",
}
PRODUCT_SLICES = {
    "price_band", "review_confidence_band", "rating_variance_band",
    "average_rating_band", "cold_start",
}
MONITOR_ONLY_SLICES = {"savings_band", "affordability_band", "emergency_fund_band"}

# Pre-training mitigation results from the last verified run.
MITIGATION_BASELINE = {
    "unemployed_before":   9.9,
    "unemployed_after":    11.8,
    "student_before":      9.7,
    "student_after":       11.7,
    "neutral_before":      4.89,
    "neutral_after":       6.92,
    "unverified_before":   4.15,
    "unverified_after":    6.81,
    "emi_errors_fixed":    743,
    "over_leveraged":      7034,
    "synthetic_rows":      972,
    "categories_before":   "80+",
    "categories_after":    46,
    "reviews_before":      2_103_990,
    "reviews_after":       2_208_864,
    "financial_before":    32_424,
    "financial_after":     34_866,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slice_category(slice_name: str) -> str:
    if slice_name in DEMOGRAPHIC_SLICES:
        return "Demographic"
    if slice_name in FINANCIAL_SLICES:
        return "Financial"
    if slice_name in PRODUCT_SLICES:
        return "Product"
    return "Other"


def _flag_emoji(passed: bool) -> str:
    return "[PASSED]" if passed else "[FAILED]"


def _parse_group_metrics(
    fairness_metrics: Dict[str, float],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Parse flat fairness_metrics dict into nested structure:
    { slice_name: { group_name: { acc, f1, green_rate, f1_disparity } } }
    """
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    prefix = "grp_"
    for key, val in fairness_metrics.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        for metric in ("_accuracy", "_f1", "_green_rate", "_f1_disparity", "_auc"):
            if rest.endswith(metric):
                slice_group = rest[: -len(metric)]
                slice_name = None
                group_name = None
                for s in list(DEMOGRAPHIC_SLICES) + list(FINANCIAL_SLICES) + list(PRODUCT_SLICES):
                    s_key = s.replace(" ", "_").replace("-", "_")
                    if slice_group.startswith(s_key + "_"):
                        slice_name = s
                        group_name = slice_group[len(s_key) + 1:]
                        break
                if slice_name and group_name:
                    result.setdefault(slice_name, {}).setdefault(group_name, {})
                    result[slice_name][group_name][metric.lstrip("_")] = val
                break
    return result


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _header(model_name: str, bias_passed: bool, timestamp: str) -> str:
    gate = _flag_emoji(bias_passed)
    return f"""# SavVio — Bias Analysis Report

**Generated:** {timestamp}
**Champion Model:** `{model_name}`
**Bias Gate:** {gate}

---
"""


def _summary_section(
    bias_passed: bool,
    all_flags: List[str],
    final_metrics: Dict[str, float],
    mitigation_applied: bool,
    mitigation_successful: bool,
) -> str:
    flags_str = "\n".join(f"  - `{f}`" for f in all_flags) if all_flags else "  - None"
    mit_str = (
        f"Applied — {'Successful [PASSED]' if mitigation_successful else 'Unsuccessful [FAILED]'}"
        if mitigation_applied else "Not needed — model passed gate on first check [PASSED]"
    )

    return f"""## Summary

| Metric | Value |
|---|---|
| Bias Gate | {_flag_emoji(bias_passed)} |
| Gate-Blocking Flags | {len(all_flags)} |
| ThresholdOptimizer | {mit_str} |
| Final Test F1 | {final_metrics.get('f1_score', 'N/A')} |
| Final Test Accuracy | {final_metrics.get('accuracy', 'N/A')} |
| Final Test ROC AUC | {final_metrics.get('roc_auc', 'N/A')} |
| Final Test PR AUC | {final_metrics.get('pr_auc', 'N/A')} |

### Bias Flags Raised
{flags_str}

---
"""


def _pretraining_section() -> str:
    m = MITIGATION_BASELINE
    return f"""## Part 1 — Pre-Training Bias Mitigation
**File:** `model_pipeline/src/data/bias_mitigation.py`

### What Was Found in Raw Data

| Problem | Value | Threshold | Action Taken |
|---|---|---|---|
| Near-zero savings users | 0.0% | ≥10% | Synthetic augmentation — {m['synthetic_rows']} rows created |
| Unemployed users | {m['unemployed_before']}% | ≥10% | Oversampled |
| Student users | {m['student_before']}% | ≥10% | Oversampled |
| EMI > 10× income | {m['emi_errors_fixed']} rows | Data quality | EMI zeroed out (data entry error) |
| EMI > income | {m['over_leveraged']} rows | Data quality | Flagged as over_leveraged, upweighted 2× |
| Neutral reviews (3 star) | {m['neutral_before']}% | ≥5% | Oversampled |
| Unverified purchases | {m['unverified_before']}% | ≥5% | Oversampled |
| Product categories | {m['categories_before']} leaf nodes | ≥5% each | Collapsed to {m['categories_after']} groups |
| helpful_vote = 0 | 80.8% | Dominance risk | Sample weights assigned |
| user_id uniqueness | 83.4% | ≥95% | Reviewer cap (max 50/user) |

### Results After Mitigation

| Group | Before | After |
|---|---|---|
| Unemployed users | {m['unemployed_before']}% | {m['unemployed_after']}% |
| Student users | {m['student_before']}% | {m['student_after']}% |
| Neutral reviews | {m['neutral_before']}% | {m['neutral_after']}% |
| Unverified purchases | {m['unverified_before']}% | {m['unverified_after']}% |
| Near-zero savings | 0.0% | {m['synthetic_rows']} synthetic rows added |
| EMI data errors | {m['emi_errors_fixed']} bad rows | Corrected |
| Over-leveraged users | Not flagged | {m['over_leveraged']} flagged + upweighted |
| Product categories | {m['categories_before']} leaf nodes | {m['categories_after']} groups |

### Dataset Sizes

| Dataset | Before | After |
|---|---|---|
| Financial records | {m['financial_before']:,} | {m['financial_after']:,} |
| Reviews | {m['reviews_before']:,} | {m['reviews_after']:,} |
| Products | 94,327 | 94,327 (unchanged — premium already above threshold) |

### New Columns Added — Must Stay in Config.COLUMNS_TO_DROP

| Column | Dataset | Purpose |
|---|---|---|
| `over_leveraged_flag` | Financial | Marks EMI > income users. Never a model feature. |
| `synthetic_flag` | Financial | Marks synthetic rows. Excluded from test set. |
| `sample_weight` | Product + Review | Passed to model.fit(). Never a model feature. |
| `category_leaf` | Product | Original category before taxonomy collapse. |

---
"""


def _posttraining_section(
    fairness_metrics: Dict[str, float],
    all_flags: List[str],
    bias_passed: bool,
) -> str:
    agg_f1 = fairness_metrics.get("aggregate_f1", "N/A")
    agg_acc = fairness_metrics.get("aggregate_accuracy", "N/A")
    agg_green = fairness_metrics.get("aggregate_green_rate", "N/A")

    group_metrics = _parse_group_metrics(fairness_metrics)

    # Build DPD/EOD table
    dpd_eod_rows = []
    for key in sorted(fairness_metrics.keys()):
        if key.startswith("bias_dpd_") or key.startswith("bias_eod_"):
            metric_type = "DPD" if "dpd" in key else "EOD"
            slice_name = key.replace("bias_dpd_", "").replace("bias_eod_", "")
            cat = _slice_category(slice_name)
            val = fairness_metrics[key]
            monitor = slice_name in MONITOR_ONLY_SLICES and metric_type == "EOD"
            flagged = any(slice_name in f for f in all_flags)
            status = "[FLAGGED]" if flagged else ("[MONITOR]" if monitor else "[OK]")
            dpd_eod_rows.append(
                f"| {slice_name} | {cat} | {metric_type} | {val:.4f} | {status} |"
            )

    dpd_eod_table = "\n".join(dpd_eod_rows) if dpd_eod_rows else "| No data available | | | | |"

    # Build per-group F1 table
    group_rows = []
    for slice_name in sorted(group_metrics.keys()):
        cat = _slice_category(slice_name)
        for group_name, metrics in sorted(group_metrics[slice_name].items()):
            f1 = metrics.get("f1", "N/A")
            acc = metrics.get("accuracy", "N/A")
            disp = metrics.get("f1_disparity", "N/A")
            flag = ""
            if isinstance(disp, float) and abs(disp) > 0.10:
                flag = "[DISPARITY]"
            if isinstance(f1, float) and f1 < 0.50:
                flag = "[LOW F1]"
            group_rows.append(
                f"| {slice_name} | {cat} | {group_name} | {f1} | {acc} | {disp} | {flag} |"
            )

    group_table = "\n".join(group_rows) if group_rows else "| No data available | | | | | | |"

    return f"""## Part 2 — Post-Training Bias Detection
**File:** `model_pipeline/src/guards/bias_detection.py`

### Aggregate Performance

| Metric | Value |
|---|---|
| Aggregate F1 | {agg_f1} |
| Aggregate Accuracy | {agg_acc} |
| Aggregate GREEN Rate | {agg_green} |

### Threshold System

| Category | DPD Checked | DPD Threshold | EOD Threshold | Reasoning |
|---|---|---|---|---|
| Demographic | Yes | 0.10 | 0.10 | Employment and region must never affect accuracy |
| Financial | No | Not checked | 0.15 | GREEN rate differences are by design |
| Product | Yes (relaxed) | 0.25 | 0.15 | Some price-based difference is expected |

Monitor-only EOD slices (not gate-blocking): `savings_band`, `affordability_band`, `emergency_fund_band`

### DPD and EOD Results

| Slice | Category | Metric | Value | Status |
|---|---|---|---|---|
{dpd_eod_table}

### Per-Group F1 Disparity Table

| Slice | Category | Group | F1 | Accuracy | ΔF1 | Flag |
|---|---|---|---|---|---|---|
| AGGREGATE | — | all | {agg_f1} | {agg_acc} | — | — |
{group_table}

### Flag Conditions

- F1 disparity > 0.10 below aggregate → flagged
- F1 < 0.50 absolute floor → flagged
- DPD or EOD exceeds category threshold → flagged
- Groups with n < 100 do not trigger gate-blocking flags

---
"""


def _mitigation_section(
    mitigation_applied: bool,
    mitigation_successful: bool,
    bias_passed_after: bool,
) -> str:
    if not mitigation_applied:
        return """## Part 3 — Post-Training Mitigation

ThresholdOptimizer was not needed — the champion model passed the bias gate on the first check with zero gate-blocking flags.

---
"""
    status = "Successful — bias gate passed after mitigation [PASSED]" if mitigation_successful \
        else "Unsuccessful — bias still present after mitigation, flagged for human review [FAILED]"

    return f"""## Part 3 — Post-Training Mitigation

**Method:** Fairlearn ThresholdOptimizer
**Fairness Axis:** savings_band (primary), employment_status (fallback)
**Constraint:** demographic_parity
**Result:** {status}

### How It Works

ThresholdOptimizer adjusts the GREEN decision threshold per group without retraining the model.
The model weights, trees, and parameters stay exactly the same. Only where the decision line
sits for each savings group is adjusted. After fitting, bias detection re-runs on the same
validation set to verify improvement.

Labels were binarized to GREEN vs not-GREEN before fitting — ThresholdOptimizer is binary only.
After optimization, binary predictions were mapped back to GREEN/RED/YELLOW for re-evaluation.

---
"""


def _deployment_section(
    bias_passed: bool,
    all_flags: List[str],
    fairness_metrics: Dict[str, float],
) -> str:
    savings_eod = fairness_metrics.get("bias_eod_savings_band", "N/A")
    nzs_f1 = fairness_metrics.get("grp_savings_band_near_zero_savings_f1", "N/A")
    agg_f1 = fairness_metrics.get("aggregate_f1", "N/A")

    if bias_passed:
        readiness = "[READY] READY — bias gate passed with zero gate-blocking flags."
    else:
        readiness = "[NOT READY] NOT READY — bias gate failed. Review flags before deploying."

    nzs_status = "[PASSED]" if isinstance(nzs_f1, float) and nzs_f1 > 0.95 else "[CHECK]"
    agg_status = "[PASSED]" if isinstance(agg_f1, float) and agg_f1 > 0.95 else "[CHECK]"

    return f"""## Deployment Readiness

**Status:** {readiness}

### Key Numbers to Verify Before Deployment

| Check | Value | Status |
|---|---|---|
| Bias gate passed | {bias_passed} | {'[PASSED]' if bias_passed else '[FAILED]'} |
| Gate-blocking flags | {len(all_flags)} | {'[PASSED]' if len(all_flags) == 0 else '[FAILED]'} |
| EOD_savings_band (monitor) | {savings_eod} | [MONITOR] Monitor — by design, not bias |
| near_zero_savings F1 | {nzs_f1} | {nzs_status} |
| Aggregate F1 | {agg_f1} | {agg_status} |

### Critical Check Before Deployment — Leakage Audit

Confirm these columns are in `Config.COLUMNS_TO_DROP` and never in `X_train`:
- `affordability_score` — used directly by deterministic labeling engine
- `discretionary_income` — used directly by deterministic labeling engine
- `emergency_fund_months` — used directly by deterministic labeling engine
- `savings_to_price_ratio` — used directly by deterministic labeling engine
- `over_leveraged_flag` — label-correlated, added by bias mitigation
- `synthetic_flag` — does not exist at inference time
- `sample_weight` — training control only

### Monitor Every Future Run

`EOD_savings_band` is logged as an MLflow tag every run. If it drops significantly
from its current value, investigate — it could mean the model is becoming less
conservative about near-zero savings users.

---
"""


def _footer() -> str:
    return """## Files Involved

| File | Role |
|---|---|
| `src/data/bias_mitigation.py` | Pre-training mitigation — all 8 mitigations |
| `src/guards/bias_detection.py` | Post-training detection + ThresholdOptimizer mitigation |
| `src/guards/bias_report.py` | This report generator |
| `src/run_pipeline.py` | Calls all three in sequence |

---
*Report generated automatically by SavVio model pipeline.*
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_bias_report(
    fairness_metrics: Dict[str, float],
    model_name: str,
    bias_passed: bool,
    all_flags: Optional[List[str]] = None,
    mitigation_applied: bool = False,
    mitigation_successful: bool = False,
    final_metrics: Optional[Dict[str, float]] = None,
    output_dir: str = "reports/",
) -> str:
    """
    Generate a complete bias analysis report and save to output_dir.

    Args:
        fairness_metrics:      Flat dict of all bias metrics from evaluate_bias().
        model_name:            Name of the champion model e.g. 'lightgbm_tuned'.
        bias_passed:           Whether the bias gate passed.
        all_flags:             List of gate-blocking flag strings.
        mitigation_applied:    Whether ThresholdOptimizer was applied.
        mitigation_successful: Whether mitigation cleared the gate.
        final_metrics:         Final test set metrics dict from evaluate_model().
        output_dir:            Directory to write the report to.

    Returns:
        Path to the generated timestamped report file.
    """
    all_flags = all_flags or []
    final_metrics = final_metrics or {}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build report
    report = ""
    report += _header(model_name, bias_passed, timestamp)
    report += _summary_section(
        bias_passed, all_flags, final_metrics,
        mitigation_applied, mitigation_successful,
    )
    report += _pretraining_section()
    report += _posttraining_section(fairness_metrics, all_flags, bias_passed)
    report += _mitigation_section(mitigation_applied, mitigation_successful, bias_passed)
    report += _deployment_section(bias_passed, all_flags, fairness_metrics)
    report += _footer()

    # Write to output directory
    os.makedirs(output_dir, exist_ok=True)

    # Timestamped version — keeps history of every run
    timestamped_path = os.path.join(
        output_dir, f"bias_report_{model_name}_{file_timestamp}.md"
    )
    # Latest version — always overwritten, easy to find
    latest_path = os.path.join(output_dir, "bias_report_latest.md")

    for path in (timestamped_path, latest_path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)

    logger.info("Bias report saved to %s", timestamped_path)
    logger.info("Latest bias report at %s", latest_path)

    return timestamped_path