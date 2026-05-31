"""
Post-Training Bias Detection for SavVio Model Pipeline.

Called from run_pipeline.py inside train_candidates() after each model
is evaluated on the validation set.

Contract (run_pipeline.py):
    model, bias_passed, metrics = detect_and_mitigate(
        model, X_train, y_train, X_val, y_val, y_pred_val,
        sens_train, sens_val,
        scenarios_raw=scenarios_val,  # row-aligned with val (not full scenarios_raw)
        y_prob_val=y_prob_val,
    )

Slices checked (17 total):
    Demographic (2) : employment_status, region
    Financial  (10) : income_band, dti_band, savings_band,
                      emergency_fund_band, discretionary_income_band,
                      saving_to_income_band, expense_burden_band,
                      affordability_band, has_loan, over_leveraged
    Product    (5)  : price_band, review_confidence_band,
                      rating_variance_band, average_rating_band,
                      cold_start

Metrics per group:
    - Accuracy
    - Weighted F1
    - GREEN rate (fraction predicted GREEN)
    - Demographic Parity Difference (DPD) via Fairlearn
    - Equalized Odds Difference (EOD) via Fairlearn

Flags bias if:
    - Any group F1 drops > F1_DISPARITY_THRESHOLD (default 0.10) below aggregate F1
    - Any group F1 < MIN_GROUP_F1 (0.50)
    - DPD/EOD above slice-specific thresholds (demographic / financial / product)

Outputs:
    - Disparity summary table printed to terminal
    - F1 bar chart logged to MLflow as artifact
    - All metrics logged to active MLflow run
    - bias_passed bool returned to run_pipeline.py
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds — different per slice category
# ---------------------------------------------------------------------------

# Demographic slices (region, employment_status):
# DPD and EOD should be close to 0 — the model must not give GREEN at
# different rates to people based on where they live or their job status.
DEMOGRAPHIC_DPD_THRESHOLD = getattr(Config, "BIAS_DISPARITY_THRESHOLD", 0.10)
DEMOGRAPHIC_EOD_THRESHOLD = getattr(Config, "BIAS_DISPARITY_THRESHOLD", 0.10)

# Financial slices (income, savings, DTI, affordability etc.):
# DPD is intentionally NOT checked — it is correct and expected that
# low-income / near-zero-savings users get GREEN less often. That is the
# entire purpose of SavVio. Applying demographic parity here is a
# category error. We only check EOD (equal error rates) and F1 disparity.
FINANCIAL_EOD_THRESHOLD  = 0.15   # slightly relaxed — some error variation is acceptable
FINANCIAL_DPD_THRESHOLD  = None   # not checked — GREEN rate differences are by design

# Product slices (price_band, rating_variance, review_confidence etc.):
# DPD loosely checked — some GREEN rate difference by price is expected
# (premium products are more expensive so fewer get GREEN). EOD checked
# to ensure the model's error rate is consistent across product types.
PRODUCT_DPD_THRESHOLD    = 0.30
PRODUCT_EOD_THRESHOLD    = 0.15

# F1 thresholds apply to all slices equally.
MIN_GROUP_F1             = 0.50   # absolute floor — no group should be below this
F1_DISPARITY_THRESHOLD   = getattr(Config, "BIAS_DISPARITY_THRESHOLD", 0.10)    
MIN_GROUP_SIZE           = 30    # compute and show metrics for groups this size
GATE_MIN_GROUP_SIZE      = 100   # only block gate for groups this size and above # skip groups with fewer samples

# Slice category mapping — used to apply the right thresholds per slice.
DEMOGRAPHIC_SLICES = {"region", "employment_status"}
FINANCIAL_SLICES   = {
    "income_band", "dti_band", "savings_band", "emergency_fund_band",
    "discretionary_income_band", "saving_to_income_band",
    "expense_burden_band", "affordability_band", "has_loan", "over_leveraged",
}

# EOD on these financial slices is logged + MLflow-tagged but does **not** fail the gate.
# Large EOD here often reflects structural label/base-rate differences by design
# (liquidity, affordability, runway), not necessarily unfair error rates.
FINANCIAL_EOD_MONITOR_ONLY_SLICES = frozenset({
    "savings_band", "affordability_band", "emergency_fund_band",
})
PRODUCT_SLICES     = {
    "price_band", "review_confidence_band", "rating_variance_band",
    "average_rating_band", "cold_start",
}


# ---------------------------------------------------------------------------
# Slice builders
# Each function takes scenarios_raw DataFrame and returns a pd.Series of
# group labels, or None if the required column is not present.
# ---------------------------------------------------------------------------

def _income_band(df: pd.DataFrame) -> Optional[pd.Series]:
    if "monthly_income" not in df.columns:
        return None
    return pd.cut(
        pd.to_numeric(df["monthly_income"], errors="coerce"),
        bins=[-float("inf"), 3_000, 7_000, float("inf")],
        labels=["low_income", "mid_income", "high_income"],
    ).astype(str)


def _dti_band(df: pd.DataFrame) -> Optional[pd.Series]:
    if "debt_to_income_ratio" not in df.columns:
        return None
    return pd.cut(
        pd.to_numeric(df["debt_to_income_ratio"], errors="coerce"),
        bins=[-float("inf"), 0.20, 0.40, float("inf")],
        labels=["safe_dti", "warning_dti", "risky_dti"],
    ).astype(str)


def _savings_band(df: pd.DataFrame) -> Optional[pd.Series]:
    col = next((c for c in df.columns if c in {"liquid_savings", "savings_balance"}), None)
    if col is None:
        return None
    return pd.cut(
        pd.to_numeric(df[col], errors="coerce"),
        bins=[-float("inf"), 500, 3_000, float("inf")],
        labels=["near_zero_savings", "low_savings", "healthy_savings"],
    ).astype(str)


def _emergency_fund_band(df: pd.DataFrame) -> Optional[pd.Series]:
    if "emergency_fund_months" not in df.columns:
        return None
    return pd.cut(
        pd.to_numeric(df["emergency_fund_months"], errors="coerce"),
        bins=[-float("inf"), 1.0, 3.0, float("inf")],
        labels=["critical_efm", "fragile_efm", "stable_efm"],
    ).astype(str)


def _discretionary_income_band(df: pd.DataFrame) -> Optional[pd.Series]:
    if "discretionary_income" not in df.columns:
        return None
    return pd.cut(
        pd.to_numeric(df["discretionary_income"], errors="coerce"),
        bins=[-float("inf"), 0, 1_000, float("inf")],
        labels=["negative_di", "tight_di", "comfortable_di"],
    ).astype(str)


def _saving_to_income_band(df: pd.DataFrame) -> Optional[pd.Series]:
    if "saving_to_income_ratio" not in df.columns:
        return None
    return pd.cut(
        pd.to_numeric(df["saving_to_income_ratio"], errors="coerce"),
        bins=[-float("inf"), 0.25, 1.0, float("inf")],
        labels=["fragile_stir", "moderate_stir", "strong_stir"],
    ).astype(str)


def _expense_burden_band(df: pd.DataFrame) -> Optional[pd.Series]:
    if "monthly_expense_burden_ratio" not in df.columns:
        return None
    return pd.cut(
        pd.to_numeric(df["monthly_expense_burden_ratio"], errors="coerce"),
        bins=[-float("inf"), 0.5, 0.8, float("inf")],
        labels=["comfortable_mebr", "tight_mebr", "overstretched_mebr"],
    ).astype(str)


def _affordability_band(df: pd.DataFrame) -> Optional[pd.Series]:
    if "affordability_score" not in df.columns:
        return None
    vals = pd.to_numeric(df["affordability_score"], errors="coerce")
    result = pd.Series("adequate_afford", index=df.index)
    result[vals < 0]                   = "below_zero_afford"
    result[(vals >= 0) & (vals < 500)] = "low_afford"
    return result


def _has_loan_slice(df: pd.DataFrame) -> Optional[pd.Series]:
    if "has_loan" not in df.columns:
        return None
    lowered = df["has_loan"].astype(str).str.strip().str.lower()
    result = pd.Series("no_loan", index=df.index)
    result[lowered.isin({"1", "true", "yes", "t", "y"})] = "has_loan"
    return result


def _over_leveraged_slice(df: pd.DataFrame) -> Optional[pd.Series]:
    if "over_leveraged_flag" not in df.columns:
        return None
    return df["over_leveraged_flag"].astype(str).map(
        {"0": "normal", "1": "over_leveraged"}
    )


def _price_band(df: pd.DataFrame) -> Optional[pd.Series]:
    col = next((c for c in df.columns if c in {"product_price", "price"}), None)
    if col is None:
        return None
    return pd.cut(
        pd.to_numeric(df[col], errors="coerce"),
        bins=[-float("inf"), 25, 200, float("inf")],
        labels=["budget", "mid_range", "premium"],
    ).astype(str)


def _review_confidence_band(df: pd.DataFrame) -> Optional[pd.Series]:
    if "rating_number" not in df.columns:
        return None
    vals = pd.to_numeric(df["rating_number"], errors="coerce")
    result = pd.Series("medium_confidence", index=df.index)
    result[vals < 10]  = "low_confidence"
    result[vals > 100] = "high_confidence"
    return result


def _rating_variance_band(df: pd.DataFrame) -> Optional[pd.Series]:
    if "rating_variance" not in df.columns:
        return None
    vals = pd.to_numeric(df["rating_variance"], errors="coerce")
    result = pd.Series("mixed", index=df.index)
    result[vals == 0.0]               = "single_review"
    result[(vals > 0) & (vals < 0.5)] = "consensus"
    result[vals > 1.0]                = "polarized"
    return result


def _average_rating_band(df: pd.DataFrame) -> Optional[pd.Series]:
    if "average_rating" not in df.columns:
        return None
    vals = pd.to_numeric(df["average_rating"], errors="coerce")
    result = pd.Series("medium_rating", index=df.index)
    result[vals <= 3.0] = "low_rating"
    result[vals > 4.0]  = "high_rating"
    return result


def _cold_start_slice(df: pd.DataFrame) -> Optional[pd.Series]:
    if "cold_start_flag" not in df.columns:
        return None
    return df["cold_start_flag"].astype(str).map(
        {"0": "established", "1": "cold_start"}
    )


# ---------------------------------------------------------------------------
# Frame builder — combines demographic + all derived slices
# ---------------------------------------------------------------------------

# Maps slice name → builder function
SLICE_BUILDERS: Dict = {
    # Financial
    "income_band":               _income_band,
    "dti_band":                  _dti_band,
    "savings_band":              _savings_band,
    "emergency_fund_band":       _emergency_fund_band,
    "discretionary_income_band": _discretionary_income_band,
    "saving_to_income_band":     _saving_to_income_band,
    "expense_burden_band":       _expense_burden_band,
    "affordability_band":        _affordability_band,
    "has_loan":                  _has_loan_slice,
    "over_leveraged":            _over_leveraged_slice,
    # Product
    "price_band":                _price_band,
    "review_confidence_band":    _review_confidence_band,
    "rating_variance_band":      _rating_variance_band,
    "average_rating_band":       _average_rating_band,
    "cold_start":                _cold_start_slice,
}


def _build_full_sensitive_frame(
    sensitive_features: pd.DataFrame,
    scenarios_raw: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine demographic columns (employment_status, region) from
    sensitive_features with all derived financial and product slices
    from scenarios_raw.
    """
    frame = sensitive_features.copy().reset_index(drop=True)

    if scenarios_raw is None:
        logger.warning(
            "scenarios_raw not provided — only demographic slices will be checked."
        )
        return frame

    raw = scenarios_raw.reset_index(drop=True)

    active, skipped = [], []
    for name, builder in SLICE_BUILDERS.items():
        result = builder(raw)
        if result is not None:
            frame[name] = result.values
            active.append(name)
        else:
            skipped.append(name)

    logger.info("Active slices (%d): %s", len(frame.columns), list(frame.columns))
    if skipped:
        logger.debug("Skipped slices (column not in scenarios): %s", skipped)

    return frame


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _compute_auc(
    y_true: np.ndarray,
    y_prob: Optional[np.ndarray],
    classes: np.ndarray,
) -> Optional[float]:
    if y_prob is None or len(classes) < 2:
        return None
    try:
        y_bin = label_binarize(y_true, classes=classes)
        if y_bin.shape[1] < 2:
            return None
        return float(roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-group metrics computation
# ---------------------------------------------------------------------------

def _compute_all_slice_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    full_sf: pd.DataFrame,
    classes: np.ndarray,
    green_class_idx: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute accuracy, F1, AUC, GREEN rate, and F1 disparity per group
    for every slice column. Returns a summary DataFrame and a flat dict
    ready for mlflow.log_metrics().
    """
    green_idx = green_class_idx if green_class_idx is not None else int(classes.min())
    agg_acc   = float(accuracy_score(y_true, y_pred))
    agg_f1    = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    agg_auc   = _compute_auc(y_true, y_prob, classes)
    agg_green = float((y_pred == green_idx).mean())

    rows = [{
        "slice": "AGGREGATE", "group": "all",
        "n": len(y_true),
        "accuracy": round(agg_acc, 4),
        "f1": round(agg_f1, 4),
        "auc": round(agg_auc, 4) if agg_auc else None,
        "green_rate": round(agg_green, 4),
        "f1_disparity": 0.0,
    }]
    flat: Dict[str, float] = {
        "aggregate_accuracy":   round(agg_acc, 4),
        "aggregate_f1":         round(agg_f1, 4),
        "aggregate_green_rate": round(agg_green, 4),
    }
    if agg_auc:
        flat["aggregate_auc"] = round(agg_auc, 4)

    for col in full_sf.columns:
        sf_col = full_sf[col].astype(str)

        for grp in sorted(sf_col.unique()):
            mask = (sf_col == grp).values
            n = int(mask.sum())
            if n < MIN_GROUP_SIZE:
                continue

            g_true = y_true[mask]
            g_pred = y_pred[mask]
            g_prob = y_prob[mask] if y_prob is not None else None

            g_acc   = float(accuracy_score(g_true, g_pred))
            g_f1    = float(f1_score(g_true, g_pred, average="weighted", zero_division=0))
            g_auc   = _compute_auc(g_true, g_prob, classes)
            g_green = float((g_pred == green_idx).mean())
            g_disp  = round(agg_f1 - g_f1, 4)

            safe = lambda s: str(s).replace(" ", "_").replace("-", "_")
            p = f"grp_{safe(col)}_{safe(grp)}"

            flat[f"{p}_accuracy"]     = round(g_acc, 4)
            flat[f"{p}_f1"]           = round(g_f1, 4)
            flat[f"{p}_green_rate"]   = round(g_green, 4)
            flat[f"{p}_f1_disparity"] = g_disp
            if g_auc:
                flat[f"{p}_auc"] = round(g_auc, 4)

            rows.append({
                "slice": col, "group": grp, "n": n,
                "accuracy": round(g_acc, 4),
                "f1": round(g_f1, 4),
                "auc": round(g_auc, 4) if g_auc else None,
                "green_rate": round(g_green, 4),
                "f1_disparity": g_disp,
            })

    return pd.DataFrame(rows), flat


# ---------------------------------------------------------------------------
# DPD and EOD via Fairlearn
# ---------------------------------------------------------------------------

def _compute_dpd_eod(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    full_sf: pd.DataFrame,
    classes: np.ndarray,
    green_class_idx: Optional[int] = None,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Compute DPD and EOD for each sensitive feature column.
    Binary: GREEN vs not-GREEN.

    Thresholds applied per slice category:
    - Demographic: DPD ≤ 0.10, EOD ≤ 0.10
    - Financial:   DPD not checked (by design), EOD ≤ 0.15 for gate — except
      ``savings_band``, ``affordability_band``, ``emergency_fund_band`` are
      EOD **monitor-only** (logged / tagged, do not fail the gate).
    - Product:     DPD ≤ 0.25, EOD ≤ 0.15
    """
    green_idx  = green_class_idx if green_class_idx is not None else int(classes.min())
    y_bin_true = (pd.Series(y_true) == green_idx).astype(int).reset_index(drop=True)
    y_bin_pred = (pd.Series(y_pred) == green_idx).astype(int).reset_index(drop=True)

    metrics: Dict[str, float] = {}
    flags:   List[str]        = []

    for col in full_sf.columns:
        sf_col = full_sf[col].astype(str).reset_index(drop=True)

        # Determine which threshold set applies to this slice.
        if col in DEMOGRAPHIC_SLICES:
            dpd_thresh = DEMOGRAPHIC_DPD_THRESHOLD
            eod_thresh = DEMOGRAPHIC_EOD_THRESHOLD
            check_dpd  = True
        elif col in FINANCIAL_SLICES:
            dpd_thresh = None   # not checked
            eod_thresh = FINANCIAL_EOD_THRESHOLD
            check_dpd  = False  # GREEN rate differences are by design
        else:  # product slices
            dpd_thresh = PRODUCT_DPD_THRESHOLD
            eod_thresh = PRODUCT_EOD_THRESHOLD
            check_dpd  = True

        try:
            dpd = float(demographic_parity_difference(
                y_bin_true, y_bin_pred, sensitive_features=sf_col
            ))
            eod = float(equalized_odds_difference(
                y_bin_true, y_bin_pred, sensitive_features=sf_col
            ))

            metrics[f"bias_dpd_{col}"] = round(dpd, 4)
            metrics[f"bias_eod_{col}"] = round(eod, 4)

            logger.info(
                "  [%s | %s] DPD=%.4f (check=%s, thresh=%s)  EOD=%.4f (thresh=%.2f)",
                col,
                "demographic" if col in DEMOGRAPHIC_SLICES
                else "financial" if col in FINANCIAL_SLICES
                else "product",
                dpd, check_dpd,
                f"{dpd_thresh:.2f}" if dpd_thresh is not None else "n/a",
                eod, eod_thresh,
            )

            if check_dpd and dpd_thresh is not None and abs(dpd) > dpd_thresh:
                flag = f"DPD_{col}={dpd:.4f}"
                flags.append(flag)
                logger.warning("  BIAS FLAG: %s", flag)

            if abs(eod) > eod_thresh:
                flag = f"EOD_{col}={eod:.4f}"
                if col in FINANCIAL_EOD_MONITOR_ONLY_SLICES:
                    logger.warning(
                        "  MONITOR (EOD not gate-blocking): %s [thresh=%.2f]",
                        flag, eod_thresh,
                    )
                    try:
                        mlflow.set_tag(f"monitor_eod_{col}", flag)
                    except Exception:
                        pass
                else:
                    flags.append(flag)
                    logger.warning("  BIAS FLAG: %s", flag)

        except Exception as e:
            logger.warning("DPD/EOD failed for %s: %s", col, e)

    return metrics, flags


# ---------------------------------------------------------------------------
# Disparity summary table — terminal output
# ---------------------------------------------------------------------------

def _print_disparity_table(slice_df: pd.DataFrame) -> None:
    agg = slice_df[slice_df["slice"] == "AGGREGATE"].iloc[0]
    logger.info("\n%s", "=" * 75)
    logger.info("DISPARITY SUMMARY TABLE")
    logger.info("%-28s %-22s %6s %6s %6s %8s %s",
                "slice", "group", "n", "acc", "f1", "ΔF1", "flag")
    logger.info("-" * 75)
    logger.info("%-28s %-22s %6d %6.3f %6.3f %8s",
                "AGGREGATE", "all", agg["n"], agg["accuracy"], agg["f1"], "—")
    logger.info("-" * 75)

    for _, row in slice_df[slice_df["slice"] != "AGGREGATE"].iterrows():
        flag = ""
        if abs(row["f1_disparity"]) > F1_DISPARITY_THRESHOLD:
            flag += " [!] DISPARITY"
        if row["f1"] < MIN_GROUP_F1:
            flag += " [!] LOW F1"
        logger.info("%-28s %-22s %6d %6.3f %6.3f %8.4f%s",
                    row["slice"], row["group"], row["n"],
                    row["accuracy"], row["f1"], row["f1_disparity"], flag)

    logger.info("=" * 75)


# ---------------------------------------------------------------------------
# F1 bar chart
# ---------------------------------------------------------------------------

def _plot_f1_chart(slice_df: pd.DataFrame) -> Optional[str]:
    try:
        plot_df = slice_df[slice_df["slice"] != "AGGREGATE"].copy()
        if plot_df.empty:
            return None

        agg_f1 = float(slice_df.loc[slice_df["slice"] == "AGGREGATE", "f1"].iloc[0])

        colors = []
        for f1 in plot_df["f1"]:
            if f1 < MIN_GROUP_F1:
                colors.append("#d62728")   # red — below floor
            elif (agg_f1 - f1) > F1_DISPARITY_THRESHOLD:
                colors.append("#ff7f0e")   # orange — disparity
            else:
                colors.append("#2ca02c")   # green — passing

        labels = [f"{r['slice']}\n{r['group']}" for _, r in plot_df.iterrows()]

        fig, ax = plt.subplots(figsize=(max(14, len(plot_df) * 0.6), 5))
        ax.bar(range(len(plot_df)), plot_df["f1"], color=colors, width=0.6)
        ax.axhline(agg_f1, color="steelblue", linestyle="--", linewidth=1.5,
                   label=f"Aggregate F1 = {agg_f1:.3f}")
        ax.axhline(MIN_GROUP_F1, color="red", linestyle=":", linewidth=1.2,
                   label=f"Min F1 floor = {MIN_GROUP_F1:.2f}")
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Weighted F1")
        ax.set_title("Per-slice F1 — green=pass, orange=disparity, red=below floor")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, dpi=120)
        plt.close(fig)
        return tmp.name

    except Exception as e:
        logger.warning("F1 chart generation failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Public API — called from run_pipeline.py
# ---------------------------------------------------------------------------

def evaluate_bias(
    y_test,
    y_pred,
    sensitive_features: pd.DataFrame,
    y_prob: Optional[np.ndarray] = None,
    scenarios_raw: Optional[pd.DataFrame] = None,
    green_class_idx: Optional[int] = None,
) -> Tuple[Dict[str, float], bool]:
    """
    Post-training bias detection across 17 slices.

    Args:
        y_test:             True labels (encoded ints) on validation set.
        y_pred:             Model predictions (encoded ints) on validation set.
        sensitive_features: DataFrame with employment_status, region columns.
        y_prob:             Predicted probabilities shape (n, n_classes). Optional.
        scenarios_raw:      Raw scenario DataFrame for deriving financial
                            and product slices. Optional but recommended.
        green_class_idx:    Encoded integer for the GREEN label. Derive from
                            label_encoder.transform(["GREEN"])[0] at the call
                            site. When None, falls back to classes.min() which
                            assumes alphabetical LabelEncoder order (GREEN=0).

    Returns:
        (fairness_metrics dict, bias_passed bool, all_flags list)
        ``all_flags`` contains the string identifiers of every violated
        threshold and is used by detect_and_mitigate to select the
        ThresholdOptimizer mitigation axis.
    """
    logger.info("=" * 60)
    logger.info("POST-TRAINING BIAS DETECTION")
    logger.info("=" * 60)

    y_true_arr = np.asarray(y_test)
    y_pred_arr = np.asarray(y_pred)
    classes    = np.unique(y_true_arr)

    # Step 1 — build full sensitive frame (demographic + financial + product)
    full_sf = _build_full_sensitive_frame(
        sensitive_features.reset_index(drop=True),
        scenarios_raw,
    )

    # Step 2 — per-group metrics (accuracy, F1, AUC, GREEN rate, disparity)
    slice_df, flat_metrics = _compute_all_slice_metrics(
        y_true_arr, y_pred_arr, y_prob, full_sf, classes,
        green_class_idx=green_class_idx,
    )

    # Step 3 — DPD and EOD via Fairlearn
    dpd_eod_metrics, dpd_eod_flags = _compute_dpd_eod(
        y_true_arr, y_pred_arr, full_sf, classes,
        green_class_idx=green_class_idx,
    )
    flat_metrics.update(dpd_eod_metrics)

    # Step 4 — collect all bias flags
    all_flags = list(dpd_eod_flags)
            
    for _, row in slice_df[slice_df["slice"] != "AGGREGATE"].iterrows():
        if row["n"] < GATE_MIN_GROUP_SIZE:
           continue  # too small for stable metrics, skip gate blocking
        if abs(row["f1_disparity"]) > F1_DISPARITY_THRESHOLD:
            all_flags.append(
               f"F1_disparity_{row['slice']}={row['group']}({row['f1_disparity']:.4f})"
            )
        if row["f1"] < MIN_GROUP_F1:
            all_flags.append(
                f"low_f1_{row['slice']}={row['group']}({row['f1']:.4f})"
            )

    # Step 5 — print disparity table to terminal
    _print_disparity_table(slice_df)

    # Step 6 — generate and log F1 bar chart to MLflow
    chart_path = _plot_f1_chart(slice_df)
    if chart_path:
        try:
            mlflow.log_artifact(chart_path, artifact_path="bias_report")
            os.unlink(chart_path)
        except Exception as e:
            logger.warning("Could not log F1 chart to MLflow: %s", e)

    # Step 7 — determine bias_passed
    bias_passed = len(all_flags) == 0
    flat_metrics["bias_gate_passed"] = int(bias_passed)
    flat_metrics["bias_flags_count"] = len(all_flags)

    if all_flags:
        logger.warning("Bias gate FAILED — %d flag(s):", len(all_flags))
        for f in all_flags[:10]:
            logger.warning("  %s", f)
    else:
        logger.info("Bias gate PASSED — no disparities above threshold.")

    # Step 8 — log all metrics to MLflow
    try:
        mlflow.log_metrics(flat_metrics)
        if all_flags:
            mlflow.set_tag("bias_flags", "; ".join(all_flags[:5]))
        mlflow.set_tag("bias_slices_checked", str(list(full_sf.columns)))
    except Exception as e:
        logger.warning("MLflow logging failed in evaluate_bias: %s", e)

    return flat_metrics, bias_passed, all_flags


# ---------------------------------------------------------------------------
# Post-training mitigation — ThresholdOptimizer (binary GREEN vs rest)
# ---------------------------------------------------------------------------

def _green_class_index(y: np.ndarray) -> int:
    """
    Encoded class index for GREEN, aligned with DPD/EOD in _compute_dpd_eod
    (uses smallest label value as GREEN).
    """
    return int(np.unique(np.asarray(y)).min())


def attach_savings_band_to_sensitive(
    sens_df: Optional[pd.DataFrame],
    scenarios_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """
    Add ``savings_band`` to the sensitive-features frame using scenario columns
    (``liquid_savings`` / ``savings_balance``). Row alignment must match
    ``scenarios_df``.
    """
    if sens_df is None:
        return None
    sb = _savings_band(scenarios_df)
    if sb is None:
        return sens_df
    out = sens_df.copy()
    out["savings_band"] = sb.astype(str).values
    return out


def _primary_mitigation_axis(
    sensitive_df: pd.DataFrame,
    dpd_eod_flags: Optional[List[str]] = None,
) -> str:
    """
    Sensitive column for ThresholdOptimizer.

    When ``dpd_eod_flags`` is provided (from _compute_dpd_eod), pick the
    flagged column that is present in ``sensitive_df``, preferring demographic
    slices over financial ones so the optimizer targets the root cause.
    Falls back to the hard-coded priority order when no flagged column is
    available in ``sensitive_df``.
    """
    if dpd_eod_flags:
        flagged_cols: List[str] = []
        for flag in dpd_eod_flags:
            for prefix in ("DPD_", "EOD_"):
                if flag.startswith(prefix):
                    col = flag[len(prefix):].split("=")[0]
                    if col in sensitive_df.columns and col not in flagged_cols:
                        flagged_cols.append(col)
                    break
        # Prefer demographic slices (the model must not discriminate on these).
        for col in flagged_cols:
            if col in DEMOGRAPHIC_SLICES:
                return col
        # Then any other flagged column present in sensitive_df.
        for col in flagged_cols:
            return col

    # Hard-coded fallback priority.
    for col in ("savings_band", "employment_status", "region"):
        if col in sensitive_df.columns:
            return col
    return sensitive_df.columns[0]


def is_fairlearn_threshold_optimizer(model: object) -> bool:
    """True if ``model`` is Fairlearn's ThresholdOptimizer (needs ``sensitive_features``)."""
    try:
        from fairlearn.postprocessing import ThresholdOptimizer
        return isinstance(model, ThresholdOptimizer)
    except ImportError:
        return False


def predict_with_sensitive_features(
    model: object,
    X: pd.DataFrame,
    sensitive: Optional[pd.DataFrame] = None,
):
    """
    Call ``predict`` for sklearn / MLflow. Fairlearn ``ThresholdOptimizer.predict``
    requires ``sensitive_features`` aligned with ``X``.
    """
    if sensitive is None or not is_fairlearn_threshold_optimizer(model):
        return model.predict(X)
    col = _primary_mitigation_axis(sensitive)
    return model.predict(
        X,
        sensitive_features=sensitive[col].astype(str),
    )


class _GreenBinaryProbaWrapper(BaseEstimator):
    """
    Fairlearn ThresholdOptimizer is strictly binary and uses predict_proba[:, 1]
    as the positive class score. This wrapper maps a multiclass estimator to
    P(not GREEN), P(GREEN) using the GREEN column from predict_proba.
    """

    def __init__(self, estimator, green_class_idx: int):
        self.estimator = estimator
        self.green_class_idx = int(green_class_idx)

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        p = self.estimator.predict_proba(X)
        p_green = p[:, self.green_class_idx]
        return np.column_stack([1.0 - p_green, p_green])

    def predict(self, X):
        return self.estimator.predict(X)


def _binary_mitigated_to_multiclass(
    y_bin: np.ndarray,
    base_model,
    X: pd.DataFrame,
    green_idx: int,
) -> np.ndarray:
    """
    ThresholdOptimizer.predict returns 0/1 (not GREEN / GREEN).
    Map back to multiclass labels for evaluate_bias.
    """
    y_bin = np.asarray(y_bin).astype(int)
    proba = base_model.predict_proba(X)
    n = len(y_bin)
    out = np.empty(n, dtype=int)
    p2 = np.asarray(proba, dtype=float).copy()
    p2[:, green_idx] = -np.inf
    out_alt = np.argmax(p2, axis=1)
    out = np.where(y_bin == 1, green_idx, out_alt)
    return out


def mitigate_bias(
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sensitive_train: pd.DataFrame,
    green_class_idx: int,
    constraint: str = "demographic_parity",
    dpd_eod_flags: Optional[List[str]] = None,
):
    """
    Apply Fairlearn ThresholdOptimizer to reduce disparity on GREEN vs not-GREEN.

    ThresholdOptimizer only supports binary {0,1} labels. We train it on
    y_binary = (y == green_class_idx) with a wrapper that exposes P(GREEN) in
    predict_proba[:, 1].     Uses `prefit=True` so the multiclass base model is not
    re-fit on binary labels.

    The mitigation axis is chosen from the actual flagged slices in
    ``dpd_eod_flags`` (demographic columns preferred), falling back to a
    hard-coded priority order when no flagged column is available.
    """
    primary_col = _primary_mitigation_axis(sensitive_train, dpd_eod_flags=dpd_eod_flags)
    primary_sf = sensitive_train[primary_col].astype(str)

    y_train_arr = np.asarray(y_train)
    y_binary = (y_train_arr == green_class_idx).astype(np.int64)

    wrapped = _GreenBinaryProbaWrapper(model, green_class_idx)

    logger.info(
        "Applying ThresholdOptimizer (constraint=%s, feature=%s, GREEN class=%s)...",
        constraint, primary_col, green_class_idx,
    )

    optimizer = ThresholdOptimizer(
        estimator=wrapped,
        constraints=constraint,
        objective="balanced_accuracy_score",
        predict_method="predict_proba",
        grid_size=50,
        prefit=True,
    )
    optimizer.fit(X_train, y_binary, sensitive_features=primary_sf)
    logger.info("ThresholdOptimizer fitted successfully.")
    return optimizer


# ---------------------------------------------------------------------------
# Orchestrator — detect + mitigate, called from run_pipeline.py
# ---------------------------------------------------------------------------

def detect_and_mitigate(
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    y_pred_val: np.ndarray,
    sensitive_train: pd.DataFrame,
    sensitive_val: pd.DataFrame,
    scenarios_raw: Optional[pd.DataFrame] = None,
    y_prob_val: Optional[np.ndarray] = None,
    green_class_idx: Optional[int] = None,
) -> Tuple[object, bool, Dict[str, float]]:
    """
    Full post-training bias detection + conditional mitigation.

    Steps:
        1. Detect bias on validation predictions
        2. If bias_passed = False → apply ThresholdOptimizer
        3. Re-evaluate mitigated model on **X_val** (same rows as y_val / sensitive_val)
        4. Return (final_model, bias_passed, fairness_metrics)

    Args:
        scenarios_raw:    Per evaluate_bias — typically **scenarios_val** (row-aligned with val).
        green_class_idx:  Encoded integer for the GREEN label from label_encoder.
                          When None, falls back to the minimum encoded class value.
    """
    # Step 1 — detect
    fairness_metrics, bias_passed, dpd_eod_flags = evaluate_bias(
        y_val, y_pred_val, sensitive_val,
        y_prob=y_prob_val,
        scenarios_raw=scenarios_raw,
        green_class_idx=green_class_idx,
    )

    if bias_passed:
        mlflow.log_metric("bias_mitigation_applied", 0)
        return model, True, fairness_metrics

    green_idx = green_class_idx if green_class_idx is not None else _green_class_index(y_train)

    # Step 2 — mitigate (binary GREEN vs rest; Fairlearn requires 0/1 labels)
    logger.warning("Bias gate FAILED — applying ThresholdOptimizer.")
    try:
        mitigated_model = mitigate_bias(
            model, X_train, y_train, sensitive_train,
            green_class_idx=green_idx,
            constraint="demographic_parity",
            dpd_eod_flags=dpd_eod_flags,
        )

        # Step 3 — re-predict on validation using the same axis the optimizer was fit on.
        primary_col = _primary_mitigation_axis(sensitive_val, dpd_eod_flags=dpd_eod_flags)
        primary_sf_val = sensitive_val[primary_col].astype(str)

        if len(X_val) != len(y_val) or len(primary_sf_val) != len(y_val):
            raise ValueError(
                "X_val, y_val, and sensitive_val must have the same length for mitigation re-eval."
            )

        y_pred_bin = mitigated_model.predict(
            X_val,
            sensitive_features=primary_sf_val,
        )
        # Optimizer outputs 0/1; convert to multiclass for the same metric pipeline
        y_pred_mitigated = _binary_mitigated_to_multiclass(
            y_pred_bin, model, X_val, green_idx,
        )

        logger.info("Re-evaluating after mitigation...")
        fairness_metrics_after, bias_passed_after, _ = evaluate_bias(
            y_val, y_pred_mitigated, sensitive_val,
            scenarios_raw=scenarios_raw,
            green_class_idx=green_class_idx,
        )

        mlflow.log_metric("bias_mitigation_applied", 1)
        mlflow.log_metric("bias_mitigation_successful", int(bias_passed_after))

        if bias_passed_after:
            logger.info("Mitigation successful — bias gate now PASSED.")
        else:
            logger.warning("Mitigation applied but bias still present — flagged for review.")

        return model, bias_passed_after, fairness_metrics_after

    except Exception as e:
        logger.error("ThresholdOptimizer mitigation failed: %s", e, exc_info=True)
        mlflow.log_metric("bias_mitigation_applied", 0)
        return model, False, fairness_metrics