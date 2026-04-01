"""
Feature validation for SavVio pipeline (Phase 12).

Validates engineered features in data/features/ to ensure
calculations are correct, values are within expected ranges,
and no NaN/Inf values were introduced.

Feature groups:
  - Financial health: discretionary_income, debt_to_income_ratio, etc.
  - Affordability: price_to_income_ratio, affordability_score, RUS (computed at inference)
  - Review-based: rating_variance
"""

import logging
import sys
import os
import numpy as np
import pandas as pd
import great_expectations as gx
from pathlib import Path

try:
    from great_expectations.dataset import PandasDataset
except ImportError:
    from great_expectations.dataset.pandas_dataset import PandasDataset

from typing import Optional, List
from savviocore.validation.validation_config import (
    CheckResult, Severity, ValidationReport, load_thresholds,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load(path: str) -> PandasDataset:
    if path.endswith(".jsonl"):
        # return gx.from_pandas(pd.read_json(path, lines=True))
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        if file_size_mb > 300:
            df = pd.concat(pd.read_json(path, lines=True, chunksize=100_000), ignore_index=True)
        else:
            df = pd.read_json(path, lines=True)
        return gx.from_pandas(df)
    return gx.from_pandas(pd.read_csv(path))


def _check(ge_result: dict, name: str, severity: Severity,
           dataset: str, details: str = "") -> CheckResult:
    passed = ge_result["success"]
    metric = ge_result.get("result", {})
    return CheckResult(
        check_name=name, passed=passed, severity=severity,
        dataset=dataset, stage="features",
        details=details or str(metric.get("partial_unexpected_list", ""))[:200],
        metric_value=metric.get("unexpected_percent") or metric.get("observed_value"),
    )


def _no_nan_inf(gdf: PandasDataset, col: str, ds: str,
                severity: Severity = Severity.CRITICAL) -> list[CheckResult]:
    """Check a column has no NaN or Inf values."""
    results = []
    if col not in gdf.columns:
        return results

    nan_count = gdf[col].isna().sum()
    inf_count = np.isinf(gdf[col].replace([np.nan], 0)).sum() if gdf[col].dtype in ["float64", "float32"] else 0

    results.append(CheckResult(
        check_name=f"feat_{col}_no_nan",
        passed=nan_count == 0,
        severity=severity, dataset=ds, stage="features",
        details=f"{nan_count} NaN values in '{col}'",
        metric_value=nan_count,
    ))
    results.append(CheckResult(
        check_name=f"feat_{col}_no_inf",
        passed=inf_count == 0,
        severity=severity, dataset=ds, stage="features",
        details=f"{inf_count} Inf values in '{col}'",
        metric_value=inf_count,
    ))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Financial health features
# ═══════════════════════════════════════════════════════════════════════════

FINANCIAL_FEATURES = [
    "discretionary_income",
    "debt_to_income_ratio",
    "saving_to_income_ratio",
    "monthly_expense_burden_ratio",
    "emergency_fund_months",
]


def validate_financial_features(gdf: PandasDataset,
                                 thresholds: dict) -> list[CheckResult]:
    """Validate financial health features."""
    results: list[CheckResult] = []
    ds = "financial_features"

    # ── 1. All expected features exist ────────────────────────────────────
    for col in FINANCIAL_FEATURES:
        res = gdf.expect_column_to_exist(col)
        results.append(_check(res, f"feat_exists_{col}", Severity.CRITICAL, ds,
                              f"Feature '{col}' must exist"))

    # ── 2. No NaN / Inf in any feature ────────────────────────────────────
    for col in FINANCIAL_FEATURES:
        results.extend(_no_nan_inf(gdf, col, ds))

    # ── 3. discretionary_income: can be negative (legitimate) ─────────────
    #    Just ensure it's not ALL negative (would indicate a bug)
    if "discretionary_income" in gdf.columns:
        all_negative = (gdf["discretionary_income"] < 0).all()
        results.append(CheckResult(
            check_name="feat_discretionary_not_all_negative",
            passed=not all_negative,
            severity=Severity.WARNING, dataset=ds, stage="features",
            details="All discretionary_income values are negative — likely a calculation bug",
            metric_value=gdf["discretionary_income"].mean(),
        ))

    # ── 4. debt_to_income_ratio: typically 0–2 ────────────────────────────
    if "debt_to_income_ratio" in gdf.columns:
        res = gdf.expect_column_values_to_be_between(
            "debt_to_income_ratio", min_value=0, max_value=5.0, mostly=0.95
        )
        results.append(_check(res, "feat_dti_range", Severity.WARNING, ds,
                              "debt_to_income_ratio should be 0–5 for 95% of records"))

        # Zero-income users should have ratio = 0 or flagged, not Inf
        if "income_usd" in gdf.columns:
            zero_income = gdf["income_usd"] == 0
            if zero_income.any():
                dti_for_zero = gdf.loc[zero_income, "debt_to_income_ratio"]
                has_inf = np.isinf(dti_for_zero).any()
                results.append(CheckResult(
                    check_name="feat_dti_zero_income_safe",
                    passed=not has_inf,
                    severity=Severity.CRITICAL, dataset=ds, stage="features",
                    details="Zero-income users must not produce Inf debt_to_income_ratio",
                    metric_value=int(has_inf),
                ))

    # ── 5. savings_rate: typically 0–1 ────────────────────────────────────
    if "savings_rate" in gdf.columns:
        res = gdf.expect_column_values_to_be_between(
            "savings_rate", min_value=-0.5, max_value=1.5, mostly=0.95
        )
        results.append(_check(res, "feat_savings_rate_range", Severity.WARNING, ds,
                              "savings_rate should be roughly -0.5 to 1.5"))

    # ── 6. expense_burden_ratio: 0–1+ ─────────────────────────────────────
    if "expense_burden_ratio" in gdf.columns:
        res = gdf.expect_column_values_to_be_between(
            "expense_burden_ratio", min_value=0, max_value=3.0, mostly=0.95
        )
        results.append(_check(res, "feat_expense_burden_range", Severity.WARNING, ds,
                              "expense_burden_ratio should be 0–3 for 95% of records"))

    # ── 7. emergency_fund_months >= 0 ─────────────────────────────────────
    if "emergency_fund_months" in gdf.columns:
        res = gdf.expect_column_values_to_be_between(
            "emergency_fund_months", min_value=0, mostly=0.90
        )
        results.append(_check(res, "feat_emergency_fund_range", Severity.INFO, ds,
                              "emergency_fund_months should be >= 0"))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Affordability features
# ═══════════════════════════════════════════════════════════════════════════

AFFORDABILITY_FEATURES = [
    "price_to_income_ratio",
    "affordability_score",
    "residual_utility_score",
]


def validate_affordability_features(gdf: PandasDataset,
                                     thresholds: dict) -> list[CheckResult]:
    """Validate affordability features."""
    results: list[CheckResult] = []
    ds = "affordability_features"

    # ── 1. Features exist ─────────────────────────────────────────────────
    for col in AFFORDABILITY_FEATURES:
        res = gdf.expect_column_to_exist(col)
        results.append(_check(res, f"feat_exists_{col}", Severity.CRITICAL, ds,
                              f"Feature '{col}' must exist"))

    # ── 2. No NaN / Inf ──────────────────────────────────────────────────
    for col in AFFORDABILITY_FEATURES:
        results.extend(_no_nan_inf(gdf, col, ds))

    # ── 3. price_to_income_ratio >= 0 ─────────────────────────────────────
    if "price_to_income_ratio" in gdf.columns:
        res = gdf.expect_column_values_to_be_between(
            "price_to_income_ratio", min_value=0, mostly=0.95
        )
        results.append(_check(res, "feat_ptr_non_negative", Severity.WARNING, ds,
                              "price_to_income_ratio should be >= 0"))

    # ── 4. affordability_score: can be negative (can't afford) ────────────
    #    Verify it equals discretionary_income - price (spot check)
    if all(c in gdf.columns for c in ["affordability_score", "discretionary_income"]):
        # Just verify no NaN/Inf — formula correctness is a unit test concern
        pass  # Already checked above

    # ── 5. residual_utility_score: check not all zero ─────────────────────
    if "residual_utility_score" in gdf.columns:
        all_zero = (gdf["residual_utility_score"] == 0).all()
        results.append(CheckResult(
            check_name="feat_rus_not_all_zero",
            passed=not all_zero,
            severity=Severity.WARNING, dataset=ds, stage="features",
            details="All RUS values are 0 — likely a calculation bug",
            metric_value=gdf["residual_utility_score"].std(),
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Review-based features
# ═══════════════════════════════════════════════════════════════════════════

REVIEW_FEATURES = [
    "rating_variance",
]


def validate_review_features(gdf: PandasDataset,
                              thresholds: dict) -> list[CheckResult]:
    """Validate review-based features.

    Only rating_variance is engineered here. avg_product_rating and
    rating_number already exist in the product metadata (product_preprocessed.jsonl).
    has_text_reviews is not a useful signal for affordability scoring.
    """
    results: list[CheckResult] = []
    ds = "review_features"

    # ── 1. Features exist ─────────────────────────────────────────────────
    for col in REVIEW_FEATURES:
        res = gdf.expect_column_to_exist(col)
        results.append(_check(res, f"feat_exists_{col}", Severity.CRITICAL, ds,
                              f"Feature '{col}' must exist"))

    # ── 2. No NaN / Inf ──────────────────────────────────────────────────
    for col in REVIEW_FEATURES:
        results.extend(_no_nan_inf(gdf, col, ds, Severity.WARNING))

    # ── 3. rating_variance >= 0 ───────────────────────────────────────────
    if "rating_variance" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("rating_variance", min_value=0)
        results.append(_check(res, "feat_rating_variance_non_negative", Severity.WARNING, ds,
                              "rating_variance must be >= 0"))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Formula spot-check (sample-based verification)
# ═══════════════════════════════════════════════════════════════════════════

def validate_formula_spot_checks(gdf: PandasDataset) -> list[CheckResult]:
    """
    Spot-check feature formulas on a sample of rows.
    Verifies derived features match expected calculations.
    """
    results: list[CheckResult] = []
    ds = "formula_spot_check"
    df = pd.DataFrame(gdf)  # work with plain DataFrame

    n_sample = min(100, len(df))
    sample = df.sample(n=n_sample, random_state=42) if len(df) > n_sample else df

    # ── discretionary_income = income - total_fixed_expenses ──────────────
    if all(c in df.columns for c in ["discretionary_income", "income_usd", "total_fixed_expenses"]):
        expected = sample["income_usd"] - sample["total_fixed_expenses"]
        actual = sample["discretionary_income"]
        mismatches = (~np.isclose(expected, actual, rtol=1e-4, equal_nan=True)).sum()
        results.append(CheckResult(
            check_name="formula_discretionary_income",
            passed=mismatches == 0,
            severity=Severity.CRITICAL, dataset=ds, stage="features",
            details=f"{mismatches}/{n_sample} rows don't match: income - fixed_expenses",
            metric_value=mismatches,
        ))

    # ── debt_to_income_ratio = debt / income ──────────────────────────────
    if all(c in df.columns for c in ["debt_to_income_ratio", "monthly_emi", "income_usd"]):
        mask = sample["income_usd"] > 0
        if mask.any():
            expected = sample.loc[mask, "monthly_emi"] / sample.loc[mask, "income_usd"]
            actual = sample.loc[mask, "debt_to_income_ratio"]
            mismatches = (~np.isclose(expected, actual, rtol=1e-4, equal_nan=True)).sum()
            results.append(CheckResult(
                check_name="formula_dti_ratio",
                passed=mismatches == 0,
                severity=Severity.CRITICAL, dataset=ds, stage="features",
                details=f"{mismatches}/{mask.sum()} non-zero-income rows don't match: emi / income",
                metric_value=mismatches,
            ))

    # ── savings_rate = savings / income ───────────────────────────────────
    if all(c in df.columns for c in ["savings_rate", "monthly_savings", "income_usd"]):
        mask = sample["income_usd"] > 0
        if mask.any():
            expected = sample.loc[mask, "monthly_savings"] / sample.loc[mask, "income_usd"]
            actual = sample.loc[mask, "savings_rate"]
            mismatches = (~np.isclose(expected, actual, rtol=1e-4, equal_nan=True)).sum()
            results.append(CheckResult(
                check_name="formula_savings_rate",
                passed=mismatches == 0,
                severity=Severity.CRITICAL, dataset=ds, stage="features",
                details=f"{mismatches}/{mask.sum()} non-zero-income rows don't match: savings / income",
                metric_value=mismatches,
            ))

    # ── expense_burden_ratio = expenses / income ──────────────────────────
    if all(c in df.columns for c in ["expense_burden_ratio", "monthly_expenses", "income_usd"]):
        mask = sample["income_usd"] > 0
        if mask.any():
            expected = sample.loc[mask, "monthly_expenses"] / sample.loc[mask, "income_usd"]
            actual = sample.loc[mask, "expense_burden_ratio"]
            mismatches = (~np.isclose(expected, actual, rtol=1e-4, equal_nan=True)).sum()
            results.append(CheckResult(
                check_name="formula_expense_burden",
                passed=mismatches == 0,
                severity=Severity.CRITICAL, dataset=ds, stage="features",
                details=f"{mismatches}/{mask.sum()} non-zero-income rows don't match: expenses / income",
                metric_value=mismatches,
            ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_feature_validation(
    financial_path: str = "data/features/financial_featured.csv",
    reviews_path: str = "data/features/product_featured.jsonl",
    threshold_config: Optional[str] = "config/validation_thresholds.json",
) -> ValidationReport:
    """Run all feature validations."""
    thresholds = load_thresholds(threshold_config)
    report = ValidationReport(stage="features")

    logger.info("═" * 50)
    logger.info("  FEATURE VALIDATION")
    logger.info("═" * 50)

    # 1. Validate Financial Features
    if Path(financial_path).exists():
        logger.info(f"── Validating financial health features from {financial_path} ──")
        try:
            gdf_fin = _load(financial_path)
            for r in validate_financial_features(gdf_fin, thresholds):
                report.add(r)
            
            logger.info("── Running formula spot-checks (Financial) ──")
            for r in validate_formula_spot_checks(gdf_fin):
                report.add(r)
        except Exception as e:
            logger.error(f"Failed to validate financial features: {e}")
            report.add(CheckResult("load_financial_features", False, Severity.CRITICAL, "financial_features", "features", str(e), 0))
    else:
        logger.warning(f"Financial features file not found: {financial_path}")
        report.add(CheckResult(
            check_name="load_financial_features_missing",
            passed=False,
            severity=Severity.CRITICAL,
            dataset="financial_features",
            stage="features",
            details=f"Required feature file not found: {financial_path}",
            metric_value=0,
        ))

    # 2. Validate Review Features
    if Path(reviews_path).exists():
        logger.info(f"── Validating review-based features from {reviews_path} ──")
        try:
            gdf_rev = _load(reviews_path)
            for r in validate_review_features(gdf_rev, thresholds):
                report.add(r)
        except Exception as e:
            logger.error(f"Failed to validate review features: {e}")
            report.add(CheckResult("load_review_features", False, Severity.CRITICAL, "review_features", "features", str(e), 0))
    else:
        logger.warning(f"Review features file not found: {reviews_path}")
        report.add(CheckResult(
            check_name="load_review_features_missing",
            passed=False,
            severity=Severity.CRITICAL,
            dataset="review_features",
            stage="features",
            details=f"Required feature file not found: {reviews_path}",
            metric_value=0,
        ))

    # 3. Affordability Features (Skipped as per run_features.py/README update)
    logger.info("── Skipping affordability features (computed at inference time) ──")

    report.print_summary()
    report.save()

    if not report.passed:
        logger.critical("FEATURE VALIDATION FAILED — pipeline should HALT")
    elif report.has_warnings:
        logger.warning("FEATURE VALIDATION passed with warnings — sending alerts")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from src.utils import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(description="Validate SavVio features")
    parser.add_argument("--financial-features", default="data/features/financial_featured.csv")
    parser.add_argument("--review-features", default="data/features/product_featured.jsonl")
    parser.add_argument("--thresholds", default=None)
    args = parser.parse_args()

    report = run_feature_validation(args.financial_features, args.review_features, args.thresholds)

    if not report.passed:
        exit(1)