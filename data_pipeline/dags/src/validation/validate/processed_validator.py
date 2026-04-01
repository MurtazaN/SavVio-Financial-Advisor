"""Processed-stage validation checks.

Validates datasets in `data/processed/` to confirm preprocessing outputs are
schema-complete, internally consistent, and safe for downstream use.
"""

import logging
import sys
import os
import pandas as pd
import great_expectations as gx
from pathlib import Path

from src.utils import setup_logging
from typing import Optional, List
from savviocore.validation.validation_config import (
    CheckResult, Severity, ValidationReport, load_thresholds,
)

try:
    from great_expectations.dataset import PandasDataset
except ImportError:
    from great_expectations.dataset.pandas_dataset import PandasDataset

# Resolve local imports from the validation package.
current_file_path = Path(__file__).resolve()
validation_dir = current_file_path.parent.parent

def _find_pipeline_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    return current_file_path.parents[2]  # fallback: .../dags/


# Ensure running from data_pipeline root so relative data paths work
pipeline_root = _find_pipeline_root(current_file_path.parent)
if os.getcwd() != str(pipeline_root):
    os.chdir(pipeline_root)



logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load(path: str) -> PandasDataset:
    """Load CSV or JSONL depending on extension."""
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
        dataset=dataset, stage="processed",
        details=details or str(metric.get("partial_unexpected_list", ""))[:200],
        metric_value=metric.get("unexpected_percent") or metric.get("observed_value"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Financial processed validation
# ═══════════════════════════════════════════════════════════════════════════

FINANCIAL_PROCESSED_COLS = [
    "user_id", "monthly_income", "monthly_expenses", "savings_balance",
]
FINANCIAL_OPTIONAL_PROCESSED = [
    "has_loan", "loan_amount", "monthly_emi",
    "credit_score", "employment_status", "region",
]


def validate_financial_processed(path: str, raw_path: str,
                                  thresholds: dict) -> list[CheckResult]:
    """Validate processed financial data."""
    results: list[CheckResult] = []
    ds = "financial"

    gdf = _load(path)

    # ── 1. Required processed columns exist ───────────────────────────────
    for col in FINANCIAL_PROCESSED_COLS:
        res = gdf.expect_column_to_exist(col)
        results.append(_check(res, f"fin_proc_col_{col}", Severity.CRITICAL, ds,
                              f"Processed column '{col}' must exist"))

    # ── 2. No new nulls in required fields ────────────────────────────────
    for col in ["monthly_income", "monthly_expenses", "savings_balance"]:
        if col not in gdf.columns:
            continue
        # Required processed financial columns should have no nulls
        res = gdf.expect_column_values_to_not_be_null(col)
        results.append(_check(res, f"fin_proc_no_nulls_{col}", Severity.CRITICAL, ds,
                              f"'{col}' should have no nulls after processing"))

    # ── 3. USD suffix columns are numeric and >= 0 ────────────────────────
    for col in ["monthly_income", "monthly_expenses", "savings_balance"]:
        if col not in gdf.columns:
            continue
        min_value = -1_000_000 if col == "savings_balance" else 0
        res = gdf.expect_column_values_to_be_between(col, min_value=min_value)
        results.append(_check(res, f"fin_proc_{col}_non_negative", Severity.WARNING, ds,
                              f"'{col}' should be within expected range after processing"))

    # ── 4. Values rounded to 2 decimals ───────────────────────────────────
    for col in ["monthly_income", "monthly_expenses", "savings_balance"]:
        if col not in gdf.columns:
            continue
        # Check values are rounded to 2 decimals (tolerance for float precision)
        vals = gdf[col].dropna()
        not_rounded = (vals.sub(vals.round(2)).abs() > 0.001).sum()
        pct_bad = not_rounded / max(len(gdf), 1)
        results.append(CheckResult(
            check_name=f"fin_proc_{col}_rounded",
            passed=pct_bad < 0.01,
            severity=Severity.INFO, dataset=ds, stage="processed",
            details=f"{pct_bad:.2%} of values not rounded to 2 decimals",
            metric_value=round(pct_bad, 4),
        ))

    # ── 5. has_loan flag is boolean-like (if present) ─────────────────────
    if "has_loan" in gdf.columns:
        res = gdf.expect_column_values_to_be_in_set("has_loan", [True, False, 0, 1])
        results.append(_check(res, "fin_proc_has_loan_bool", Severity.WARNING, ds,
                              "has_loan should be boolean"))

    # ── 6. Record count comparison (raw vs processed) ─────────────────────
    try:
        raw_df = pd.read_csv(raw_path)
        raw_count = len(raw_df)
        proc_count = len(gdf)
        loss_pct = 1 - proc_count / max(raw_count, 1)

        sev = Severity.CRITICAL if loss_pct > 0.20 else (
              Severity.WARNING if loss_pct > 0.05 else Severity.INFO)
        results.append(CheckResult(
            check_name="fin_proc_record_loss",
            passed=loss_pct <= 0.20,
            severity=sev, dataset=ds, stage="processed",
            details=f"Raw: {raw_count} → Processed: {proc_count} (loss: {loss_pct:.1%})",
            metric_value=round(loss_pct, 4),
        ))
    except Exception as e:
        logger.warning("Could not compare raw vs processed counts: %s", e)

    # ── 7. No duplicates after processing ─────────────────────────────────
    if "user_id" in gdf.columns:
        dup_count = gdf.duplicated(subset=["user_id"]).sum()
        results.append(CheckResult(
            check_name="fin_proc_no_duplicates",
            passed=dup_count == 0,
            severity=Severity.WARNING, dataset=ds, stage="processed",
            details=f"{dup_count} duplicate user_ids after processing",
            metric_value=dup_count,
        ))

    # ── 8. monthly_emi vs monthly_expenses (sanity, if available) ─────────
    # EMI can exceed expenses in valid cases (e.g. expenses = non-housing; EMI = mortgage)
    if "monthly_emi" in gdf.columns and "monthly_expenses" in gdf.columns:
        violations = (gdf["monthly_emi"] > gdf["monthly_expenses"]).sum()
        pct = violations / max(len(gdf), 1)
        results.append(CheckResult(
            check_name="fin_proc_emi_lte_total",
            passed=pct < 0.50,  # Flag only if >50% violate (likely data definition issue)
            severity=Severity.INFO, dataset=ds, stage="processed",
            details=f"{violations} rows where monthly_emi > monthly_expenses ({pct:.1%})",
            metric_value=round(pct, 4),
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Products processed validation
# ═══════════════════════════════════════════════════════════════════════════

PRODUCT_PROCESSED_COLS = [
    "product_id", "product_name", "price",
]
PRODUCT_OPTIONAL_PROCESSED = [
    "average_rating", "rating_number", "description", "features",
    "details", "category", "appliance_type", "description_length",
]


def validate_products_processed(path: str, raw_path: str,
                                 thresholds: dict) -> list[CheckResult]:
    """Validate processed products data."""
    results: list[CheckResult] = []
    ds = "products"

    gdf = _load(path)

    # ── 1. Required columns ───────────────────────────────────────────────
    for col in PRODUCT_PROCESSED_COLS:
        res = gdf.expect_column_to_exist(col)
        results.append(_check(res, f"prod_proc_col_{col}", Severity.CRITICAL, ds,
                              f"Processed column '{col}' must exist"))

    # ── 2. product_name not empty (cleaned title) ─────────────────────────
    if "product_name" in gdf.columns:
        res = gdf.expect_column_value_lengths_to_be_between("product_name", min_value=1, mostly=0.99)
        results.append(_check(res, "prod_proc_name_non_empty", Severity.WARNING, ds,
                              "product_name should not be empty after cleaning"))

    # ── 3. Price valid range and no nulls (critical) ──────────────────────
    if "price" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("price", min_value=0.01, max_value=100_000)
        results.append(_check(res, "prod_proc_price_range", Severity.WARNING, ds,
                              "price should be $0.01–$100,000"))

        # No nulls in price after processing
        res = gdf.expect_column_values_to_not_be_null("price")
        results.append(_check(res, "prod_proc_price_no_nulls", Severity.CRITICAL, ds,
                              "price must not be null after processing"))

    # ── 4. main_category should be dropped (constant column) ──────────────
    if "main_category" in gdf.columns:
        n_unique = gdf["main_category"].nunique()
        results.append(CheckResult(
            check_name="prod_proc_main_cat_dropped",
            passed=False,  # It should have been dropped
            severity=Severity.INFO, dataset=ds, stage="processed",
            details=f"main_category still present ({n_unique} unique values) — expected to be dropped",
            metric_value=n_unique,
        ))

    # ── 5. description_length is non-negative integer ─────────────────────
    if "description_length" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("description_length", min_value=0)
        results.append(_check(res, "prod_proc_desc_len_valid", Severity.INFO, ds,
                              "description_length should be >= 0"))

    # ── 6. appliance_type populated ───────────────────────────────────────
    if "appliance_type" in gdf.columns:
        null_pct = gdf["appliance_type"].isnull().mean()
        results.append(CheckResult(
            check_name="prod_proc_appliance_type_populated",
            passed=null_pct < 0.30,
            severity=Severity.INFO, dataset=ds, stage="processed",
            details=f"appliance_type null rate = {null_pct:.1%}",
            metric_value=round(null_pct, 4),
        ))

    # ── 7. No duplicate products ──────────────────────────────────────────
    if "product_id" in gdf.columns:
        dup_count = gdf.duplicated(subset=["product_id"]).sum()
        results.append(CheckResult(
            check_name="prod_proc_no_duplicates",
            passed=dup_count == 0,
            severity=Severity.WARNING, dataset=ds, stage="processed",
            details=f"{dup_count} duplicate product_ids after deduplication",
            metric_value=dup_count,
        ))

    # ── 8. Record count comparison ────────────────────────────────────────
    try:
        # raw_df = pd.read_json(raw_path, lines=True)
        raw_size_mb = os.path.getsize(raw_path) / (1024 * 1024)
        if raw_size_mb > 300:
            raw_df = pd.concat(pd.read_json(raw_path, lines=True, chunksize=100_000), ignore_index=True)
        else:
            raw_df = pd.read_json(raw_path, lines=True)
        raw_count = len(raw_df)
        proc_count = len(gdf)
        loss_pct = 1 - proc_count / max(raw_count, 1)
        sev = Severity.CRITICAL if loss_pct > 0.30 else (
              Severity.WARNING if loss_pct > 0.10 else Severity.INFO)
        results.append(CheckResult(
            check_name="prod_proc_record_loss",
            passed=loss_pct <= 0.30,
            severity=sev, dataset=ds, stage="processed",
            details=f"Raw: {raw_count} → Processed: {proc_count} (loss: {loss_pct:.1%})",
            metric_value=round(loss_pct, 4),
        ))
    except Exception as e:
        logger.warning("Could not compare raw vs processed product counts: %s", e)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Reviews processed validation
# ═══════════════════════════════════════════════════════════════════════════

REVIEW_PROCESSED_COLS = [
    "user_id", "asin", "rating",
]
REVIEW_OPTIONAL_PROCESSED = [
    "product_id", "review_title", "review_text", "sentiment",
    "review_length", "verified_purchase", "helpful_vote",
]


def validate_reviews_processed(path: str, raw_path: str,
                                thresholds: dict) -> list[CheckResult]:
    """Validate processed reviews data."""
    results: list[CheckResult] = []
    ds = "reviews"

    gdf = _load(path)

    # ── 1. Required columns ───────────────────────────────────────────────
    for col in REVIEW_PROCESSED_COLS:
        res = gdf.expect_column_to_exist(col)
        results.append(_check(res, f"rev_proc_col_{col}", Severity.CRITICAL, ds,
                              f"Processed column '{col}' must exist"))

    # ── 2. Rating still 1–5 ──────────────────────────────────────────────
    if "rating" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("rating", min_value=1.0, max_value=5.0)
        results.append(_check(res, "rev_proc_rating_range", Severity.CRITICAL, ds,
                              "rating must remain 1.0–5.0 after processing"))

    # ── 3. Sentiment mapping correct ──────────────────────────────────────
    if "sentiment" in gdf.columns:
        valid_sentiments = ["positive", "neutral", "negative"]
        res = gdf.expect_column_values_to_be_in_set("sentiment", valid_sentiments)
        results.append(_check(res, "rev_proc_sentiment_valid", Severity.WARNING, ds,
                              "sentiment must be positive/neutral/negative"))

        # Cross-check: sentiment matches rating
        if "rating" in gdf.columns:
            df_check = pd.DataFrame({"rating": gdf["rating"], "sentiment": gdf["sentiment"]})
            mismatches = (
                ((df_check["rating"] >= 4) & (df_check["sentiment"] != "positive")) |
                ((df_check["rating"] == 3) & (df_check["sentiment"] != "neutral")) |
                ((df_check["rating"] <= 2) & (df_check["sentiment"] != "negative"))
            ).sum()
            mismatch_pct = mismatches / max(len(gdf), 1)
            results.append(CheckResult(
                check_name="rev_proc_sentiment_rating_consistency",
                passed=mismatch_pct < 0.01,
                severity=Severity.WARNING, dataset=ds, stage="processed",
                details=f"{mismatches} sentiment-rating mismatches ({mismatch_pct:.1%})",
                metric_value=round(mismatch_pct, 4),
            ))

    # ── 4. review_length non-negative ─────────────────────────────────────
    if "review_length" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("review_length", min_value=0)
        results.append(_check(res, "rev_proc_review_length_valid", Severity.INFO, ds,
                              "review_length should be >= 0"))

    # ── 5. helpful_vote no negatives after fillna ─────────────────────────
    if "helpful_vote" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("helpful_vote", min_value=0)
        results.append(_check(res, "rev_proc_helpful_vote_valid", Severity.INFO, ds,
                              "helpful_vote should be >= 0 after fillna"))

        # No nulls after processing
        null_pct = gdf["helpful_vote"].isnull().mean()
        results.append(CheckResult(
            check_name="rev_proc_helpful_vote_no_nulls",
            passed=null_pct == 0,
            severity=Severity.INFO, dataset=ds, stage="processed",
            details=f"helpful_vote null rate = {null_pct:.2%} (should be 0 after fillna)",
            metric_value=round(null_pct, 4),
        ))

    # ── 6. No duplicate reviews ───────────────────────────────────────────
    dedup_cols = ["user_id", "asin"]
    if all(c in gdf.columns for c in dedup_cols):
        dup_count = gdf.duplicated(subset=dedup_cols).sum()
        results.append(CheckResult(
            check_name="rev_proc_no_duplicates",
            passed=dup_count == 0,
            severity=Severity.WARNING, dataset=ds, stage="processed",
            details=f"{dup_count} duplicate (user_id, asin) after deduplication",
            metric_value=dup_count,
        ))

    # ── 7. Record count comparison ────────────────────────────────────────
    try:
        # raw_df = pd.read_json(raw_path, lines=True)
        raw_size_mb = os.path.getsize(raw_path) / (1024 * 1024)
        if raw_size_mb > 300:
            raw_df = pd.concat(pd.read_json(raw_path, lines=True, chunksize=100_000), ignore_index=True)
        else:
            raw_df = pd.read_json(raw_path, lines=True)
        raw_count = len(raw_df)
        proc_count = len(gdf)
        loss_pct = 1 - proc_count / max(raw_count, 1)
        sev = Severity.CRITICAL if loss_pct > 0.30 else (
              Severity.WARNING if loss_pct > 0.10 else Severity.INFO)
        results.append(CheckResult(
            check_name="rev_proc_record_loss",
            passed=loss_pct <= 0.30,
            severity=sev, dataset=ds, stage="processed",
            details=f"Raw: {raw_count} → Processed: {proc_count} (loss: {loss_pct:.1%})",
            metric_value=round(loss_pct, 4),
        ))
    except Exception as e:
        logger.warning("Could not compare raw vs processed review counts: %s", e)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_processed_validation(
    financial_path: str = "data/processed/financial_preprocessed.csv",
    products_path: str = "data/processed/product_preprocessed.jsonl",
    reviews_path: str = "data/processed/review_preprocessed.jsonl",
    raw_financial: str = "data/raw/financial_data.csv",
    raw_products: str = "data/raw/product_data.jsonl",
    raw_reviews: str = "data/raw/review_data.jsonl",
    threshold_config: Optional[str] = "config/validation_thresholds.json",
) -> ValidationReport:
    """Run all processed data validations."""
    thresholds = load_thresholds(threshold_config)
    report = ValidationReport(stage="processed")

    logger.info("═" * 50)
    logger.info("  PROCESSED DATA VALIDATION")
    logger.info("═" * 50)

    logger.info("── Validating financial_processed ──")
    for r in validate_financial_processed(financial_path, raw_financial, thresholds):
        report.add(r)

    logger.info("── Validating products_processed ──")
    for r in validate_products_processed(products_path, raw_products, thresholds):
        report.add(r)

    logger.info("── Validating reviews_processed ──")
    for r in validate_reviews_processed(reviews_path, raw_reviews, thresholds):
        report.add(r)

    report.print_summary()
    report.save()

    if not report.passed:
        logger.critical("PROCESSED VALIDATION FAILED — pipeline should HALT")
    elif report.has_warnings:
        logger.warning("PROCESSED VALIDATION passed with warnings — sending alerts")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    setup_logging()

    parser = argparse.ArgumentParser(description="Validate processed SavVio data")
    parser.add_argument("--financial", default="data/processed/financial_preprocessed.csv")
    parser.add_argument("--products", default="data/processed/product_preprocessed.jsonl")
    parser.add_argument("--reviews", default="data/processed/review_preprocessed.jsonl")
    parser.add_argument("--raw-financial", default="data/raw/financial_data.csv")
    parser.add_argument("--raw-products", default="data/raw/product_data.jsonl")
    parser.add_argument("--raw-reviews", default="data/raw/review_data.jsonl")
    parser.add_argument("--thresholds", default=None)
    args = parser.parse_args()

    report = run_processed_validation(
        args.financial, args.products, args.reviews,
        args.raw_financial, args.raw_products, args.raw_reviews,
        args.thresholds,
    )

    if not report.passed:
        exit(1)
