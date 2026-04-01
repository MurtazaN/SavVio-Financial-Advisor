"""Raw-stage validation checks.

Validates source datasets in `data/raw/` using Great Expectations and
project-specific rule checks mapped to INFO/WARNING/CRITICAL severities.
"""

import logging
import sys
import os
import pandas as pd
import great_expectations
import great_expectations as gx
from pathlib import Path

try:
    from great_expectations.dataset import PandasDataset
except ImportError:
    from great_expectations.dataset.pandas_dataset import PandasDataset

# Resolve local imports from the validation package.
current_file_path = Path(__file__).resolve()
validation_dir = current_file_path.parent.parent  # .../dags/src/validation/

def _find_pipeline_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    return current_file_path.parents[2]  # fallback: .../dags/


# Ensure running from data_pipeline root so relative data paths work
pipeline_root = _find_pipeline_root(current_file_path.parent)
if os.getcwd() != str(pipeline_root):
    os.chdir(pipeline_root)

from typing import Optional, List
from savviocore.validation.validation_config import (
    CheckResult, Severity, ValidationReport, load_thresholds,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_csv(path: str) -> PandasDataset:
    df = pd.read_csv(path)
    return gx.from_pandas(df)


def _load_jsonl(path: str) -> PandasDataset:
    # df = pd.read_json(path, lines=True)
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    if file_size_mb > 300:
        chunks = pd.read_json(path, lines=True, chunksize=100_000)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_json(path, lines=True)
    return gx.from_pandas(df)


def _check(ge_result: dict, name: str, severity: Severity,
           dataset: str, details: str = "") -> CheckResult:
    """Convert a GE expectation result dict into a CheckResult."""
    passed = ge_result["success"]
    metric = ge_result.get("result", {})
    return CheckResult(
        check_name=name,
        passed=passed,
        severity=severity,
        dataset=dataset,
        stage="raw",
        details=details or str(metric.get("partial_unexpected_list", ""))[:200],
        metric_value=metric.get("unexpected_percent")
                     or metric.get("observed_value"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Financial CSV validation
# ═══════════════════════════════════════════════════════════════════════════

FINANCIAL_REQUIRED_COLS = [
    "user_id", "monthly_income_usd", "monthly_expenses_usd", "savings_usd",
]
FINANCIAL_OPTIONAL_COLS = [
    "has_loan", "loan_amount_usd", "monthly_emi_usd", "loan_interest_rate_pct",
    "loan_term_months", "credit_score", "employment_status", "region",
]
FINANCIAL_NUMERIC_COLS = [
    "monthly_income_usd", "monthly_expenses_usd", "savings_usd",
    "loan_amount_usd", "monthly_emi_usd", "loan_interest_rate_pct",
    "loan_term_months", "credit_score",
]


def validate_financial_raw(path: str, thresholds: dict) -> list[CheckResult]:
    """Validate raw financial.csv."""
    results: list[CheckResult] = []
    ds = "financial"

    gdf = _load_csv(path)
    df = gdf  # also use as regular DataFrame via .head(), .shape, etc.

    # ── 1. Row count ──────────────────────────────────────────────────────
    n_rows = len(gdf)
    results.append(CheckResult(
        check_name="fin_row_count_critical",
        passed=n_rows >= thresholds["min_records_critical"],
        severity=Severity.CRITICAL, dataset=ds, stage="raw",
        details=f"Expected >= {thresholds['min_records_critical']} rows",
        metric_value=n_rows,
    ))
    results.append(CheckResult(
        check_name="fin_row_count_warning",
        passed=n_rows >= thresholds["min_records_warning"],
        severity=Severity.WARNING, dataset=ds, stage="raw",
        details=f"Expected >= {thresholds['min_records_warning']} rows",
        metric_value=n_rows,
    ))

    # ── 2. Required columns exist ─────────────────────────────────────────
    for col in FINANCIAL_REQUIRED_COLS:
        res = gdf.expect_column_to_exist(col)
        results.append(_check(res, f"fin_col_exists_{col}", Severity.CRITICAL, ds,
                              f"Required column '{col}' must exist"))

    # ── 3. Optional columns (INFO if missing) ─────────────────────────────
    for col in FINANCIAL_OPTIONAL_COLS:
        res = gdf.expect_column_to_exist(col)
        results.append(_check(res, f"fin_col_exists_{col}", Severity.INFO, ds,
                              f"Optional column '{col}' missing"))

    # ── 4. Null checks on required columns ────────────────────────────────
    for col in FINANCIAL_REQUIRED_COLS:
        if col not in gdf.columns:
            continue
        null_pct = gdf[col].isnull().mean()

        sev = Severity.INFO
        if null_pct > thresholds["null_pct_critical"]:
            sev = Severity.CRITICAL
        elif null_pct > thresholds["null_pct_warning"]:
            sev = Severity.WARNING

        res = gdf.expect_column_values_to_not_be_null(col, mostly=1 - thresholds["null_pct_warning"])
        results.append(_check(res, f"fin_nulls_{col}", sev, ds,
                              f"Null % = {null_pct:.2%}"))

    # ── 5. Data type checks (numeric columns) ─────────────────────────────
    for col in FINANCIAL_NUMERIC_COLS:
        if col not in gdf.columns:
            continue
        res = gdf.expect_column_values_to_be_in_type_list(
            col, ["int", "int64", "float", "float64", "int32", "float32"]
        )
        results.append(_check(res, f"fin_dtype_{col}", Severity.CRITICAL, ds,
                              f"Column '{col}' must be numeric"))

    # ── 6. Value range checks ─────────────────────────────────────────────
    # monthly_income_usd >= 0
    if "monthly_income_usd" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("monthly_income_usd", min_value=0)
        results.append(_check(res, "fin_income_non_negative", Severity.WARNING, ds,
                              "monthly_income_usd should be >= 0"))

    # monthly_expenses_usd >= 0
    if "monthly_expenses_usd" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("monthly_expenses_usd", min_value=0)
        results.append(_check(res, "fin_expenses_non_negative", Severity.WARNING, ds,
                              "monthly_expenses_usd should be >= 0"))

    # savings_usd (can be negative but flag extreme)
    if "savings_usd" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("savings_usd", min_value=-1_000_000, max_value=10_000_000)
        results.append(_check(res, "fin_savings_range", Severity.WARNING, ds,
                              "savings_usd outside plausible range"))

    # credit_score 300–850
    if "credit_score" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("credit_score", min_value=300, max_value=850)
        results.append(_check(res, "fin_credit_score_range", Severity.WARNING, ds,
                              "credit_score should be 300–850"))

    # loan_interest_rate_pct 0–100
    if "loan_interest_rate_pct" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("loan_interest_rate_pct", min_value=0, max_value=100)
        results.append(_check(res, "fin_interest_rate_range", Severity.INFO, ds,
                              "loan_interest_rate_pct should be 0–100"))

    # ── 7. Duplicate check ────────────────────────────────────────────────
    if "user_id" in gdf.columns:
        dup_pct = 1 - gdf["user_id"].nunique() / max(len(gdf), 1)
        sev = Severity.INFO
        if dup_pct > thresholds["dup_pct_critical"]:
            sev = Severity.CRITICAL
        elif dup_pct > thresholds["dup_pct_warning"]:
            sev = Severity.WARNING
        results.append(CheckResult(
            check_name="fin_duplicate_user_ids",
            passed=dup_pct <= thresholds["dup_pct_warning"],
            severity=sev, dataset=ds, stage="raw",
            details=f"Duplicate user_id rate = {dup_pct:.2%}",
            metric_value=round(dup_pct, 4),
        ))

    # ── 8. employment_status valid set ────────────────────────────────────
    if "employment_status" in gdf.columns:
        valid_statuses = ["employed", "self-employed", "unemployed", "retired",
                          "student", "part-time", "freelance", "Employed", "Self-employed", "Unemployed", "Retired", "Student"] # Add capitalized versions seen in head
        res = gdf.expect_column_values_to_be_in_set(
            "employment_status", valid_statuses, mostly=0.95
        )
        results.append(_check(res, "fin_employment_status_valid", Severity.INFO, ds,
                              "Unexpected employment_status values"))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Products JSONL validation
# ═══════════════════════════════════════════════════════════════════════════

PRODUCT_REQUIRED_COLS = ["parent_asin", "title", "price"]
PRODUCT_OPTIONAL_COLS = [
    "main_category", "store", "average_rating", "rating_number",
    "description", "features", "details",
]


def validate_products_raw(path: str, thresholds: dict) -> list[CheckResult]:
    """Validate raw products.jsonl."""
    results: list[CheckResult] = []
    ds = "products"

    gdf = _load_jsonl(path)

    # ── 1. Row count ──────────────────────────────────────────────────────
    n_rows = len(gdf)
    results.append(CheckResult(
        check_name="prod_row_count_critical",
        passed=n_rows >= thresholds["min_records_critical"],
        severity=Severity.CRITICAL, dataset=ds, stage="raw",
        details=f"Expected >= {thresholds['min_records_critical']} rows",
        metric_value=n_rows,
    ))

    # ── 2. Required columns exist ─────────────────────────────────────────
    for col in PRODUCT_REQUIRED_COLS:
        res = gdf.expect_column_to_exist(col)
        results.append(_check(res, f"prod_col_exists_{col}", Severity.CRITICAL, ds,
                              f"Required column '{col}' must exist"))

    for col in PRODUCT_OPTIONAL_COLS:
        res = gdf.expect_column_to_exist(col)
        results.append(_check(res, f"prod_col_exists_{col}", Severity.INFO, ds,
                              f"Optional column '{col}' missing"))

    # ── 3. Null checks ────────────────────────────────────────────────────
    for col in PRODUCT_REQUIRED_COLS:
        if col not in gdf.columns:
            continue
        null_pct = gdf[col].isnull().mean()
        sev = Severity.INFO
        if null_pct > thresholds["null_pct_critical"]:
            sev = Severity.CRITICAL
        elif null_pct > thresholds["null_pct_warning"]:
            sev = Severity.WARNING
        
        # DOWNGRADE severity for 'price' because preprocessing imputes missing values
        if col == "price" and sev == Severity.CRITICAL:
            sev = Severity.WARNING

        res = gdf.expect_column_values_to_not_be_null(col, mostly=1 - thresholds["null_pct_warning"])
        results.append(_check(res, f"prod_nulls_{col}", sev, ds,
                              f"Null % = {null_pct:.2%}"))

    # ── 4. Price validation ───────────────────────────────────────────────
    if "price" in gdf.columns:
        # Price should be numeric and > 0
        # Amazon JSONL sometimes has price as string or None
        res = gdf.expect_column_values_to_be_between("price", min_value=0.01, max_value=100_000, mostly=0.90)
        results.append(_check(res, "prod_price_positive", Severity.WARNING, ds,
                              "price should be > 0 and < $100,000"))

    # ── 5. parent_asin non-empty ──────────────────────────────────────────
    if "parent_asin" in gdf.columns:
        res = gdf.expect_column_value_lengths_to_be_between("parent_asin", min_value=1)
        results.append(_check(res, "prod_asin_non_empty", Severity.CRITICAL, ds,
                              "parent_asin must not be empty string"))

    # ── 6. title non-empty ────────────────────────────────────────────────
    if "title" in gdf.columns:
        res = gdf.expect_column_value_lengths_to_be_between("title", min_value=1, mostly=0.95)
        results.append(_check(res, "prod_title_non_empty", Severity.WARNING, ds,
                              "title should not be empty"))

    # ── 7. average_rating range ───────────────────────────────────────────
    if "average_rating" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("average_rating", min_value=1.0, max_value=5.0, mostly=0.95)
        results.append(_check(res, "prod_avg_rating_range", Severity.INFO, ds,
                              "average_rating should be 1.0–5.0"))

    # ── 8. rating_number >= 0 ─────────────────────────────────────────────
    if "rating_number" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("rating_number", min_value=0, mostly=0.95)
        results.append(_check(res, "prod_rating_number_non_negative", Severity.INFO, ds,
                              "rating_number should be >= 0"))

    # ── 9. Duplicate parent_asin check ────────────────────────────────────
    if "parent_asin" in gdf.columns:
        dup_pct = 1 - gdf["parent_asin"].nunique() / max(len(gdf), 1)
        sev = Severity.WARNING if dup_pct > thresholds["dup_pct_warning"] else Severity.INFO
        results.append(CheckResult(
            check_name="prod_duplicate_asins",
            passed=dup_pct <= thresholds["dup_pct_warning"],
            severity=sev, dataset=ds, stage="raw",
            details=f"Duplicate parent_asin rate = {dup_pct:.2%}",
            metric_value=round(dup_pct, 4),
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Reviews JSONL validation
# ═══════════════════════════════════════════════════════════════════════════

REVIEW_REQUIRED_COLS = ["rating", "asin", "user_id"]
REVIEW_OPTIONAL_COLS = [
    "parent_asin", "title", "text", "verified_purchase",
    "helpful_vote", "timestamp", "images",
]


def validate_reviews_raw(path: str, thresholds: dict) -> list[CheckResult]:
    """Validate raw reviews.jsonl."""
    results: list[CheckResult] = []
    ds = "reviews"

    gdf = _load_jsonl(path)

    # ── 1. Row count ──────────────────────────────────────────────────────
    n_rows = len(gdf)
    results.append(CheckResult(
        check_name="rev_row_count_critical",
        passed=n_rows >= thresholds["min_records_critical"],
        severity=Severity.CRITICAL, dataset=ds, stage="raw",
        details=f"Expected >= {thresholds['min_records_critical']} rows",
        metric_value=n_rows,
    ))

    # ── 2. Required columns exist ─────────────────────────────────────────
    for col in REVIEW_REQUIRED_COLS:
        res = gdf.expect_column_to_exist(col)
        results.append(_check(res, f"rev_col_exists_{col}", Severity.CRITICAL, ds,
                              f"Required column '{col}' must exist"))

    for col in REVIEW_OPTIONAL_COLS:
        res = gdf.expect_column_to_exist(col)
        results.append(_check(res, f"rev_col_exists_{col}", Severity.INFO, ds,
                              f"Optional column '{col}' missing"))

    # ── 3. Null checks on required ────────────────────────────────────────
    for col in REVIEW_REQUIRED_COLS:
        if col not in gdf.columns:
            continue
        null_pct = gdf[col].isnull().mean()
        sev = Severity.CRITICAL if null_pct > thresholds["null_pct_critical"] else (
              Severity.WARNING if null_pct > thresholds["null_pct_warning"] else Severity.INFO)
        res = gdf.expect_column_values_to_not_be_null(col, mostly=1 - thresholds["null_pct_warning"])
        results.append(_check(res, f"rev_nulls_{col}", sev, ds, f"Null % = {null_pct:.2%}"))

    # ── 4. Rating range 1–5 ──────────────────────────────────────────────
    if "rating" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("rating", min_value=1.0, max_value=5.0)
        results.append(_check(res, "rev_rating_range", Severity.CRITICAL, ds,
                              "rating must be 1.0–5.0"))

    # ── 5. asin non-empty ─────────────────────────────────────────────────
    if "asin" in gdf.columns:
        res = gdf.expect_column_value_lengths_to_be_between("asin", min_value=1)
        results.append(_check(res, "rev_asin_non_empty", Severity.CRITICAL, ds,
                              "asin must not be empty"))

    # ── 6. user_id non-empty ──────────────────────────────────────────────
    if "user_id" in gdf.columns:
        res = gdf.expect_column_value_lengths_to_be_between("user_id", min_value=1, mostly=0.99)
        results.append(_check(res, "rev_user_id_non_empty", Severity.WARNING, ds,
                              "user_id should not be empty"))

    # ── 7. verified_purchase is boolean ───────────────────────────────────
    if "verified_purchase" in gdf.columns:
        res = gdf.expect_column_values_to_be_in_set("verified_purchase", [True, False, 0, 1], mostly=0.99)
        results.append(_check(res, "rev_verified_purchase_bool", Severity.INFO, ds,
                              "verified_purchase should be boolean"))

    # ── 8. helpful_vote >= 0 ──────────────────────────────────────────────
    if "helpful_vote" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("helpful_vote", min_value=0, mostly=0.99)
        results.append(_check(res, "rev_helpful_vote_non_negative", Severity.INFO, ds,
                              "helpful_vote should be >= 0"))

    # ── 9. Referential integrity: asin should match products ──────────────
    #    NOTE: Cross-dataset check — requires products to be loaded first.
    #    Skipped here; run via validate_cross_references() below.

    # ── 10. Duplicate check (user_id + asin) ──────────────────────────────
    if "user_id" in gdf.columns and "asin" in gdf.columns:
        total = len(gdf)
        deduped = gdf.drop_duplicates(subset=["user_id", "asin"]).shape[0]
        dup_pct = 1 - deduped / max(total, 1)
        sev = Severity.WARNING if dup_pct > thresholds["dup_pct_warning"] else Severity.INFO
        results.append(CheckResult(
            check_name="rev_duplicate_user_asin",
            passed=dup_pct <= thresholds["dup_pct_warning"],
            severity=sev, dataset=ds, stage="raw",
            details=f"Duplicate (user_id, asin) rate = {dup_pct:.2%}",
            metric_value=round(dup_pct, 4),
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Cross-dataset referential check
# ═══════════════════════════════════════════════════════════════════════════

def validate_cross_references(products_path: str, reviews_path: str) -> list[CheckResult]:
    """Check that review ASINs reference existing product ASINs."""
    results: list[CheckResult] = []

    # products_df = pd.read_json(products_path, lines=True)
    # reviews_df = pd.read_json(reviews_path, lines=True)
    prod_size_mb = os.path.getsize(products_path) / (1024 * 1024)
    if prod_size_mb > 300:
        products_df = pd.concat(pd.read_json(products_path, lines=True, chunksize=100_000), ignore_index=True)
    else:
        products_df = pd.read_json(products_path, lines=True)
    rev_size_mb = os.path.getsize(reviews_path) / (1024 * 1024)
    if rev_size_mb > 300:
        reviews_df = pd.concat(pd.read_json(reviews_path, lines=True, chunksize=100_000), ignore_index=True)
    else:
        reviews_df = pd.read_json(reviews_path, lines=True)

    # Determine which ASIN column reviews use to reference products
    review_asin_col = "parent_asin" if "parent_asin" in reviews_df.columns else "asin"
    product_asin_col = "parent_asin" if "parent_asin" in products_df.columns else "asin"

    if product_asin_col in products_df.columns and review_asin_col in reviews_df.columns:
        product_asins = set(products_df[product_asin_col].dropna().unique())
        review_asins = set(reviews_df[review_asin_col].dropna().unique())
        orphan_asins = review_asins - product_asins
        orphan_pct = len(orphan_asins) / max(len(review_asins), 1)

        sev = Severity.CRITICAL if orphan_pct > 0.20 else (
              Severity.WARNING if orphan_pct > 0.05 else Severity.INFO)
        results.append(CheckResult(
            check_name="cross_ref_review_asin_exists_in_products",
            passed=orphan_pct <= 0.05,
            severity=sev,
            dataset="reviews→products",
            stage="raw",
            details=f"{len(orphan_asins)} orphan ASINs ({orphan_pct:.1%}) in reviews not found in products",
            metric_value=round(orphan_pct, 4),
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_raw_validation(
    financial_path: str = "data/raw/financial_data.csv",
    products_path: str = "data/raw/product_data.jsonl",
    reviews_path: str = "data/raw/review_data.jsonl",
    threshold_config: Optional[str] = "config/validation_thresholds.json",
) -> ValidationReport:
    """
    Run all raw data validations.
    Returns a ValidationReport with pipeline action (CONTINUE / ALERT / HALT).
    """
    thresholds = load_thresholds(threshold_config)
    report = ValidationReport(stage="raw")

    logger.info("═" * 50)
    logger.info("  RAW DATA VALIDATION")
    logger.info("═" * 50)

    # Financial
    logger.info("── Validating financial.csv ──")
    for r in validate_financial_raw(financial_path, thresholds):
        report.add(r)

    # Products
    logger.info("── Validating products.jsonl ──")
    for r in validate_products_raw(products_path, thresholds):
        report.add(r)

    # Reviews
    logger.info("── Validating reviews.jsonl ──")
    for r in validate_reviews_raw(reviews_path, thresholds):
        report.add(r)

    # Cross-references
    logger.info("── Validating cross-references ──")
    for r in validate_cross_references(products_path, reviews_path):
        report.add(r)

    report.print_summary()
    report.save()

    if not report.passed:
        logger.critical("RAW VALIDATION FAILED — pipeline should HALT")
    elif report.has_warnings:
        logger.warning("RAW VALIDATION passed with warnings — sending alerts")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from src.utils import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(description="Validate raw SavVio data")
    parser.add_argument("--financial", default="data/raw/financial_data.csv")
    parser.add_argument("--products", default="data/raw/product_data.jsonl")
    parser.add_argument("--reviews", default="data/raw/review_data.jsonl")
    parser.add_argument("--thresholds", default=None)
    args = parser.parse_args()

    report = run_raw_validation(args.financial, args.products, args.reviews, args.thresholds)

    if not report.passed:
        exit(1)
