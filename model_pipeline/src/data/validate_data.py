"""
Data Validation for the Model Pipeline.

Validates that data loaded from either CSV or PostgreSQL meets the schema
and quality expectations required by the training pipeline. Run this
after loading data and before passing it to feature engineering / training.

Checks performed:
    1. Required columns exist
    2. No unexpected nulls in critical columns
    3. Data types are correct (numeric vs categorical)
    4. Value ranges are reasonable (no negative incomes, credit scores in range, etc.)
    5. Minimum row count to avoid training on empty/tiny datasets
    6. Target variable distribution is not completely degenerate

Usage:
    from data.validate_data import validate_financial_data, validate_products

    df = load_data()  # or load_financial_profiles()
    validate_financial_data(df)  # raises ValidationError on failure
"""

import logging
import pandas as pd
import great_expectations as gx
from typing import Optional

from savviocore.validation.feature_validator import validate_financial_features, validate_review_features
from savviocore.validation.validation_config import load_thresholds, ValidationReport

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when loaded data fails validation checks."""
    pass


# ---------------------------------------------------------------------------
# Validation functions (Delegated to savviocore)
# ---------------------------------------------------------------------------
def validate_financial_data(
    df: pd.DataFrame,
    raise_on_error: bool = True,
) -> dict:
    """
    Validate a financial profiles DataFrame.

    Args:
        df: Financial data loaded from CSV or PostgreSQL.
        raise_on_error: If True, raises DataValidationError on any failure.

    Returns:
        dict with 'valid', 'errors', 'warnings', 'summary'.
    """
    try:
        # Utilizing the core feature validator
        report = validate_financial_features(df)
        is_valid = report.passed
        errors = [f"CRITICAL: {r.check_name} - {r.details}" for r in report.results if not r.passed and r.severity.name == "CRITICAL"]
        warnings = [f"WARNING: {r.check_name} - {r.details}" for r in report.results if not r.passed and r.severity.name == "WARNING"]
        
        summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "null_pct": df.isnull().mean().to_dict(),
        }

        if is_valid:
            logger.info("Financial data validation PASSED — %d rows", len(df))
            print(f"[VALIDATION] Financial data PASSED — {len(df)} rows, {len(df.columns)} cols")
        else:
            logger.error("Financial data validation FAILED:\n%s", "\n".join(errors))
            print(f"[VALIDATION] Financial data FAILED — {len(errors)} error(s)")
            for e in errors:
                print(f"  ERROR: {e}")

        if warnings:
            for w in warnings:
                logger.warning(w)
                print(f"  WARNING: {w}")

        if raise_on_error and not is_valid:
            raise DataValidationError(f"Financial data validation failed with {len(errors)} error(s):\n" + "\n".join(errors))

        return {"valid": is_valid, "errors": errors, "warnings": warnings, "summary": summary}
    
    except Exception as e:
        if raise_on_error:
            raise DataValidationError(f"Financial data validation exception: {str(e)}")
        return {"valid": False, "errors": [str(e)], "warnings": [], "summary": {}}


def validate_products(
    df: pd.DataFrame,
    raise_on_error: bool = True,
) -> dict:
    """
    Validate a products DataFrame.

    Args:
        df: Product data loaded from CSV or PostgreSQL.
        raise_on_error: If True, raises DataValidationError on any failure.

    Returns:
        dict with 'valid', 'errors', 'warnings', 'summary'.
    """
    # Using the review/product combination core validator logic for basic checks
    try:
        report = validate_review_features(df)
        is_valid = report.passed
        errors = [f"CRITICAL: {r.check_name} - {r.details}" for r in report.results if not r.passed and r.severity.name == "CRITICAL"]
        warnings = [f"WARNING: {r.check_name} - {r.details}" for r in report.results if not r.passed and r.severity.name == "WARNING"]

        summary = {
            "rows": len(df),
            "columns": len(df.columns),
        }

        if is_valid:
            logger.info("Products validation PASSED — %d rows", len(df))
            print(f"[VALIDATION] Products data PASSED — {len(df)} rows")
        else:
            logger.error("Products validation FAILED:\n%s", "\n".join(errors))
            print(f"[VALIDATION] Products data FAILED — {len(errors)} error(s)")
            for e in errors:
                print(f"  ERROR: {e}")

        if warnings:
            for w in warnings:
                print(f"  WARNING: {w}")

        if raise_on_error and not is_valid:
            raise DataValidationError(f"Products validation failed with {len(errors)} error(s):\n" + "\n".join(errors))

        return {"valid": is_valid, "errors": errors, "warnings": warnings, "summary": summary}

    except Exception as e:
        if raise_on_error:
            raise DataValidationError(f"Product data validation exception: {str(e)}")
        return {"valid": False, "errors": [str(e)], "warnings": [], "summary": {}}


def validate_target_distribution(
    y: pd.Series,
    min_minority_pct: float = 5.0,
    raise_on_error: bool = True,
) -> dict:
    """
    Check that the target variable is not completely degenerate.

    Args:
        y: Binary or multi-class target series.
        min_minority_pct: Minimum percentage for the rarest class.
        raise_on_error: If True, raises on degenerate target.

    Returns:
        dict with 'valid', 'distribution', 'errors'.
    """
    dist = y.value_counts(normalize=True) * 100
    errors = []

    if len(dist) < 2:
        errors.append(f"Target has only {len(dist)} class(es) — model cannot learn")

    minority_pct = dist.min()
    if minority_pct < min_minority_pct:
        errors.append(
            f"Minority class is only {minority_pct:.1f}% — severe class imbalance "
            f"(threshold: {min_minority_pct}%)"
        )

    is_valid = len(errors) == 0

    if is_valid:
        print(f"[VALIDATION] Target distribution OK:\n{dist.to_string()}")
    else:
        print(f"[VALIDATION] Target distribution WARNING:")
        for e in errors:
            print(f"  {e}")

    if raise_on_error and not is_valid:
        raise DataValidationError("\n".join(errors))

    return {"valid": is_valid, "distribution": dist.to_dict(), "errors": errors}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    from config import Config
    from data_loader import load_data, define_target

    print("=" * 60)
    print("Running Data Validation on temp_data/financial_featured.csv")
    print("=" * 60)

    try:
        df = load_data()
        result = validate_financial_data(df, raise_on_error=False)

        y = define_target(df)
        target_result = validate_target_distribution(y, raise_on_error=False)

        if result["valid"] and target_result["valid"]:
            print("\nAll validations PASSED.")
        else:
            print("\nSome validations FAILED — review errors above.")
            sys.exit(1)

    except FileNotFoundError:
        print("Data file not found. Run the Airflow pipeline first or place CSV in temp_data/.")
        sys.exit(1)
