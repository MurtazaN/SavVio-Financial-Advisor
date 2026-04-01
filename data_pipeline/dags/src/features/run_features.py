"""
Feature Engineering Orchestrator.

Runs the complete feature engineering pipeline:
1. Financial Feature Engineering
2. Review Feature Engineering (produces product_featured.jsonl + review_featured.jsonl)

Usage:
    python3 src/features/run_features.py
"""

import os
import logging
import argparse

from src.features.financial_features import run_financial_features
from src.features.product_review_features import run_review_features
from src.utils import setup_logging

# Configure module logging.
setup_logging()
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Feature Engineering Pipeline")
    parser.add_argument("--skip-financial", action="store_true", help="Skip financial features")
    parser.add_argument("--skip-reviews", action="store_true", help="Skip review features")
    args = parser.parse_args()

    # Resolve data paths (data is at dags/data).
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    # Financial input/output files.
    FIN_INPUT = os.path.join(DATA_DIR, "processed/financial_preprocessed.csv")
    FIN_OUTPUT = os.path.join(DATA_DIR, "features/financial_featured.csv")
    
    # Review & product input/output files.
    REV_INPUT = os.path.join(DATA_DIR, "processed/review_preprocessed.jsonl")
    PROD_INPUT = os.path.join(DATA_DIR, "processed/product_preprocessed.jsonl")
    PROD_OUTPUT = os.path.join(DATA_DIR, "features/product_featured.jsonl")
    REV_OUTPUT = os.path.join(DATA_DIR, "features/review_featured.jsonl")

    # Step 1: Financial features.
    if not args.skip_financial:
        logger.info("--- Starting Financial Feature Engineering ---")
        try:
            run_financial_features(FIN_INPUT, FIN_OUTPUT)
            logger.info("Financial features complete.")
        except Exception as e:
            logger.error(f"Financial feature engineering failed: {e}")

    # Step 2: Review features (produces product_featured.jsonl + review_featured.jsonl).
    if not args.skip_reviews:
        logger.info("--- Starting Review Feature Engineering ---")
        try:
            run_review_features(
                reviews_path=REV_INPUT,
                products_path=PROD_INPUT,
                product_output_path=PROD_OUTPUT,
                review_output_path=REV_OUTPUT,
            )
            logger.info("Review features complete.")
        except Exception as e:
            logger.error(f"Review feature engineering failed: {e}")

    logger.info("Feature engineering pipeline finished.")


# ---------- Airflow task wrappers ----------

def feature_financial_task(**context):
    """Airflow task: run financial feature engineering."""
    logger.info(">>> Running Financial Feature Engineering...")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    run_financial_features(
        input_path=os.path.join(DATA_DIR, "processed/financial_preprocessed.csv"),
        output_path=os.path.join(DATA_DIR, "features/financial_featured.csv"),
    )
    logger.info(">>> Financial Feature Engineering: SUCCESS")


def feature_review_task(**context):
    """Airflow task: run review feature engineering (produces product + review featured files)."""
    logger.info(">>> Running Review Feature Engineering...")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    run_review_features(
        reviews_path=os.path.join(DATA_DIR, "processed/review_preprocessed.jsonl"),
        products_path=os.path.join(DATA_DIR, "processed/product_preprocessed.jsonl"),
        product_output_path=os.path.join(DATA_DIR, "features/product_featured.jsonl"),
        review_output_path=os.path.join(DATA_DIR, "features/review_featured.jsonl"),
    )
    logger.info(">>> Review Feature Engineering: SUCCESS")


if __name__ == "__main__":
    main()