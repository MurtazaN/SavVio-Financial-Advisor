#!/usr/bin/env python3
"""
Orchestrator script for running all preprocessing tasks (Financial, Product, Review).
Run this script to execute the full preprocessing pipeline in order.
"""

import sys
import os
import logging
from pathlib import Path

from src.preprocess import financial, product, review
from src.utils import setup_logging

# Ensure running from dags root so relative data paths (data/raw, data/processed) work correctly
current_script_path = Path(__file__).resolve()
dags_root = current_script_path.parent.parent.parent  # .../dags/
if os.getcwd() != str(dags_root):
    print(f"Changing working directory to: {dags_root}")
    os.chdir(dags_root)

setup_logging()
logger = logging.getLogger("run_preprocessing")

def run_pipeline():
    logger.info("=" * 60)
    logger.info("STARTING DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # 1. Financial Data
    try:
        logger.info("\n>>> [1/3] Processing Financial Data...")
        financial.main()
        logger.info(">>> Financial Data Preprocessing: SUCCESS")
    except Exception as e:
        logger.error(f">>> Financial Data Preprocessing: FAILED\nError: {e}")
        sys.exit(1)

    # 2. Product Data
    try:
        logger.info("\n>>> [2/3] Processing Product Data...")
        product.main()
        logger.info(">>> Product Data Preprocessing: SUCCESS")
    except Exception as e:
        logger.error(f">>> Product Data Preprocessing: FAILED\nError: {e}")
        sys.exit(1)

    # 3. Review Data
    try:
        logger.info("\n>>> [3/3] Processing Review Data...")
        review.main()
        logger.info(">>> Review Data Preprocessing: SUCCESS")
    except Exception as e:
        logger.error(f">>> Review Data Preprocessing: FAILED\nError: {e}")
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("ALL PREPROCESSING TASKS COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────
# Individual Airflow task functions (for parallel preprocessing)
# ──────────────────────────────────────────────────────────────

def preprocess_financial_task(**context):
    """Airflow task: preprocess financial data only."""
    logger.info(">>> Preprocessing Financial Data...")
    financial.main()
    logger.info(">>> Financial Data Preprocessing: SUCCESS")


def preprocess_product_task(**context):
    """Airflow task: preprocess product data only."""
    logger.info(">>> Preprocessing Product Data...")
    product.main()
    logger.info(">>> Product Data Preprocessing: SUCCESS")


def preprocess_review_task(**context):
    """Airflow task: preprocess review data only."""
    logger.info(">>> Preprocessing Review Data...")
    review.main()
    logger.info(">>> Review Data Preprocessing: SUCCESS")


if __name__ == "__main__":
    run_pipeline()
