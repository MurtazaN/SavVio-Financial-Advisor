"""
Data Ingestion Orchestrator for SavVio Pipeline
Routes data loading to appropriate source (GCS or API) based on environment configuration.
This is the main entry point for the data ingestion step in the Airflow DAG.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple
import pandas as pd

from src.ingestion.config import (
    ENVIRONMENT,
    DATA_SOURCE,
    GCS_BUCKET_NAME,
    FINANCIAL_BLOB,
    PRODUCT_BLOB,
    REVIEW_BLOB,
    FINANCIAL_RAW_PATH,
    PRODUCT_RAW_PATH,
    REVIEW_RAW_PATH,
    GCP_CREDENTIALS_PATH,
    GCP_PROJECT_ID,
    API_BASE_URL,
    API_KEY,
    API_TIMEOUT,
    FINANCIAL_API_ENDPOINT,
    PRODUCT_API_ENDPOINT,
    REVIEW_API_ENDPOINT,
    get_config_summary
)

# Configure logging
from src.utils import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


def load_from_gcs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all datasets from Google Cloud Storage in parallel (download + parse).
    
    Returns:
        Tuple of (financial_df, product_df, review_df)
    """
    logger.info("=" * 80)
    logger.info("DATA SOURCE: Google Cloud Storage (GCS)")
    logger.info("=" * 80)
    
    from src.ingestion.gcs_loader import load_financial_data, load_product_data, load_review_data
    
    def _load_financial():
        return load_financial_data(
            bucket_name=GCS_BUCKET_NAME,
            blob_name=FINANCIAL_BLOB,
            destination_path=FINANCIAL_RAW_PATH,
            credentials_path=GCP_CREDENTIALS_PATH,
            project_id=GCP_PROJECT_ID,
        )
    
    def _load_product():
        return load_product_data(
            bucket_name=GCS_BUCKET_NAME,
            blob_name=PRODUCT_BLOB,
            destination_path=PRODUCT_RAW_PATH,
            credentials_path=GCP_CREDENTIALS_PATH,
            project_id=GCP_PROJECT_ID,
        )
    
    def _load_review():
        return load_review_data(
            bucket_name=GCS_BUCKET_NAME,
            blob_name=REVIEW_BLOB,
            destination_path=REVIEW_RAW_PATH,
            credentials_path=GCP_CREDENTIALS_PATH,
            project_id=GCP_PROJECT_ID,
        )
    
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_financial = executor.submit(_load_financial)
            future_product = executor.submit(_load_product)
            future_review = executor.submit(_load_review)
            financial_df = future_financial.result()
            product_df = future_product.result()
            review_df = future_review.result()
        
        logger.info("=" * 80)
        logger.info("GCS DATA LOADING COMPLETE")
        logger.info(f"Financial records: {len(financial_df)}")
        logger.info(f"Product records: {len(product_df)}")
        logger.info(f"Review records: {len(review_df)}")
        logger.info("=" * 80)
        
        return financial_df, product_df, review_df
        
    except Exception as e:
        logger.error(f"Failed to load data from GCS: {e}")
        raise


def load_from_api() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all datasets from API endpoints.
    
    Returns:
        Tuple of (financial_df, product_df, review_df)
    """
    logger.info("=" * 80)
    logger.info("DATA SOURCE: REST API")
    logger.info("=" * 80)
    
    # Import API loader
    from src.ingestion.api_loader import load_financial_data, load_product_data, load_review_data
    
    try:
        # Load financial data
        financial_df = load_financial_data(
            api_base_url=API_BASE_URL,
            endpoint=FINANCIAL_API_ENDPOINT.replace(API_BASE_URL, ''),  # Extract endpoint path
            destination_path=FINANCIAL_RAW_PATH,
            api_key=API_KEY,
            timeout=API_TIMEOUT
        )
        
        # Load product data
        product_df = load_product_data(
            api_base_url=API_BASE_URL,
            endpoint=PRODUCT_API_ENDPOINT.replace(API_BASE_URL, ''),
            destination_path=PRODUCT_RAW_PATH,
            api_key=API_KEY,
            timeout=API_TIMEOUT
        )
        
        # Load review data
        review_df = load_review_data(
            api_base_url=API_BASE_URL,
            endpoint=REVIEW_API_ENDPOINT.replace(API_BASE_URL, ''),
            destination_path=REVIEW_RAW_PATH,
            api_key=API_KEY,
            timeout=API_TIMEOUT
        )
        
        logger.info("=" * 80)
        logger.info("API DATA LOADING COMPLETE")
        logger.info(f"Financial records: {len(financial_df)}")
        logger.info(f"Product records: {len(product_df)}")
        logger.info(f"Review records: {len(review_df)}")
        logger.info("=" * 80)
        
        return financial_df, product_df, review_df
        
    except Exception as e:
        logger.error(f"Failed to load data from API: {e}")
        raise


def run_ingestion() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main ingestion function that routes to appropriate data source.
    This is the function called by Airflow DAG.
    
    Returns:
        Tuple of (financial_df, product_df, review_df)
        
    Raises:
        ValueError: If data source is not supported
        Exception: If data loading fails
    """
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 25 + "SAVVIO DATA INGESTION" + " " * 32 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    
    # Log configuration
    logger.info("\nConfiguration:")
    config_summary = get_config_summary()
    for key, value in config_summary.items():
        logger.info(f"  {key}: {value}")
    
    logger.info(f"\nEnvironment: {ENVIRONMENT}")
    logger.info(f"Data Source: {DATA_SOURCE}")
    
    try:
        # Route to appropriate loader based on data source
        if DATA_SOURCE.lower() == 'gcs':
            financial_df, product_df, review_df = load_from_gcs()
        elif DATA_SOURCE.lower() == 'api':
            financial_df, product_df, review_df = load_from_api()
        else:
            raise ValueError(
                f"Unsupported data source: {DATA_SOURCE}. "
                "Must be 'gcs' or 'api'"
            )
        
        # Validate that we got data
        if financial_df.empty:
            logger.warning("Financial data is empty!")
        if product_df.empty:
            logger.warning("Product data is empty!")
        if review_df.empty:
            logger.warning("Review data is empty!")
        
        logger.info("\n" + "=" * 80)
        logger.info("DATA INGESTION SUCCESSFUL")
        logger.info("=" * 80)
        
        return financial_df, product_df, review_df
        
    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error("DATA INGESTION FAILED")
        logger.error(f"Error: {e}")
        logger.error("=" * 80)
        raise


# ──────────────────────────────────────────────────────────────
# Individual Airflow task functions (for parallel ingestion)
# ──────────────────────────────────────────────────────────────

def ingest_financial_task(**context):
    """Airflow task: ingest financial data only."""
    logger.info("Ingesting financial data...")
    if DATA_SOURCE.lower() == 'gcs':
        from src.ingestion.gcs_loader import load_financial_data
        df = load_financial_data(
            bucket_name=GCS_BUCKET_NAME, blob_name=FINANCIAL_BLOB,
            destination_path=FINANCIAL_RAW_PATH,
            credentials_path=GCP_CREDENTIALS_PATH, project_id=GCP_PROJECT_ID,
        )
    else:
        from src.ingestion.api_loader import load_financial_data
        df = load_financial_data(
            api_base_url=API_BASE_URL,
            endpoint=FINANCIAL_API_ENDPOINT.replace(API_BASE_URL, ''),
            destination_path=FINANCIAL_RAW_PATH,
            api_key=API_KEY, timeout=API_TIMEOUT,
        )
    context['ti'].xcom_push(key='financial_path', value=FINANCIAL_RAW_PATH)
    logger.info(f"Financial ingestion complete: {len(df)} records")
    return {'records': len(df), 'status': 'success'}


def ingest_product_task(**context):
    """Airflow task: ingest product data only."""
    logger.info("Ingesting product data...")
    if DATA_SOURCE.lower() == 'gcs':
        from src.ingestion.gcs_loader import load_product_data
        df = load_product_data(
            bucket_name=GCS_BUCKET_NAME, blob_name=PRODUCT_BLOB,
            destination_path=PRODUCT_RAW_PATH,
            credentials_path=GCP_CREDENTIALS_PATH, project_id=GCP_PROJECT_ID,
        )
    else:
        from src.ingestion.api_loader import load_product_data
        df = load_product_data(
            api_base_url=API_BASE_URL,
            endpoint=PRODUCT_API_ENDPOINT.replace(API_BASE_URL, ''),
            destination_path=PRODUCT_RAW_PATH,
            api_key=API_KEY, timeout=API_TIMEOUT,
        )
    context['ti'].xcom_push(key='product_path', value=PRODUCT_RAW_PATH)
    logger.info(f"Product ingestion complete: {len(df)} records")
    return {'records': len(df), 'status': 'success'}


def ingest_review_task(**context):
    """Airflow task: ingest review data only."""
    logger.info("Ingesting review data...")
    if DATA_SOURCE.lower() == 'gcs':
        from src.ingestion.gcs_loader import load_review_data
        df = load_review_data(
            bucket_name=GCS_BUCKET_NAME, blob_name=REVIEW_BLOB,
            destination_path=REVIEW_RAW_PATH,
            credentials_path=GCP_CREDENTIALS_PATH, project_id=GCP_PROJECT_ID,
        )
    else:
        from src.ingestion.api_loader import load_review_data
        df = load_review_data(
            api_base_url=API_BASE_URL,
            endpoint=REVIEW_API_ENDPOINT.replace(API_BASE_URL, ''),
            destination_path=REVIEW_RAW_PATH,
            api_key=API_KEY, timeout=API_TIMEOUT,
        )
    context['ti'].xcom_push(key='review_path', value=REVIEW_RAW_PATH)
    logger.info(f"Review ingestion complete: {len(df)} records")
    return {'records': len(df), 'status': 'success'}


# Standalone execution for testing
if __name__ == "__main__":
    try:
        logger.info("Running data ingestion in standalone mode...")
        financial_df, product_df, review_df = run_ingestion()
        
        print("\n" + "=" * 80)
        print("INGESTION SUMMARY")
        print("=" * 80)
        print(f"\nFinancial Data:")
        print(f"  Shape: {financial_df.shape}")
        print(f"  Columns: {list(financial_df.columns)}")
        print(f"  Sample:\n{financial_df.head(3)}")
        
        print(f"\nProduct Data:")
        print(f"  Shape: {product_df.shape}")
        print(f"  Columns: {list(product_df.columns)}")
        print(f"  Sample:\n{product_df.head(3)}")
        
        print(f"\nReview Data:")
        print(f"  Shape: {review_df.shape}")
        print(f"  Columns: {list(review_df.columns)}")
        print(f"  Sample:\n{review_df.head(3)}")
        
        print("\n" + "=" * 80)
        print("Standalone ingestion test completed successfully")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Standalone execution failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)