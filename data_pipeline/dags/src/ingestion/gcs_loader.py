"""
GCS Data Loader for SavVio Pipeline
Handles downloading and uploading financial and product data from/to Google Cloud Storage.
Supports both CSV and JSON formats.
"""

import os
import json
import logging
from typing import Optional
from pathlib import Path
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account

# Configure logging
from src.utils import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


class GCSLoader:
    """Handles data loading and uploading to/from Google Cloud Storage."""
    
    def __init__(self, credentials_path: Optional[str] = None, project_id: Optional[str] = None):
        """
        Initialize GCS client.
        
        Args:
            credentials_path: Path to GCS service account JSON key file.
                            If None, uses Application Default Credentials (ADC).
            project_id: GCP project ID (optional, inferred from credentials if not provided)
        """
        try:
            if credentials_path and os.path.exists(credentials_path):
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path
                )
                self.client = storage.Client(
                    credentials=credentials,
                    project=project_id
                )
                logger.info(f"GCS client initialized with credentials from {credentials_path}")
            else:
                # Use Application Default Credentials (works in GCP environments)
                self.client = storage.Client(project=project_id)
                logger.info("GCS client initialized with Application Default Credentials")
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise
    
    def download_blob(
        self, 
        bucket_name: str, 
        blob_name: str, 
        destination_path: str
    ) -> str:
        """
        Download a single blob from GCS bucket if it has changed.
        Compares remote MD5 with local file MD5 to skip unchanged files.
        Data versioning and caching are handled by DVC when tracking this directory.
        
        Args:
            bucket_name: Name of the GCS bucket
            blob_name: Path to the file in the bucket (e.g., 'raw/financial.csv')
            destination_path: Local path where file should be saved
            
        Returns:
            str: Path to the downloaded file
            
        Raises:
            FileNotFoundError: If blob doesn't exist
            Exception: If download fails
        """
        try:
            # Create destination directory if it doesn't exist
            Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Get bucket and blob
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            # Check if blob exists
            if not blob.exists():
                raise FileNotFoundError(
                    f"Blob '{blob_name}' not found in bucket '{bucket_name}'"
                )
            
            # --- Incremental: skip download if local file matches remote ---
            blob.reload()  # fetch metadata including md5_hash
            remote_md5 = blob.md5_hash  # base64-encoded MD5
            if remote_md5 and os.path.exists(destination_path):
                import base64, hashlib
                local_md5 = hashlib.md5(open(destination_path, "rb").read()).digest()
                local_md5_b64 = base64.b64encode(local_md5).decode()
                if local_md5_b64 == remote_md5:
                    logger.info(
                        f"File unchanged (MD5 match) — skipping download: "
                        f"gs://{bucket_name}/{blob_name}"
                    )
                    return destination_path
            
            # Download the blob (file is new or has changed)
            logger.info(f"Downloading gs://{bucket_name}/{blob_name} to {destination_path}")
            blob.download_to_filename(destination_path)
            
            # Verify download
            if not os.path.exists(destination_path):
                raise FileNotFoundError(f"Failed to download file to {destination_path}")
            
            file_size = os.path.getsize(destination_path)
            logger.info(f"Successfully downloaded {file_size} bytes to {destination_path}")
            
            return destination_path
            
        except Exception as e:
            logger.error(f"Failed to download blob: {e}")
            raise
    
    def upload_blob(
        self,
        bucket_name: str,
        source_path: str,
        destination_blob_name: str,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload a file to GCS bucket.
        
        Args:
            bucket_name: Name of the GCS bucket
            source_path: Local path to the file to upload
            destination_blob_name: Destination path in GCS (e.g., 'processed/financial.csv')
            content_type: Optional MIME type (e.g., 'text/csv', 'application/json')
            
        Returns:
            str: GCS URI of uploaded file (gs://bucket/path)
            
        Raises:
            FileNotFoundError: If source file doesn't exist
            Exception: If upload fails
        """
        try:
            # Verify source file exists
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Source file not found: {source_path}")
            
            # Get bucket and create blob
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)
            
            # Set content type if provided
            if content_type:
                blob.content_type = content_type
            
            # Upload the file
            file_size = os.path.getsize(source_path)
            logger.info(f"Uploading {source_path} ({file_size} bytes) to gs://{bucket_name}/{destination_blob_name}")
            
            blob.upload_from_filename(source_path)
            
            gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
            logger.info(f"Successfully uploaded to {gcs_uri}")
            
            return gcs_uri
            
        except Exception as e:
            logger.error(f"Failed to upload blob: {e}")
            raise
    
    def load_csv_from_gcs(
        self,
        bucket_name: str,
        blob_name: str,
        destination_path: str
    ) -> pd.DataFrame:
        """
        Download CSV from GCS and load into a DataFrame.
        Raw financial data is always CSV; product and review are JSONL (use load_json_from_gcs).
        
        Args:
            bucket_name: Name of the GCS bucket
            blob_name: Path to the CSV file in the bucket
            destination_path: Local path where file should be saved
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # Download the file
            local_path = self.download_blob(bucket_name, blob_name, destination_path)
            
            # Load into DataFrame
            logger.info(f"Loading CSV from {local_path}")
            df = pd.read_csv(local_path)
            
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV from GCS: {e}")
            raise
    
    def load_json_from_gcs(
        self,
        bucket_name: str,
        blob_name: str,
        destination_path: str
    ) -> pd.DataFrame:
        """
        Download JSONL (JSON Lines) from GCS and load into a DataFrame.
        Raw product and review data are always JSONL; financial is always CSV (use load_csv_from_gcs).
        
        Args:
            bucket_name: Name of the GCS bucket
            blob_name: Path to the JSONL file in the bucket
            destination_path: Local path where file should be saved
            
        Returns:
            pd.DataFrame: One row per line in the JSONL file
        """
        try:
            local_path = self.download_blob(bucket_name, blob_name, destination_path)
            logger.info(f"Loading JSONL from {local_path}")
            # df = pd.read_json(local_path, lines=True)
            # Chunked reading — reduces peak memory ~3-4x for large files (e.g. 929 MB review data)
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.1f} MB")
            if file_size_mb > 300: # To handle large files - Adjust as needed - default 300MB
                logger.info(f"Large file detected — reading in chunks of 100,000 rows") # Default chunk size 100_000 rows - 20 to 30 MB per chunk
                chunks = pd.read_json(local_path, lines=True, chunksize=100_000)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_json(local_path, lines=True)
            logger.info(f"Loaded {len(df)} records and {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Failed to load JSONL from GCS: {e}")
            raise
    
    def upload_dataframe(
        self,
        df: pd.DataFrame,
        bucket_name: str,
        destination_blob_name: str,
        format: str = 'csv'
    ) -> str:
        """
        Upload pandas DataFrame to GCS as CSV or JSON.
        
        Args:
            df: DataFrame to upload
            bucket_name: Name of the GCS bucket
            destination_blob_name: Destination path in GCS
            format: 'csv' or 'json'
            
        Returns:
            str: GCS URI of uploaded file
        """
        try:
            # Create temporary local file
            temp_dir = Path('data/temp')
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'csv':
                temp_path = temp_dir / f"temp_{Path(destination_blob_name).name}"
                df.to_csv(temp_path, index=False)
                content_type = 'text/csv'
            elif format.lower() == 'json':
                temp_path = temp_dir / f"temp_{Path(destination_blob_name).name}"
                df.to_json(temp_path, orient='records', indent=2)
                content_type = 'application/json'
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'")
            
            logger.info(f"Saving DataFrame ({df.shape[0]} rows) as {format.upper()}")
            
            # Upload to GCS
            gcs_uri = self.upload_blob(
                bucket_name=bucket_name,
                source_path=str(temp_path),
                destination_blob_name=destination_blob_name,
                content_type=content_type
            )
            
            # Clean up temp file
            temp_path.unlink()
            logger.info(f"Cleaned up temporary file: {temp_path}")
            
            return gcs_uri
            
        except Exception as e:
            logger.error(f"Failed to upload DataFrame: {e}")
            raise
    
    def list_blobs(self, bucket_name: str, prefix: Optional[str] = None) -> list:
        """
        List all blobs in a bucket with optional prefix filter.
        
        Args:
            bucket_name: Name of the GCS bucket
            prefix: Optional prefix to filter blobs (e.g., 'raw/')
            
        Returns:
            list: List of blob names
        """
        try:
            bucket = self.client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            
            blob_names = [blob.name for blob in blobs]
            logger.info(f"Found {len(blob_names)} blobs in gs://{bucket_name}/{prefix or ''}")
            
            return blob_names
            
        except Exception as e:
            logger.error(f"Failed to list blobs: {e}")
            raise


# Convenience functions for SavVio data pipeline

def load_financial_data(
    bucket_name: str,
    blob_name: str,
    destination_path: str = "data/raw/financial_data.csv",
    credentials_path: Optional[str] = None,
    project_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Load financial data (CSV) from GCS bucket.
    
    Args:
        bucket_name: GCS bucket name
        blob_name: Path to financial data CSV in bucket
        destination_path: Local destination path
        credentials_path: Optional path to service account credentials
        project_id: Optional GCP project ID
        
    Returns:
        pd.DataFrame: Financial data
    """
    logger.info("=" * 60)
    logger.info("Loading Financial Data from GCS (CSV)")
    logger.info("=" * 60)
    
    loader = GCSLoader(credentials_path, project_id)
    df = loader.load_csv_from_gcs(bucket_name, blob_name, destination_path)
    
    logger.info(f"Financial data loaded successfully: {df.shape}")
    return df


def load_product_data(
    bucket_name: str,
    blob_name: str,
    destination_path: str = "data/raw/product_data.json",
    credentials_path: Optional[str] = None,
    project_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Load product data (JSON) from GCS bucket.
    
    Args:
        bucket_name: GCS bucket name
        blob_name: Path to product data JSON in bucket
        destination_path: Local destination path
        credentials_path: Optional path to service account credentials
        project_id: Optional GCP project ID
        
    Returns:
        pd.DataFrame: Product data
    """
    logger.info("=" * 60)
    logger.info("Loading Product Data from GCS (JSON)")
    logger.info("=" * 60)
    
    loader = GCSLoader(credentials_path, project_id)
    data = loader.load_json_from_gcs(bucket_name, blob_name, destination_path)
    
    # Ensure it's a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Product JSON must be a list of records")
    
    logger.info(f"Product data loaded successfully: {data.shape}")
    return data


def load_review_data(
    bucket_name: str,
    blob_name: str,
    destination_path: str = "data/raw/review_data.json",
    credentials_path: Optional[str] = None,
    project_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Load product review data (JSON) from GCS bucket.
    
    Args:
        bucket_name: GCS bucket name
        blob_name: Path to review data JSON in bucket
        destination_path: Local destination path
        credentials_path: Optional path to service account credentials
        project_id: Optional GCP project ID
        
    Returns:
        pd.DataFrame: Review data
    """
    logger.info("=" * 60)
    logger.info("Loading Product Review Data from GCS (JSON)")
    logger.info("=" * 60)
    
    loader = GCSLoader(credentials_path, project_id)
    data = loader.load_json_from_gcs(bucket_name, blob_name, destination_path)
    
    # Ensure it's a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Review JSON must be a list of records")
    
    logger.info(f"Review data loaded successfully: {data.shape}")
    return data


# Example usage and testing
if __name__ == "__main__":
    # This will be replaced by config.py in actual usage
    from dotenv import load_dotenv
    load_dotenv()
    
    BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "savvio-data-bucket")
    CREDENTIALS_PATH = os.getenv("GCP_CREDENTIALS_PATH")
    PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    
    try:
        # Example: Load financial data
        financial_df = load_financial_data(
            bucket_name=BUCKET_NAME,
            blob_name="raw/financial_data.csv",
            credentials_path=CREDENTIALS_PATH,
            project_id=PROJECT_ID
        )
        print("\nFinancial Data Preview:")
        print(financial_df.head())
        
        # Example: Load product data
        product_df = load_product_data(
            bucket_name=BUCKET_NAME,
            blob_name="raw/product_data.json",
            credentials_path=CREDENTIALS_PATH,
            project_id=PROJECT_ID
        )
        print("\nProduct Data Preview:")
        print(product_df.head())
        
        # Example: Upload processed data back to GCS
        loader = GCSLoader(CREDENTIALS_PATH, PROJECT_ID)
        gcs_uri = loader.upload_dataframe(
            df=financial_df,
            bucket_name=BUCKET_NAME,
            destination_blob_name="processed/financial_processed.csv",
            format='csv'
        )
        print(f"\nUploaded to: {gcs_uri}")
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise