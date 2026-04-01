"""
Configuration Management for SavVio Data Pipeline
Loads configuration from environment variables (.env file)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# This will only work if .env exists (local development)
# In production/Airflow, environment variables are set directly
load_dotenv()


# ============================================================================
# GCP Configuration
# ============================================================================

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_CREDENTIALS_PATH = os.getenv("GCP_CREDENTIALS_PATH")

# Validate GCP credentials path if provided
if GCP_CREDENTIALS_PATH and not os.path.exists(GCP_CREDENTIALS_PATH):
    raise FileNotFoundError(
        f"GCP credentials file not found: {GCP_CREDENTIALS_PATH}"
    )


# ============================================================================
# GCS Storage Configuration
# ============================================================================

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "savvio-data-bucket")

# GCS folder paths
GCS_RAW_PATH = os.getenv("GCS_RAW_PATH", "raw/")
GCS_PROCESSED_PATH = os.getenv("GCS_PROCESSED_PATH", "processed/")
GCS_FEATURES_PATH = os.getenv("GCS_FEATURES_PATH", "features/")
GCS_VALIDATED_PATH = os.getenv("GCS_VALIDATED_PATH", "validated/")

# Specific blob paths for each dataset
FINANCIAL_BLOB = os.getenv("FINANCIAL_BLOB", "raw/financial_data.csv")
PRODUCT_BLOB = os.getenv("PRODUCT_BLOB", "raw/product_data.jsonl")
REVIEW_BLOB = os.getenv("REVIEW_BLOB", "raw/review_data.jsonl")



# ============================================================================
# Local File Paths
# ============================================================================


# Define Project Root (data_pipeline directory)
# config.py is in data_pipeline/ingestion-scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent

# Base directory for data storage
_data_dir_env = os.getenv("DATA_DIR")
if _data_dir_env:
    _path = Path(_data_dir_env)
    if _path.is_absolute():
        DATA_DIR = _path
    else:
        # Assume relative paths in .env are relative to repository root
        DATA_DIR = REPO_ROOT / _path
else:
    # Default to data_pipeline/data
    DATA_DIR = PROJECT_ROOT / "data"

# Subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VALIDATED_DATA_DIR = DATA_DIR / "validated"
FEATURES_DATA_DIR = DATA_DIR / "features"
TEMP_DATA_DIR = DATA_DIR / "temp"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VALIDATED_DATA_DIR, 
                  FEATURES_DATA_DIR, TEMP_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Specific file paths
FINANCIAL_RAW_PATH = str(RAW_DATA_DIR / "financial_data.csv")
PRODUCT_RAW_PATH = str(RAW_DATA_DIR / "product_data.jsonl")
REVIEW_RAW_PATH = str(RAW_DATA_DIR / "review_data.jsonl")


# ============================================================================
# API Configuration (for production)
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.savvio.com/v1")
API_KEY = os.getenv("API_KEY")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

# API endpoints
FINANCIAL_API_ENDPOINT = os.getenv("FINANCIAL_API_ENDPOINT", f"{API_BASE_URL}/financial")
PRODUCT_API_ENDPOINT = os.getenv("PRODUCT_API_ENDPOINT", f"{API_BASE_URL}/products")
REVIEW_API_ENDPOINT = os.getenv("REVIEW_API_ENDPOINT", f"{API_BASE_URL}/reviews")


# ============================================================================
# Environment Configuration
# ============================================================================

ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")  # dev, staging, prod

# Determine data source based on environment
if ENVIRONMENT == "dev":
    DATA_SOURCE = "gcs"
elif ENVIRONMENT == "prod":
    DATA_SOURCE = "api"
else:
    DATA_SOURCE = os.getenv("DATA_SOURCE", "gcs")


# ============================================================================
# Pipeline Configuration
# ============================================================================

# Data validation thresholds
MAX_MISSING_VALUES_PCT = float(os.getenv("MAX_MISSING_VALUES_PCT", "0.1"))  # 10%
MIN_RECORDS_REQUIRED = int(os.getenv("MIN_RECORDS_REQUIRED", "100"))

# Feature engineering parameters
MONTHLY_INCOME_COLS = os.getenv("MONTHLY_INCOME_COLS", "income,salary").split(",")
MONTHLY_EXPENSE_COLS = os.getenv("MONTHLY_EXPENSE_COLS", "rent,bills,subscriptions").split(",")


# ============================================================================
# Logging Configuration
# ============================================================================



LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

_log_dir_env = os.getenv("LOG_DIR")
if _log_dir_env:
    _path = Path(_log_dir_env)
    if _path.is_absolute():
        LOG_DIR = _path
    else:
        # Assume relative paths in .env are relative to repository root
        LOG_DIR = REPO_ROOT / _path
else:
    # Default to data_pipeline/logs
    LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DVC Configuration
# ============================================================================

DVC_REMOTE_NAME = os.getenv("DVC_REMOTE_NAME", "gcs")
DVC_REMOTE_URL = os.getenv("DVC_REMOTE_URL", f"gs://{GCS_BUCKET_NAME}/dvc-store")


# ============================================================================
# Configuration Validation
# ============================================================================

def validate_config():
    """
    Validate that required configuration values are set.
    Raises ValueError if critical configs are missing.
    """
    errors = []
    
    # Check required GCS configs for dev environment
    if ENVIRONMENT == "dev" or DATA_SOURCE == "gcs":
        if not GCS_BUCKET_NAME:
            errors.append("GCS_BUCKET_NAME is required for GCS data source")
    
    # Check required API configs for prod environment
    if ENVIRONMENT == "prod" or DATA_SOURCE == "api":
        if not API_BASE_URL:
            errors.append("API_BASE_URL is required for API data source")
        if not API_KEY:
            errors.append("API_KEY is required for API data source")
    
    if errors:
        raise ValueError(
            "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def get_config_summary() -> dict:
    """
    Get a summary of current configuration (for logging/debugging).
    Excludes sensitive information like API keys.
    
    Returns:
        dict: Configuration summary
    """
    return {
        "environment": ENVIRONMENT,
        "data_source": DATA_SOURCE,
        "gcp_project": GCP_PROJECT_ID,
        "gcs_bucket": GCS_BUCKET_NAME,
        "api_base_url": API_BASE_URL,
        "log_level": LOG_LEVEL,
        "data_dir": str(DATA_DIR),
    }


# Run validation when module is imported
try:
    validate_config()
except ValueError as e:
    print(f"WARNING: {e}")
    print("Some features may not work correctly.")


# For debugging: print config when module is run directly
if __name__ == "__main__":
    import json
    print("=" * 60)
    print("SavVio Configuration Summary")
    print("=" * 60)
    print(json.dumps(get_config_summary(), indent=2))
    print("=" * 60)