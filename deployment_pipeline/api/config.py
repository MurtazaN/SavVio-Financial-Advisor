"""
API Configuration — Settings for FastAPI, database, CORS, and logging.
Reads from .env with fallback defaults for local development.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------

# deployment_pipeline/api/config.py → deployment_pipeline/ → SavVio/
_DEPLOYMENT_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _DEPLOYMENT_DIR.parent

# Load environment variables from project and pipeline dirs
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_PROJECT_ROOT / "model_pipeline" / ".env")


class APIConfig:
    """Configuration for the SavVio deployment API."""

    # ---------------------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------------------
    PROJECT_ROOT = _PROJECT_ROOT
    DEPLOYMENT_DIR = _DEPLOYMENT_DIR

    # Model artifact — single bundled pyfunc saved by run_pipeline.py's save_best_model_local()
    MODEL_ARTIFACT_DIR = os.getenv(
        "MODEL_ARTIFACT_DIR",
        str(_PROJECT_ROOT / "model_pipeline" / "models" / "model"),
    )

    # ---------------------------------------------------------------------------
    # Database
    # ---------------------------------------------------------------------------
    # "dev" or "prod" — passed to savviocore.database.db_connection.get_engine()
    DB_ENV = os.getenv("DB_ENV", "dev")

    # ---------------------------------------------------------------------------
    # Server
    # ---------------------------------------------------------------------------
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "3500"))

    # CORS — Vite dev server (3000), Streamlit, etc.
    CORS_ORIGINS = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8501,*",
    ).split(",")

    # ---------------------------------------------------------------------------
    # API Metadata
    # ---------------------------------------------------------------------------
    API_TITLE = "SavVio API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = (
        "AI-driven financial purchase recommendation API. "
        "Provides Green/Yellow/Red signals based on real-time financial health."
    )

    # ---------------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------------
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # ---------------------------------------------------------------------------
    # Product catalog browse (GET /products) — default price band when query omits min/max
    # ---------------------------------------------------------------------------
    PRODUCT_BROWSE_PRICE_MIN = float(os.getenv("PRODUCT_BROWSE_PRICE_MIN", "50"))
    PRODUCT_BROWSE_PRICE_MAX = float(os.getenv("PRODUCT_BROWSE_PRICE_MAX", "2000"))
    PRODUCT_BROWSE_DEFAULT_LIMIT = int(os.getenv("PRODUCT_BROWSE_DEFAULT_LIMIT", "100"))
    PRODUCT_BROWSE_MAX_LIMIT = int(os.getenv("PRODUCT_BROWSE_MAX_LIMIT", "500"))

    # ---------------------------------------------------------------------------
    # ML — display name for API / UI (e.g. technical panel)
    # ---------------------------------------------------------------------------
    ML_MODEL_DISPLAY_NAME = os.getenv("ML_MODEL_DISPLAY_NAME", "SavVio classifier")

    # ---------------------------------------------------------------------------
    # MLflow (for loading models from registry — optional)
    # ---------------------------------------------------------------------------
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")

    # ---------------------------------------------------------------------------
    # Monitoring (Prometheus / Grafana)
    # ---------------------------------------------------------------------------
    METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    METRICS_PREFIX = os.getenv("METRICS_PREFIX", "savvio")
    GRAFANA_CLOUD_REMOTE_WRITE_URL = os.getenv("GRAFANA_CLOUD_REMOTE_WRITE_URL", "")
    GRAFANA_CLOUD_USERNAME = os.getenv("GRAFANA_CLOUD_USERNAME", "")
    GRAFANA_CLOUD_API_KEY = os.getenv("GRAFANA_CLOUD_API_KEY", "")

    # ---------------------------------------------------------------------------
    # Model Pipeline Source Paths (for importing existing modules)
    # ---------------------------------------------------------------------------
    MODEL_PIPELINE_SRC = str(_PROJECT_ROOT / "model_pipeline" / "src")
    SAVVIOCORE_SRC = str(_PROJECT_ROOT / "savviocore" / "src")
