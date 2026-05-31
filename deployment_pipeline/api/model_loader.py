"""
Model Loader — Resource management for SavVio ML models, DB, and LLM providers.
"""

from __future__ import annotations

import logging
import os
import sys

from deployment_pipeline.api.config import APIConfig

logger = logging.getLogger(__name__)


def _ensure_import_paths():
    """Add model_pipeline/src and savviocore/src to sys.path."""
    for path in [APIConfig.MODEL_PIPELINE_SRC, APIConfig.SAVVIOCORE_SRC]:
        if path not in sys.path:
            sys.path.insert(0, path)
            logger.info("Added to sys.path: %s", path)


_ensure_import_paths()


class ModelManager:
    """Loads all inference resources and exposes a unified interface."""

    def __init__(self):
        self.model = None
        self.db_engine = None
        self.llm_provider = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self):
        """Load all resources. Call once at API startup."""
        self._load_model(APIConfig.MODEL_ARTIFACT_DIR)
        self._connect_db(APIConfig.DB_ENV)
        self._init_llm_provider()
        self._loaded = True
        logger.info("Model manager fully loaded.")

    def _load_model(self, artifact_dir: str) -> None:
        """Load the bundled MLflow pyfunc artifact."""
        logger.info("Loading model from: %s", artifact_dir)
        if not os.path.exists(artifact_dir):
            logger.error("Model artifact directory not found: %s", artifact_dir)
            return

        import mlflow.pyfunc
        self.model = mlflow.pyfunc.load_model(artifact_dir)
        logger.info("Loaded bundled pyfunc model from: %s", artifact_dir)

    def _connect_db(self, env: str) -> None:
        """Initialize the SQLAlchemy database engine."""
        try:
            from savviocore.database.db_connection import get_engine
            self.db_engine = get_engine(env=env)
            from sqlalchemy import text
            with self.db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connected (env=%s)", env)
        except Exception as e:
            logger.warning("Database initialization failed: %s", e)
            self.db_engine = None

    def _init_llm_provider(self) -> None:
        """Initialize the LLM provider."""
        try:
            from llm.llm_provider import get_provider
            self.llm_provider = get_provider()
            logger.info("LLM provider initialized: %s", self.llm_provider.provider_name)
        except Exception as e:
            logger.warning("LLM provider initialization failed: %s — using mock.", e)
            from llm.llm_provider import MockProvider
            self.llm_provider = MockProvider()

    def predict(self, raw_features_df):
        """Run the model. Returns result DataFrame with label, confidence, and financial features."""
        if self.model is None:
            return None
        return self.model.predict(raw_features_df)

    def check_db_connection(self) -> bool:
        if self.db_engine is None:
            return False
        try:
            from sqlalchemy import text
            with self.db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def get_llm_provider_name(self) -> str:
        if self.llm_provider is None:
            return "none"
        return self.llm_provider.provider_name

model_manager = ModelManager()
