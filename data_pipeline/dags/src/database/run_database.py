"""
Database Loading Orchestrator – Airflow task wrappers.

Wraps the upload_to_db and vector_embed modules so Airflow
PythonOperator can call them with a consistent *_task(**context) interface.
"""

import os
import logging


from src.database.upload_to_db import load_financial, load_products, load_reviews, load_all
from savviocore.database.db_connection import get_engine, ensure_pgvector
from savviocore.database.db_schema import create_tables
from src.database.vector_embed import (
    load_model,
    embed_products,
    embed_reviews,
    _ensure_embedding_tables,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared setup — engine, tables, data path (called once per task)
# ---------------------------------------------------------------------------

def _setup():
    """Create engine, return (data_dir, engine)."""
    # __file__ = .../dags/src/database/run_database.py
    # We need 3 dirname() calls to reach .../dags/
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base, "data")
    engine = get_engine()
    return data_dir, engine


# ---------------------------------------------------------------------------
# Airflow task wrappers
# ---------------------------------------------------------------------------

#TODO: data added to DB might change after bias detection and mitigation

def setup_database_task(**context):
    """Airflow task: create PostgreSQL tables if they don't already exist."""
    logger.info(">>> Setting up database schema (create if not exists)...")
    engine = get_engine()
    from savviocore.database.db_schema import create_tables
    create_tables(engine)
    logger.info(">>> Database Setup: SUCCESS")

def load_financial_task(**context):
    """Airflow task: load financial profiles into PostgreSQL."""
    logger.info(">>> Loading Financial Profiles into PostgreSQL...")
    data, engine = _setup()
    load_financial(engine, os.path.join(data, "features/financial_featured.csv"))
    logger.info(">>> Financial Profiles Load: SUCCESS")


def load_products_task(**context):
    """Airflow task: load products into PostgreSQL."""
    logger.info(">>> Loading Products into PostgreSQL...")
    data, engine = _setup()
    load_products(engine, os.path.join(data, "features/product_featured.jsonl"))
    logger.info(">>> Products Load: SUCCESS")


def load_reviews_task(**context):
    """Airflow task: load reviews into PostgreSQL."""
    logger.info(">>> Loading Reviews into PostgreSQL...")
    data, engine = _setup()
    load_reviews(engine, os.path.join(data, "features/review_featured.jsonl"))
    logger.info(">>> Reviews Load: SUCCESS")


def generate_and_load_embedding_task(**context):
    """Airflow task: generate embeddings for products & reviews, store in pgvector."""
    logger.info(">>> Generating Embeddings and Loading into pgvector...")
    data, engine = _setup()
    ensure_pgvector(engine)
    _ensure_embedding_tables(engine)

    model = load_model()

    n_prod = embed_products(
        engine,
        os.path.join(data, "features/product_featured.jsonl"),
        model,
    )
    logger.info("Embedded %d products", n_prod)

    n_rev = embed_reviews(
        engine,
        os.path.join(data, "features/review_featured.jsonl"),
        model,
    )
    logger.info("Embedded %d reviews", n_rev)
    logger.info(">>> Embedding Generation & Load: SUCCESS")


# ---------------------------------------------------------------------------
# CLI – optional, for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.utils import setup_logging
    setup_logging()
    logger.info("Running full database loading pipeline...")
    load_financial_task()
    load_products_task()
    load_reviews_task()
    generate_and_load_embedding_task()
    logger.info("All done.")
