"""
Database connection configuration for SavVio.
Supports local (dev) and GCP Cloud SQL (prod) environments.

Environment variables required:
  DB_USER, DB_PASSWORD, DB_NAME
  For prod: GCP_PROJECT, GCP_REGION, GCP_INSTANCE
"""

import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from dotenv import load_dotenv
# Automatically load the credentials from the project's root .env file
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), '.env')
load_dotenv(dotenv_path)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config per environment
# ---------------------------------------------------------------------------

def _dev_url() -> str:
    """Local PostgreSQL connection string."""
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "postgres")
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ.get("DB_NAME", "savvio")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


def _prod_url() -> str:
    """
    GCP Cloud SQL connection string.
    Supports two modes:
    - TCP proxy (CI / local): DB_HOST=127.0.0.1
    - Unix socket (Cloud Run): DB_HOST=/cloudsql/project:region:instance
    """
    user = os.environ["DB_USER"]
    # Cloud Run Terraform sets DB_PASS; CI/local uses DB_PASSWORD.
    password = os.environ.get("DB_PASSWORD") or os.environ.get("DB_PASS", "")
    name = os.environ["DB_NAME"]

    proxy_host = os.environ.get("DB_HOST", "127.0.0.1")
    proxy_port = os.environ.get("DB_PORT", "5432")

    # Unix socket path (Cloud Run with Cloud SQL Auth Proxy volume mount)
    if proxy_host.startswith("/"):
        return (
            f"postgresql+psycopg2://{user}:{password}@/{name}"
            f"?host={proxy_host}"
        )

    return f"postgresql+psycopg2://{user}:{password}@{proxy_host}:{proxy_port}/{name}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ENV_MAP = {
    "dev": _dev_url,
    "prod": _prod_url,
}


def get_engine(env: str = "dev", echo: bool = False):
    """
    Create and return a SQLAlchemy engine.

    Args:
        env:  "dev" or "prod"
        echo: If True, log all SQL statements (useful for debugging).
    """
    if env not in ENV_MAP:
        raise ValueError(f"Unknown environment '{env}'. Choose from {list(ENV_MAP)}")

    url = ENV_MAP[env]()
    logger.info("Creating engine for env=%s", env)
    engine = create_engine(url, echo=echo, pool_pre_ping=True)
    return engine


def get_session(engine):
    """Return a new SQLAlchemy session bound to the given engine."""
    return sessionmaker(bind=engine)()


def ensure_pgvector(engine):
    """Enable the pgvector extension if not already present."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    logger.info("pgvector extension ensured")


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Test DB connection")
    parser.add_argument("--env", default="dev", choices=["dev", "prod"])
    args = parser.parse_args()

    engine = get_engine(args.env, echo=True)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print(f"Connection OK: {result.scalar()}")
