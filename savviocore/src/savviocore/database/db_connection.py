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
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    host = os.environ["DB_HOST"]
    port = os.environ["DB_PORT"]
    name = os.environ["DB_NAME"]
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


def _prod_url() -> str:
    """
    GCP Cloud SQL connection string.
    Expects Cloud SQL Auth Proxy running locally or a Unix socket.
    """
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    name = os.environ["DB_NAME"]

    # Option A: Cloud SQL Auth Proxy on localhost (default)
    proxy_host = os.environ.get("DB_HOST", "127.0.0.1")
    proxy_port = os.environ.get("DB_PORT", "5432")

    # Option B: Unix socket (uncomment if using socket-based proxy)
    # project = os.environ["GCP_PROJECT"]
    # region  = os.environ["GCP_REGION"]
    # instance = os.environ["GCP_INSTANCE"]
    # socket = f"/cloudsql/{project}:{region}:{instance}"
    # return (
    #     f"postgresql+psycopg2://{user}:{password}@/{name}"
    #     f"?host={socket}"
    # )

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
