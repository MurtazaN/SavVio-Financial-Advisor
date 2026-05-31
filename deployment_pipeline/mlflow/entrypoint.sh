#!/bin/bash
set -e

# URL-encode the password so special characters (], {, ), etc.) don't break
# Python's urlparse when MLflow parses the backend store URI.
DB_PASS_ENCODED=$(python3 -c "import urllib.parse, os; print(urllib.parse.quote(os.environ['DB_PASS'], safe=''))")

# DB_HOST is the Cloud SQL Unix socket dir: /cloudsql/PROJECT:REGION:INSTANCE
# pool_size=2 max_overflow=0: hard cap of 2 connections per instance so db-f1-micro
# (max_connections≈25) is never exhausted during rolling deploys.
# pool_size / max_overflow / pool_pre_ping are SQLAlchemy engine args — they
# cannot appear as DSN query params (psycopg2 rejects them). Pass them via the
# SQLAlchemy-specific create_engine_kwargs query string instead.
BACKEND_URI="postgresql://${DB_USER}:${DB_PASS_ENCODED}@/${DB_NAME}?host=${DB_HOST}"

exec mlflow server \
  --backend-store-uri "${BACKEND_URI}" \
  --artifacts-destination "${MLFLOW_ARTIFACT_ROOT}" \
  --host 0.0.0.0 \
  --port "${PORT:-5000}" \
  --serve-artifacts \
  --workers 1
