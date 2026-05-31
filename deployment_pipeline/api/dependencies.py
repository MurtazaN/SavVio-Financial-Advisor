"""API Dependencies — Dependency injection for ModelManager and Database engine."""

from __future__ import annotations

from fastapi import Depends, HTTPException

from deployment_pipeline.api.model_loader import ModelManager, model_manager


def get_model_manager() -> ModelManager:
    """Provide the ModelManager singleton."""
    return model_manager


def require_db(
    manager: ModelManager = Depends(get_model_manager),
):
    """Ensure database is available; raise 503 if not.

    Returns the SQLAlchemy engine for direct use.
    """
    if not manager.db_engine:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable.",
        )
    return manager.db_engine
