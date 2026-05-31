"""
SavVio Inference API — Production FastAPI entry point.

Endpoints:
    /health   - Service status and resource readiness
    /products - Paginated product catalog search
    /predict  - ML-led purchase recommendation pipeline
    /user     - Financial profile retrieval
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from deployment_pipeline.api.config import APIConfig
from deployment_pipeline.api.dependencies import get_model_manager, require_db
from deployment_pipeline.api.inference import run_inference, _load_user_financial_profile
from deployment_pipeline.api.metrics import setup_metrics, track_active_request, release_active_request
from deployment_pipeline.api.model_loader import ModelManager
from deployment_pipeline.api.products_catalog import list_products as fetch_products
from deployment_pipeline.api.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ProductListItem,
    ProductListResponse,
    UserProfileResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, APIConfig.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Resource initialization and cleanup."""
    logger.info("Initializing API v%s", APIConfig.API_VERSION)
    manager = get_model_manager()
    try:
        manager.load()
        logger.info("All resources loaded successfully.")
    except Exception as e:
        logger.error("Startup resource loading failed: %s", e, exc_info=True)
        logger.warning("API will start but some features may be unavailable.")
    yield
    logger.info("Shutting down SavVio API...")


# ---------------------------------------------------------------------------
# App Factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title=APIConfig.API_TITLE,
    version=APIConfig.API_VERSION,
    description=APIConfig.API_DESCRIPTION,
    lifespan=lifespan,
)

# CORS — allow Vite frontend and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=APIConfig.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Prometheus instrumentation
setup_metrics(app)


# ---------------------------------------------------------------------------
# Request Logging Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """HTTP request logging with latency tracking."""
    is_metrics_scrape = request.url.path == "/metrics"
    if not is_metrics_scrape:
        track_active_request()
    start = time.time()
    try:
        response = await call_next(request)
        elapsed = time.time() - start
        logger.info(
            "request_log | method=%s path=%s status=%d latency=%.3fs",
            request.method, request.url.path,
            response.status_code, elapsed,
        )
        return response
    finally:
        if not is_metrics_scrape:
            release_active_request()


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"http_{exc.status_code}",
            detail=str(exc.detail),
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Normalize 422 bodies to the same `ErrorResponse` shape as other errors."""
    parts: list[str] = []
    for err in exc.errors():
        loc = " → ".join(str(x) for x in err.get("loc", ()) if x != "body")
        msg = err.get("msg", "invalid")
        if loc:
            parts.append(f"{loc}: {msg}")
        else:
            parts.append(str(msg))
    detail = "; ".join(parts) if parts else "Invalid request body."
    logger.warning("validation_error | path=%s detail=%s", request.url.path, detail)
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="validation_error", detail=detail).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            detail="An unexpected error occurred. Please try again.",
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/user/{user_id}/profile",
    response_model=UserProfileResponse,
    tags=["User"],
)
async def get_user_profile(
    user_id: str,
    db_engine=Depends(require_db),
    manager: ModelManager = Depends(get_model_manager),
):
    """Return the user's financial profile from PostgreSQL."""
    row = _load_user_financial_profile(user_id, db_engine)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No financial profile found for user_id={user_id!r}.",
        )
    return UserProfileResponse(**row)


@app.get(
    "/products",
    response_model=ProductListResponse,
    tags=["Catalog"],
)
async def browse_products(
    q: Optional[str] = Query(
        None,
        description="Substring filter on product_name (ILIKE).",
    ),
    price_min: Optional[float] = Query(
        None,
        description="Minimum price.",
    ),
    price_max: Optional[float] = Query(
        None,
        description="Maximum price.",
    ),
    limit: int = Query(
        APIConfig.PRODUCT_BROWSE_DEFAULT_LIMIT,
        ge=1,
        le=APIConfig.PRODUCT_BROWSE_MAX_LIMIT,
    ),
    offset: int = Query(0, ge=0),
    db_engine=Depends(require_db),
):
    """List products for catalog selection."""
    lim = min(limit, APIConfig.PRODUCT_BROWSE_MAX_LIMIT)
    rows, total, pmin_applied, pmax_applied = fetch_products(
        db_engine,
        q=q,
        price_min=price_min,
        price_max=price_max,
        limit=lim,
        offset=offset,
        default_price_min=APIConfig.PRODUCT_BROWSE_PRICE_MIN,
        default_price_max=APIConfig.PRODUCT_BROWSE_PRICE_MAX,
    )
    return ProductListResponse(
        items=[ProductListItem(**r) for r in rows],
        total=total,
        price_min_applied=pmin_applied,
        price_max_applied=pmax_applied,
        limit=lim,
        offset=offset,
    )


@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(
    manager: ModelManager = Depends(get_model_manager),
):
    """Liveness check — model loaded, DB connected, LLM provider active."""
    return HealthResponse(
        status="healthy" if manager.is_loaded else "degraded",
        model_loaded=manager.model is not None,
        db_connected=manager.check_db_connection(),
        llm_provider=manager.get_llm_provider_name(),
        version=APIConfig.API_VERSION,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(
    request: PredictRequest,
    manager: ModelManager = Depends(get_model_manager),
):
    """Orchestrate ML scoring, rule-based logic, and LLM explanation generation."""
    try:
        return run_inference(request, manager)
    except Exception as e:
        logger.error(
            "Inference failed for user=%s: %s",
            request.user_id, e, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Inference pipeline failed. Please try again.",
        )


@app.get(
    "/user/{user_id}/evaluate",
    response_model=PredictResponse,
    tags=["Inference"],
)
async def evaluate_product(
    user_id: str,
    product_id: str = Query(
        ..., description="Product ID to evaluate.",
    ),
    manager: ModelManager = Depends(get_model_manager),
):
    """Direct product evaluation — skips LLM intent parsing."""
    request = PredictRequest(
        user_query=f"Evaluate product {product_id}",
        user_id=user_id,
        product_id=product_id,
    )
    try:
        return run_inference(request, manager)
    except Exception as e:
        logger.error(
            "Evaluation failed for user=%s, product=%s: %s",
            user_id, product_id, e, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Product evaluation failed. Please try again.",
        )


# ---------------------------------------------------------------------------
# Run with: uvicorn deployment_pipeline.api.main:app --host 0.0.0.0 --port 3500
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "deployment_pipeline.api.main:app",
        host=APIConfig.API_HOST,
        port=APIConfig.API_PORT,
        reload=True,
    )
