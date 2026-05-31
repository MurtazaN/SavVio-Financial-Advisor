"""
Prometheus Metrics — HTTP and Business Logic instrumentation.
Custom metrics: Latency histograms, Inference counters, Guardrail alerts, and Active requests.
"""

from __future__ import annotations

import logging
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram

from deployment_pipeline.api.config import APIConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric prefix — all custom metrics start with this
# ---------------------------------------------------------------------------
_PREFIX = APIConfig.METRICS_PREFIX

# ---------------------------------------------------------------------------
# Custom Business Metrics
# ---------------------------------------------------------------------------

# Full inference pipeline latency (not just HTTP round-trip)
INFERENCE_DURATION = Histogram(
    f"{_PREFIX}_inference_duration_seconds",
    "End-to-end inference pipeline latency in seconds",
    labelnames=["recommendation", "evaluation_mode"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# Inference request counter by recommendation color and evaluation mode
INFERENCE_TOTAL = Counter(
    f"{_PREFIX}_inference_total",
    "Total inference requests by recommendation color and evaluation mode",
    labelnames=["recommendation", "evaluation_mode"],
)

# Guardrail failure counter
GUARDRAIL_FAILURES = Counter(
    f"{_PREFIX}_guardrail_failures_total",
    "Total LLM responses that failed guardrail checks",
)

# Layer 2 downgrade counter
LAYER2_DOWNGRADES = Counter(
    f"{_PREFIX}_layer2_downgrades_total",
    "Total Layer 2 downgrades (product/review quality signals)",
    labelnames=["from_color", "to_color"],
)

# ML model confidence distribution
ML_CONFIDENCE = Histogram(
    f"{_PREFIX}_ml_confidence",
    "ML model confidence score distribution",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

# In-flight request gauge
ACTIVE_REQUESTS = Gauge(
    f"{_PREFIX}_active_requests",
    "Number of currently in-flight requests",
)

# LLM provider latency
LLM_LATENCY = Histogram(
    f"{_PREFIX}_llm_latency_seconds",
    "LLM provider response time in seconds",
    labelnames=["provider", "operation"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)


# ---------------------------------------------------------------------------
# Setup — attach instrumentator to FastAPI app
# ---------------------------------------------------------------------------

def setup_metrics(app) -> None:
    """Configures Prometheus FastAPI Instrumentator and exposes /metrics endpoint."""
    if not APIConfig.METRICS_ENABLED:
        logger.info("Prometheus metrics disabled (METRICS_ENABLED=false)")
        return

    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        instrumentator = Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_respect_env_var=False,
            excluded_handlers=["/health", "/docs", "/openapi.json"],
            env_var_name="ENABLE_METRICS",
            inprogress_name=f"{_PREFIX}_http_requests_inprogress",
            inprogress_labels=True,
        )

        instrumentator.instrument(app).expose(
            app,
            endpoint="/metrics",
            include_in_schema=True,
            tags=["Monitoring"],
        )

        logger.info("Prometheus instrumentator attached to FastAPI app")

    except ImportError:
        logger.warning(
            "prometheus-fastapi-instrumentator not installed — "
            "HTTP metrics will not be available. "
            "Install with: pip install prometheus-fastapi-instrumentator"
        )
    except Exception as e:
        logger.error("Failed to setup Prometheus metrics: %s", e, exc_info=True)


# ---------------------------------------------------------------------------
# Recording Helpers — called from inference.py
# ---------------------------------------------------------------------------

def record_inference(
    color: str,
    evaluation_mode: str,
    latency: float,
) -> None:
    """Updates inference counters and duration histograms."""
    if not APIConfig.METRICS_ENABLED:
        return

    INFERENCE_TOTAL.labels(
        recommendation=color,
        evaluation_mode=evaluation_mode,
    ).inc()

    INFERENCE_DURATION.labels(
        recommendation=color,
        evaluation_mode=evaluation_mode,
    ).observe(latency)


def record_guardrail_failure() -> None:
    """Record a guardrail check failure."""
    if not APIConfig.METRICS_ENABLED:
        return
    GUARDRAIL_FAILURES.inc()


def record_layer2_downgrade(from_color: str, to_color: str) -> None:
    """Record a Layer 2 downgrade event.

    Args:
        from_color: Original color before downgrade (e.g. GREEN).
        to_color: Final color after downgrade (e.g. YELLOW).
    """
    if not APIConfig.METRICS_ENABLED:
        return
    LAYER2_DOWNGRADES.labels(from_color=from_color, to_color=to_color).inc()


def observe_ml_confidence(confidence: float) -> None:
    """Record an ML model confidence score observation.

    Args:
        confidence: Confidence score between 0 and 1.
    """
    if not APIConfig.METRICS_ENABLED:
        return
    ML_CONFIDENCE.observe(confidence)


def observe_llm_latency(
    provider: str,
    operation: str,
    latency: float,
) -> None:
    """Record LLM provider response latency.

    Args:
        provider: Provider name (e.g. "groq", "openai", "mock").
        operation: Operation type (e.g. "intent_parse", "response_gen").
        latency: Response time in seconds.
    """
    if not APIConfig.METRICS_ENABLED:
        return
    LLM_LATENCY.labels(provider=provider, operation=operation).observe(latency)


def track_active_request():
    """Context-manager-style tracker for in-flight requests.

    Usage:
        ACTIVE_REQUESTS.inc()
        try:
            ...
        finally:
            ACTIVE_REQUESTS.dec()
    """
    ACTIVE_REQUESTS.inc()


def release_active_request():
    """Decrement the active request gauge."""
    ACTIVE_REQUESTS.dec()


# ---------------------------------------------------------------------------
# Pre-initialize labeled metrics so time series exist from the first scrape.
# Without this, labeled counters/histograms only appear in Prometheus after
# the first request with those label values — causing "No data" in Grafana
# after every container restart (deployment).
# ---------------------------------------------------------------------------

_COLORS = ["GREEN", "YELLOW", "RED"]
_EVAL_MODES = ["catalog", "hypothetical", "none"]

for _color in _COLORS:
    for _mode in _EVAL_MODES:
        INFERENCE_TOTAL.labels(recommendation=_color, evaluation_mode=_mode)
        INFERENCE_DURATION.labels(recommendation=_color, evaluation_mode=_mode)

for _from_color in _COLORS:
    for _to_color in _COLORS:
        if _from_color != _to_color:
            LAYER2_DOWNGRADES.labels(from_color=_from_color, to_color=_to_color)
