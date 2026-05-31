"""
Tests for Prometheus Metrics Module.

Verifies:
    - /metrics endpoint is exposed and returns Prometheus format
    - Custom SavVio counters/histograms are registered
    - Recording helpers correctly increment metrics
    - METRICS_ENABLED=false disables instrumentation
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure deployment package is importable
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_model_src = _project_root / "model_pipeline" / "src"
_savviocore_src = _project_root / "savviocore" / "src"
for p in [_model_src, _savviocore_src]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture
def metrics_client():
    """TestClient with Prometheus metrics explicitly attached."""
    from unittest.mock import MagicMock, patch

    manager = MagicMock()
    manager.is_loaded = True
    manager.model = MagicMock()
    manager.label_encoder = MagicMock()
    manager.db_engine = MagicMock()
    manager.llm_provider = MagicMock()
    manager.llm_provider.provider_name = "mock"
    manager.check_db_connection.return_value = True
    manager.get_llm_provider_name.return_value = "mock"

    from deployment_pipeline.api.main import app
    from deployment_pipeline.api.metrics import setup_metrics
    from deployment_pipeline.api.dependencies import get_model_manager
    from fastapi.testclient import TestClient

    app.dependency_overrides[get_model_manager] = lambda: manager

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


class TestMetricsEndpoint:
    """Tests for the /metrics Prometheus endpoint."""

    def test_metrics_endpoint_returns_200(self, metrics_client):
        """GET /metrics should return HTTP 200."""
        response = metrics_client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_returns_prometheus_format(self, metrics_client):
        """Response should contain Prometheus text exposition format."""
        response = metrics_client.get("/metrics")
        content = response.text
        # Prometheus format starts with metrics or comments
        assert "# HELP" in content or "# TYPE" in content or "python_info" in content

    def test_metrics_contains_custom_savvio_metrics(self, metrics_client):
        """Custom SavVio metrics should be registered in the output."""
        response = metrics_client.get("/metrics")
        content = response.text
        # These metrics are created at module import time, so they should exist
        # even before any inference runs
        assert "savvio_inference_total" in content or "savvio_inference_duration_seconds" in content

    def test_metrics_contains_active_requests_gauge(self, metrics_client):
        """Active requests gauge should be present."""
        response = metrics_client.get("/metrics")
        content = response.text
        assert "savvio_active_requests" in content


class TestMetricsRecording:
    """Tests for the custom metric recording helper functions."""

    def test_record_inference_increments_counter(self):
        """record_inference should increment the inference total counter."""
        from deployment_pipeline.api.metrics import INFERENCE_TOTAL, record_inference

        # Get the current value
        before = INFERENCE_TOTAL.labels(
            recommendation="GREEN", evaluation_mode="catalog"
        )._value.get()

        record_inference(color="GREEN", evaluation_mode="catalog", latency=0.5)

        after = INFERENCE_TOTAL.labels(
            recommendation="GREEN", evaluation_mode="catalog"
        )._value.get()

        assert after == before + 1

    def test_record_inference_observes_histogram(self):
        """record_inference should observe the duration histogram."""
        from deployment_pipeline.api.metrics import INFERENCE_DURATION, record_inference

        before_count = INFERENCE_DURATION.labels(
            recommendation="YELLOW", evaluation_mode="hypothetical"
        )._sum.get()

        record_inference(color="YELLOW", evaluation_mode="hypothetical", latency=1.23)

        after_count = INFERENCE_DURATION.labels(
            recommendation="YELLOW", evaluation_mode="hypothetical"
        )._sum.get()

        assert after_count > before_count

    def test_record_guardrail_failure(self):
        """record_guardrail_failure should increment the failure counter."""
        from deployment_pipeline.api.metrics import GUARDRAIL_FAILURES, record_guardrail_failure

        before = GUARDRAIL_FAILURES._value.get()
        record_guardrail_failure()
        after = GUARDRAIL_FAILURES._value.get()

        assert after == before + 1

    def test_record_layer2_downgrade(self):
        """record_layer2_downgrade should increment with correct labels."""
        from deployment_pipeline.api.metrics import LAYER2_DOWNGRADES, record_layer2_downgrade

        before = LAYER2_DOWNGRADES.labels(
            from_color="GREEN", to_color="YELLOW"
        )._value.get()

        record_layer2_downgrade(from_color="GREEN", to_color="YELLOW")

        after = LAYER2_DOWNGRADES.labels(
            from_color="GREEN", to_color="YELLOW"
        )._value.get()

        assert after == before + 1

    def test_observe_ml_confidence(self):
        """observe_ml_confidence should record to the histogram."""
        from deployment_pipeline.api.metrics import ML_CONFIDENCE, observe_ml_confidence

        before_count = ML_CONFIDENCE._sum.get()
        observe_ml_confidence(0.87)
        after_count = ML_CONFIDENCE._sum.get()

        assert after_count > before_count

    def test_record_inference_all_colors(self):
        """All three recommendation colors should be recordable."""
        from deployment_pipeline.api.metrics import INFERENCE_TOTAL, record_inference

        for color in ["GREEN", "YELLOW", "RED"]:
            record_inference(color=color, evaluation_mode="catalog", latency=0.1)
            val = INFERENCE_TOTAL.labels(
                recommendation=color, evaluation_mode="catalog"
            )._value.get()
            assert val >= 1, f"Counter for {color} should be >= 1"


class TestMetricsDisabled:
    """Tests for METRICS_ENABLED=false behavior."""

    def test_record_inference_noop_when_disabled(self):
        """When METRICS_ENABLED=false, recording helpers should be no-ops."""
        from deployment_pipeline.api.metrics import INFERENCE_TOTAL

        with patch("deployment_pipeline.api.metrics.APIConfig") as mock_config:
            mock_config.METRICS_ENABLED = False

            from deployment_pipeline.api.metrics import record_inference

            before = INFERENCE_TOTAL.labels(
                recommendation="RED", evaluation_mode="catalog"
            )._value.get()

            record_inference(color="RED", evaluation_mode="catalog", latency=0.5)

            after = INFERENCE_TOTAL.labels(
                recommendation="RED", evaluation_mode="catalog"
            )._value.get()

            # Should NOT have incremented (disabled)
            assert after == before
