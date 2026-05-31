"""
API Endpoint Tests for SavVio Deployment.

Tests the /health, /predict, and /evaluate endpoints via FastAPI TestClient
with mocked backend resources (no real DB or model required).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest  # noqa: F401 — fixtures used implicitly

# Ensure model_pipeline/src is on the path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
_model_src = _project_root / "model_pipeline" / "src"
_savviocore_src = _project_root / "savviocore" / "src"
for p in [str(_project_root), str(_model_src), str(_savviocore_src)]:
    if p not in sys.path:
        sys.path.insert(0, p)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_response_fields(self, test_client):
        response = test_client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "db_connected" in data
        assert "llm_provider" in data
        assert "version" in data

    def test_health_reports_mock_provider(self, test_client):
        response = test_client.get("/health")
        data = response.json()
        assert data["llm_provider"] == "mock"

    def test_health_reports_version(self, test_client):
        response = test_client.get("/health")
        data = response.json()
        assert data["version"] == "1.0.0"


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_requires_user_query(self, test_client):
        """Missing user_query -> 422 validation error."""
        response = test_client.post(
            "/predict", json={"user_id": "user_001"},
        )
        assert response.status_code == 422

    def test_predict_requires_user_id(self, test_client):
        """Missing user_id -> 422 validation error."""
        response = test_client.post(
            "/predict", json={"user_query": "Can I buy AirPods?"},
        )
        assert response.status_code == 422

    def test_predict_empty_query_rejected(self, test_client):
        """Empty user_query -> 422 validation error (min_length=1)."""
        response = test_client.post("/predict", json={
            "user_query": "",
            "user_id": "user_001",
        })
        assert response.status_code == 422

    def test_predict_returns_valid_response_schema(
        self, test_client, green_user_profile, sample_product,
    ):
        """Valid request -> response has all required fields."""
        with patch("deployment_pipeline.api.main.run_inference") as mock_run:
            from deployment_pipeline.api.schemas import PredictResponse
            mock_run.return_value = PredictResponse(
                recommendation="GREEN",
                confidence=0.92,
                explanation="This purchase fits well within your budget.",
                product_name="Sony WH-1000XM5",
                product_price=349.99,
            )
            response = test_client.post("/predict", json={
                "user_query": "Can I buy the Sony WH-1000XM5?",
                "user_id": "test_green_user",
            })
            assert response.status_code == 200
            data = response.json()
            assert "recommendation" in data
            assert "confidence" in data
            assert "explanation" in data
            assert "product_name" in data
            assert "product_price" in data

    def test_predict_recommendation_is_valid_color(self, test_client):
        """Recommendation must be GREEN, YELLOW, or RED."""
        with patch("deployment_pipeline.api.main.run_inference") as mock_run:
            from deployment_pipeline.api.schemas import PredictResponse
            for color in ["GREEN", "YELLOW", "RED"]:
                mock_run.return_value = PredictResponse(
                    recommendation=color,
                    confidence=0.85,
                    explanation=f"Test {color} response.",
                )
                response = test_client.post("/predict", json={
                    "user_query": "Can I buy test product?",
                    "user_id": "test_user",
                })
                assert response.status_code == 200
                assert response.json()["recommendation"] == color

    def test_predict_confidence_in_range(self, test_client):
        """Confidence score must be between 0 and 1."""
        with patch("deployment_pipeline.api.main.run_inference") as mock_run:
            from deployment_pipeline.api.schemas import PredictResponse
            mock_run.return_value = PredictResponse(
                recommendation="GREEN",
                confidence=0.87,
                explanation="Good to buy.",
            )
            response = test_client.post("/predict", json={
                "user_query": "Can I buy something?",
                "user_id": "test_user",
            })
            data = response.json()
            assert 0.0 <= data["confidence"] <= 1.0


class TestEvaluateEndpoint:
    """Tests for GET /user/{user_id}/evaluate."""

    def test_evaluate_requires_product_id(self, test_client):
        """Missing product_id query param -> 422."""
        response = test_client.get("/user/test_user/evaluate")
        assert response.status_code == 422

    def test_evaluate_returns_valid_response(self, test_client):
        """Valid request with product_id -> 200."""
        with patch("deployment_pipeline.api.main.run_inference") as mock_run:
            from deployment_pipeline.api.schemas import PredictResponse
            mock_run.return_value = PredictResponse(
                recommendation="GREEN",
                confidence=0.90,
                explanation="Looks good!",
                product_name="Test Product",
                product_price=99.99,
            )
            response = test_client.get(
                "/user/test_user/evaluate",
                params={"product_id": "B0TESTPROD"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["recommendation"] == "GREEN"
