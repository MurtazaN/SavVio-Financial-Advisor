"""
Shared test fixtures for SavVio deployment tests.

Provides:
    - Mock ModelManager with pre-configured test data
    - FastAPI TestClient (uses app.dependency_overrides for clean DI)
    - Sample financial profiles for GREEN / YELLOW / RED scenarios
    - Sample products and reviews
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure deployment package is importable
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Also ensure model_pipeline/src and savviocore/src are importable
_model_src = _project_root / "model_pipeline" / "src"
_savviocore_src = _project_root / "savviocore" / "src"
for p in [_model_src, _savviocore_src]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ---------------------------------------------------------------------------
# Sample Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def green_user_profile():
    """A financially healthy user — should trigger GREEN."""
    return {
        "user_id": "test_green_user",
        "monthly_income": 8000.0,
        "monthly_expenses": 3000.0,
        "savings_balance": 25000.0,
        "has_loan": 0,
        "loan_amount": 0.0,
        "monthly_emi": 0.0,
        "loan_interest_rate": 0.0,
        "loan_term_months": 0.0,
        "credit_score": 780,
        "employment_status": "employed",
        "region": "northeast",
        "liquid_savings": 25000.0,
        "discretionary_income": 5000.0,
        "debt_to_income_ratio": 0.0,
        "saving_to_income_ratio": 3.125,
        "monthly_expense_burden_ratio": 0.375,
        "emergency_fund_months": 8.33,
    }


@pytest.fixture
def yellow_user_profile():
    """A moderately strained user — should trigger YELLOW."""
    return {
        "user_id": "test_yellow_user",
        "monthly_income": 4000.0,
        "monthly_expenses": 2800.0,
        "savings_balance": 3000.0,
        "has_loan": 1,
        "loan_amount": 5000.0,
        "monthly_emi": 200.0,
        "loan_interest_rate": 5.0,
        "loan_term_months": 36.0,
        "credit_score": 620,
        "employment_status": "employed",
        "region": "southeast",
        "liquid_savings": 3000.0,
        "discretionary_income": 1000.0,
        "debt_to_income_ratio": 0.05,
        "saving_to_income_ratio": 0.75,
        "monthly_expense_burden_ratio": 0.75,
        "emergency_fund_months": 1.0,
    }


@pytest.fixture
def red_user_profile():
    """A financially distressed user — should trigger RED."""
    return {
        "user_id": "test_red_user",
        "monthly_income": 2500.0,
        "monthly_expenses": 2200.0,
        "savings_balance": 500.0,
        "has_loan": 1,
        "loan_amount": 15000.0,
        "monthly_emi": 800.0,
        "loan_interest_rate": 12.0,
        "loan_term_months": 48.0,
        "credit_score": 520,
        "employment_status": "part_time",
        "region": "west",
        "liquid_savings": 500.0,
        "discretionary_income": -500.0,
        "debt_to_income_ratio": 0.32,
        "saving_to_income_ratio": 0.2,
        "monthly_expense_burden_ratio": 0.88,
        "emergency_fund_months": 0.17,
    }


@pytest.fixture
def sample_product():
    """A mid-range product for testing."""
    return {
        "product_id": "B09V3KXJPB",
        "product_name": "Sony WH-1000XM5",
        "price": 349.99,
        "average_rating": 4.5,
        "rating_number": 5000,
        "rating_variance": 0.6,
        "category": "Electronics",
    }


@pytest.fixture
def cheap_product():
    """A cheap product (low PIR) — should not trigger RED."""
    return {
        "product_id": "B0CHEAP001",
        "product_name": "USB Cable",
        "price": 5.99,
        "average_rating": 4.2,
        "rating_number": 12000,
        "rating_variance": 0.3,
        "category": "Electronics",
    }


# ---------------------------------------------------------------------------
# Mock ModelManager
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model_manager():
    """A mock ModelManager that doesn't require real model artifacts or DB."""
    manager = MagicMock()
    manager.is_loaded = True
    manager.model = MagicMock()
    manager.label_encoder = MagicMock()
    manager.db_engine = MagicMock()
    manager.llm_provider = MagicMock()
    manager.llm_provider.provider_name = "mock"
    manager.category_stats = {
        "Electronics": {
            "min_price": 5.0,
            "max_price": 2000.0,
            "mean_rating": 4.0,
        },
    }
    manager.max_rating_number = 50000.0
    manager.check_db_connection.return_value = True
    manager.get_llm_provider_name.return_value = "mock"
    return manager


# ---------------------------------------------------------------------------
# FastAPI TestClient — uses dependency overrides (no monkeypatching)
# ---------------------------------------------------------------------------

@pytest.fixture
def test_client(mock_model_manager):
    """FastAPI TestClient with DI-overridden ModelManager."""
    from fastapi.testclient import TestClient

    from deployment_pipeline.api.dependencies import get_model_manager
    from deployment_pipeline.api.main import app

    app.dependency_overrides[get_model_manager] = (
        lambda: mock_model_manager
    )
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()
