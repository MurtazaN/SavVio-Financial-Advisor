# model_pipeline/tests/test_validation.py
import os
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# ---------------------------------------------------------------------------
# Stub external dependencies
# ---------------------------------------------------------------------------
def _stub_modules():
    # great_expectations
    gx = types.ModuleType("great_expectations")
    sys.modules.setdefault("great_expectations", gx)

    # savviocore
    savviocore = types.ModuleType("savviocore")
    savviocore.validation = types.ModuleType("savviocore.validation")

    # Mock ValidationReport result
    mock_result = MagicMock()
    mock_result.passed = True
    mock_result.results = []

    mock_report = MagicMock()
    mock_report.passed = True
    mock_report.results = []

    savviocore.validation.feature_validator = types.ModuleType("savviocore.validation.feature_validator")
    savviocore.validation.feature_validator.validate_financial_features = MagicMock(return_value=mock_report)
    savviocore.validation.feature_validator.validate_review_features = MagicMock(return_value=mock_report)

    savviocore.validation.validation_config = types.ModuleType("savviocore.validation.validation_config")
    savviocore.validation.validation_config.load_thresholds = MagicMock()
    savviocore.validation.validation_config.ValidationReport = MagicMock()

    sys.modules.setdefault("savviocore", savviocore)
    sys.modules.setdefault("savviocore.validation", savviocore.validation)
    sys.modules.setdefault("savviocore.validation.feature_validator", savviocore.validation.feature_validator)
    sys.modules.setdefault("savviocore.validation.validation_config", savviocore.validation.validation_config)

_stub_modules()

from data.validate_data import (
    validate_financial_data,
    validate_products,
    validate_target_distribution,
    DataValidationError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_financial_df(n=10):
    return pd.DataFrame({
        "discretionary_income":        [2500.0] * n,
        "debt_to_income_ratio":         [0.2]   * n,
        "saving_to_income_ratio":       [0.3]   * n,
        "monthly_expense_burden_ratio": [0.5]   * n,
        "emergency_fund_months":        [6.0]   * n,
    })

def _make_products_df(n=10):
    return pd.DataFrame({
        "product_id":     [f"p{i}" for i in range(n)],
        "price":          [100.0] * n,
        "average_rating": [4.0]   * n,
        "rating_variance":[0.5]   * n,
    })

def _make_failing_report():
    result = MagicMock()
    result.passed = False
    result.check_name = "test_check"
    result.details = "Check failed"
    result.severity = MagicMock()
    result.severity.name = "CRITICAL"
    report = MagicMock()
    report.passed = False
    report.results = [result]
    return report


# ---------------------------------------------------------------------------
# Tests: validate_financial_data
# ---------------------------------------------------------------------------
class TestValidateFinancialData:

    def test_valid_data_returns_dict(self):
        df = _make_financial_df()
        result = validate_financial_data(df, raise_on_error=False)
        assert isinstance(result, dict)

    def test_valid_data_has_correct_keys(self):
        df = _make_financial_df()
        result = validate_financial_data(df, raise_on_error=False)
        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result
        assert "summary" in result

    def test_valid_data_passes(self):
        df = _make_financial_df()
        result = validate_financial_data(df, raise_on_error=False)
        assert result["valid"] is True

    def test_summary_contains_rows(self):
        df = _make_financial_df(n=15)
        result = validate_financial_data(df, raise_on_error=False)
        assert result["summary"]["rows"] == 15

    def test_failing_validation_raises_when_flag_set(self):
        from savviocore.validation.feature_validator import validate_financial_features
        validate_financial_features.return_value = _make_failing_report()
        df = _make_financial_df()
        with pytest.raises(DataValidationError):
            validate_financial_data(df, raise_on_error=True)
        # Reset mock
        mock_report = MagicMock()
        mock_report.passed = True
        mock_report.results = []
        validate_financial_features.return_value = mock_report

    def test_failing_validation_no_raise_when_flag_false(self):
        from savviocore.validation.feature_validator import validate_financial_features
        validate_financial_features.return_value = _make_failing_report()
        df = _make_financial_df()
        result = validate_financial_data(df, raise_on_error=False)
        assert result["valid"] is False
        # Reset mock
        mock_report = MagicMock()
        mock_report.passed = True
        mock_report.results = []
        validate_financial_features.return_value = mock_report

    def test_errors_list_empty_on_success(self):
        df = _make_financial_df()
        result = validate_financial_data(df, raise_on_error=False)
        assert result["errors"] == []


# ---------------------------------------------------------------------------
# Tests: validate_products
# ---------------------------------------------------------------------------
class TestValidateProducts:

    def test_returns_dict(self):
        df = _make_products_df()
        result = validate_products(df, raise_on_error=False)
        assert isinstance(result, dict)

    def test_valid_data_passes(self):
        df = _make_products_df()
        result = validate_products(df, raise_on_error=False)
        assert result["valid"] is True

    def test_summary_contains_rows(self):
        df = _make_products_df(n=20)
        result = validate_products(df, raise_on_error=False)
        assert result["summary"]["rows"] == 20


# ---------------------------------------------------------------------------
# Tests: validate_target_distribution
# ---------------------------------------------------------------------------
class TestValidateTargetDistribution:

    def test_balanced_target_passes(self):
        y = pd.Series([0, 1, 2] * 20)
        result = validate_target_distribution(y, raise_on_error=False)
        assert result["valid"] is True

    def test_returns_distribution_dict(self):
        y = pd.Series([0, 1, 2] * 20)
        result = validate_target_distribution(y, raise_on_error=False)
        assert "distribution" in result
        assert isinstance(result["distribution"], dict)

    def test_single_class_fails(self):
        y = pd.Series([0] * 20)
        result = validate_target_distribution(y, raise_on_error=False)
        assert result["valid"] is False

    def test_single_class_raises_when_flag_set(self):
        y = pd.Series([0] * 20)
        with pytest.raises(DataValidationError):
            validate_target_distribution(y, raise_on_error=True)

    def test_severe_imbalance_fails(self):
        # 99% class 0, 1% class 1 — below default 5% threshold
        y = pd.Series([0] * 99 + [1] * 1)
        result = validate_target_distribution(y, raise_on_error=False)
        assert result["valid"] is False

    def test_errors_populated_on_failure(self):
        y = pd.Series([0] * 20)
        result = validate_target_distribution(y, raise_on_error=False)
        assert len(result["errors"]) > 0

    def test_custom_threshold(self):
        # 10% minority — passes with default 5% but fails with 15%
        y = pd.Series([0] * 90 + [1] * 10)
        result_pass = validate_target_distribution(y, min_minority_pct=5.0, raise_on_error=False)
        result_fail = validate_target_distribution(y, min_minority_pct=15.0, raise_on_error=False)
        assert result_pass["valid"] is True
        assert result_fail["valid"] is False