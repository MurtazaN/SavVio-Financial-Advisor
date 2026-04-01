"""
Tests for Bias Detection Runner — run_bias.py.
"""
import os
import sys
import importlib.util
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "src", "bias", "run_bias.py"),
        os.path.join(PROJECT_ROOT, "src", "bias", "run_bias.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        try:
            spec = importlib.util.spec_from_file_location("run_bias", fpath)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["run_bias"] = mod
            spec.loader.exec_module(mod)
            return mod
        except Exception as e:
            raise ImportError(f"Found {fpath} but failed to load: {e}")
    raise ImportError(f"Could not find run_bias.py. Searched:\n" + "\n".join(candidates))

M = _load()


# =============================================================================
# Airflow Task Tests
# =============================================================================

@patch("run_bias.run_financial_bias")
@patch("run_bias._setup", return_value=("/mock/proc", "/mock/feat"))
def test_bias_financial_task(mock_setup, mock_run):
    M.bias_financial_task()
    mock_run.assert_called_once_with(
        processed_path=os.path.join("/mock/proc", "financial_preprocessed.csv"),
        featured_path=os.path.join("/mock/feat", "financial_featured.csv")
    )


@patch("run_bias.run_product_bias")
@patch("run_bias._setup", return_value=("/mock/proc", "/mock/feat"))
def test_bias_product_task(mock_setup, mock_run):
    M.bias_product_task()
    mock_run.assert_called_once_with(
        preprocessed_path=os.path.join("/mock/proc", "product_preprocessed.jsonl"),
        featured_path=os.path.join("/mock/feat", "product_featured.jsonl")
    )


@patch("run_bias.run_review_bias")
@patch("run_bias._setup", return_value=("/mock/proc", "/mock/feat"))
def test_bias_review_task(mock_setup, mock_run):
    M.bias_review_task()
    mock_run.assert_called_once_with(
        preprocessed_path=os.path.join("/mock/proc", "review_preprocessed.jsonl"),
        featured_path=os.path.join("/mock/feat", "review_featured.jsonl")
    )


# =============================================================================
# Main script tests
# =============================================================================

@patch("sys.argv", ["run_bias.py", "--skip-product", "--skip-review"])
@patch("run_bias.run_financial_bias")
@patch("run_bias._setup", return_value=("/mock/proc", "/mock/feat"))
def test_main_skip_some(mock_setup, mock_finance):
    M.main()
    mock_finance.assert_called_once()


@patch("sys.argv", ["run_bias.py"])
@patch("run_bias.run_review_bias", side_effect=Exception("Review error"))
@patch("run_bias.run_product_bias")
@patch("run_bias.run_financial_bias")
@patch("run_bias._setup", return_value=("/mock/proc", "/mock/feat"))
def test_main_failure(mock_setup, mock_finance, mock_product, mock_review):
    with pytest.raises(SystemExit) as exc_info:
        M.main()
    
    # Exits with 1 due to exception in review bias
    assert exc_info.value.code == 1
    mock_finance.assert_called_once()
    mock_product.assert_called_once()
    mock_review.assert_called_once()
