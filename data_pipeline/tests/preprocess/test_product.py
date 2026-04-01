"""
Tests for Data Preprocessing — preprocess/product.py.

Covers product data cleaning: text normalisation, category standardisation,
title-based category grouping, deduplication, and the full preprocessing
pipeline for JSONL product records.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, mock_open

# sys.path set up by conftest.py

from dags.src.preprocess.product import (
    _normalize_text,
    _normalize_categories,
    _title_group_key,
    _normalize_parent_asin,
    _process_stage1_batch,
    preprocess_product_data,
    ProductStats,
    WORKING_COLUMNS,
)

# =============================================================================
# 1. Tests for text and category normalization
# =============================================================================

@pytest.mark.parametrize("input_val, expected", [
    ("  Apple Watch  ", "Apple Watch"), # strip leading/trailing whitespace
    (None, ""),                         # handle None
    (np.nan, ""),                       # handle NaN
    (["A", "B"], '["A", "B"]'),         # list -> JSON string
    ({"key": "val"}, '{"key": "val"}')  # dict -> JSON string
])
def test_normalize_text_scenarios(input_val, expected):
    """Test conversion logic for multiple input types into normalized string output."""
    assert _normalize_text(input_val) == expected

def test_normalize_categories_list():
    """Scenario: A category list should be joined with ' > ' in the correct order."""
    test_list = ["Electronics", "Computers", "Laptops"]
    assert _normalize_categories(test_list) == "Electronics > Computers > Laptops"

def test_normalize_parent_asin_tokens():
    """Scenario: Invalid ID tokens should be normalized to empty string."""
    assert _normalize_parent_asin("NaN") == ""
    assert _normalize_parent_asin("NULL") == ""
    assert _normalize_parent_asin("A123") == "A123"

# =============================================================================
# 2. Tests for grouping logic
# =============================================================================

def test_title_group_key_logic():
    """Scenario: Title group key should use the first three tokens of a title."""
    title = "Sony PlayStation 5 Console White"
    # expected: sony_playstation_5
    assert _title_group_key(title) == "sony_playstation_5"

# =============================================================================
# 3. Tests for the core pipeline (advanced mocking to avoid IO conflicts)
# =============================================================================

@patch("dags.src.preprocess.product.ensure_output_dir")
@patch("os.remove")
def test_preprocess_product_full_flow(mock_remove, mock_ensure):
    """
    Scenario: Test the full preprocessing flow.
    Use side_effect to simulate a two-stage read process so we validate the code logic,
    not just final data shape.
    """
    # Raw input data (includes duplicates). Keep it as a single string to avoid line break issues.
    input_jsonl = (
        '{"parent_asin": "P1", "title": "Game A", "price": 50, "categories": ["Toy"]}\n'
        '{"parent_asin": "P1", "title": "Game A", "price": 50, "categories": ["Toy"]}\n'
        '{"parent_asin": "P2", "title": "Game A New", "price": null, "categories": ["Toy"]}\n'
    )
    
    # Simulated stage-1 temp output (already de-duplicated)
    stage1_jsonl = (
        '{"parent_asin": "P1", "title": "Game A", "price": 50, "title_group_key": "game_a", "categories": "Toy"}\n'
        '{"parent_asin": "P2", "title": "Game A New", "price": null, "title_group_key": "game_a", "categories": "Toy"}\n'
    )
    
    # side_effect to return different content depending on which file path is being read
    def open_side_effect(path, mode='r', **kwargs):
        if 'r' in mode:
            # If reading the raw input file, return input_jsonl
            if "mock_in" in str(path):
                return mock_open(read_data=input_jsonl).return_value
            # Otherwise, treat it as reading the stage-1 temp file
            return mock_open(read_data=stage1_jsonl).return_value
        # For write/append operations (w/a), return a generic mock
        return MagicMock()

    with patch("builtins.open", side_effect=open_side_effect), \
         patch("os.replace"):
        # Run the function under test
        df_result = preprocess_product_data("mock_in.jsonl", "mock_out.jsonl")
        
        # Assertion 1: de-duplication logic should keep only 2 rows
        assert len(df_result) == 2
        
        # Assertion 2: missing price should be filled (median should be 50.0)
        p2_price = df_result[df_result["product_id"] == "P2"]["price"].values[0]
        assert p2_price == 50.0


# =============================================================================
# 4. Tests for _process_stage1_batch
# =============================================================================

def _make_record(**overrides):
    base = {
        "parent_asin": "A001",
        "title": "Widget",
        "price": 10.0,
        "average_rating": 4.5,
        "rating_number": 100,
        "description": "A widget",
        "features": "durable",
        "details": "{}",
        "categories": ["Tools"],
    }
    base.update(overrides)
    return base


def test_stage1_empty_records():
    stats = ProductStats()
    df = _process_stage1_batch([], set(), stats)
    assert len(df) == 0
    assert list(df.columns) == WORKING_COLUMNS


def test_stage1_removes_missing_parent_asin():
    stats = ProductStats()
    records = [_make_record(parent_asin=""), _make_record(parent_asin="A001")]
    df = _process_stage1_batch(records, set(), stats)
    assert len(df) == 1
    assert stats.removed_missing_parent_asin == 1


def test_stage1_deduplicates_on_parent_asin():
    stats = ProductStats()
    records = [_make_record(parent_asin="A001"), _make_record(parent_asin="A001")]
    df = _process_stage1_batch(records, set(), stats)
    assert len(df) == 1
    assert stats.duplicates_removed == 1


def test_stage1_global_dedup_across_batches():
    """Seen set persists across batch calls."""
    seen = set()
    stats = ProductStats()
    _process_stage1_batch([_make_record(parent_asin="A001")], seen, stats)
    df2 = _process_stage1_batch([_make_record(parent_asin="A001")], seen, stats)
    assert len(df2) == 0
    assert stats.duplicates_removed == 1


def test_stage1_removes_negative_price():
    stats = ProductStats()
    records = [_make_record(parent_asin="A001", price=-5.0)]
    df = _process_stage1_batch(records, set(), stats)
    assert len(df) == 0
    assert stats.negative_price_removed == 1


def test_stage1_removes_zero_price():
    stats = ProductStats()
    records = [_make_record(parent_asin="A001", price=0)]
    df = _process_stage1_batch(records, set(), stats)
    assert len(df) == 0
    assert stats.low_price_removed == 1


def test_stage1_normalizes_text_fields():
    stats = ProductStats()
    records = [_make_record(parent_asin="A002", title="  Hello World  ")]
    df = _process_stage1_batch(records, set(), stats)
    assert df.iloc[0]["title"] == "Hello World"