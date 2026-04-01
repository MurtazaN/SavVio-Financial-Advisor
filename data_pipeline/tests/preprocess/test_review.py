"""
Tests for Data Preprocessing — preprocess/review.py.

Covers review data cleaning: normalize_text, normalize_id, _to_bool,
_process_batch (dedup, missing critical field removal, type coercion),
and the integration pipeline reading raw JSONL to output preprocessed JSONL.
"""
import os
import sys
import json
import types
import importlib.util
from dataclasses import dataclass

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path constants  (sys.path set up by conftest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------------
# Stub preprocess.utils
# ---------------------------------------------------------------------------
def _stub_utils():
    utils = types.ModuleType("preprocess.utils")
    utils.ensure_output_dir = lambda path: os.makedirs(os.path.dirname(path), exist_ok=True)
    utils.get_raw_path = lambda name: os.path.join("data", "raw", name)
    utils.get_processed_path = lambda name, **kw: os.path.join("data", "processed", name)
    utils.setup_logging = lambda *a, **kw: None
    sys.modules["preprocess.utils"] = utils
    sys.modules["preprocess"] = types.ModuleType("preprocess")
    sys.modules["preprocess"].utils = utils

_stub_utils()

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "src", "preprocess", "review.py"),
        os.path.join(PROJECT_ROOT, "src", "preprocess", "review.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        spec = importlib.util.spec_from_file_location(
            "preprocess.review", fpath,
            submodule_search_locations=[]
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "preprocess"
        sys.modules["preprocess.review"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ImportError("Could not find preprocess/review.py. Searched:\n" + "\n".join(candidates))

M = _load()
PreprocessStats = M.PreprocessStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stats(): return PreprocessStats()
def _seen(): return set()

def _base_record(**overrides):
    r = {
        "user_id": "U1",
        "asin": "A1",
        "parent_asin": "PA1",
        "rating": 4,
        "title": "Great",
        "text": "Loved it",
        "verified_purchase": True,
        "helpful_vote": 2,
    }
    r.update(overrides)
    return r


# =============================================================================
# 1) _normalize_text
# =============================================================================

def test_normalize_text_none_returns_empty():
    assert M._normalize_text(None) == ""

def test_normalize_text_nan_returns_empty():
    import numpy as np
    assert M._normalize_text(float("nan")) == ""

def test_normalize_text_strips_string():
    assert M._normalize_text("  hello  ") == "hello"

def test_normalize_text_list_returns_json():
    out = M._normalize_text(["a", "b"])
    assert out == '["a", "b"]'

def test_normalize_text_dict_returns_json():
    out = M._normalize_text({"k": "v"})
    assert "k" in out

def test_normalize_text_int_returns_string():
    assert M._normalize_text(42) == "42"


# =============================================================================
# 2) _normalize_id
# =============================================================================

def test_normalize_id_nan_string_returns_empty():
    assert M._normalize_id("nan") == ""

def test_normalize_id_strips_whitespace():
    assert M._normalize_id("  U1  ") == "U1"

def test_normalize_id_none_returns_empty():
    assert M._normalize_id(None) == ""


# =============================================================================
# 3) _to_bool
# =============================================================================

@pytest.mark.parametrize("val,expected", [
    (True,    True),
    (False,   False),
    ("true",  True),
    ("1",     True),
    ("yes",   True),
    ("false", False),
    ("0",     False),
    ("no",    False),
    ("",      False),
    (None,    False),
    (float("nan"), False),
])
def test_to_bool(val, expected):
    assert M._to_bool(val) == expected


# =============================================================================
# 4) _process_batch
# =============================================================================

def test_process_batch_empty_records_returns_empty_df():
    out = M._process_batch([], _seen(), _stats())
    assert isinstance(out, pd.DataFrame)
    assert out.empty

def test_process_batch_output_columns():
    records = [_base_record()]
    out = M._process_batch(records, _seen(), _stats())
    assert list(out.columns) == M.FINAL_COLUMNS

def test_process_batch_basic_row():
    records = [_base_record()]
    out = M._process_batch(records, _seen(), _stats())
    assert len(out) == 1
    assert out.iloc[0]["user_id"] == "U1"
    assert out.iloc[0]["product_id"] == "PA1"
    assert out.iloc[0]["review_title"] == "Great"
    assert out.iloc[0]["review_text"] == "Loved it"

def test_process_batch_drops_missing_parent_asin():
    records = [_base_record(parent_asin=None), _base_record(parent_asin="PA2", asin="A2")]
    stats = _stats()
    out = M._process_batch(records, _seen(), stats)
    assert len(out) == 1
    assert stats.removed_missing_parent_asin == 1

def test_process_batch_drops_missing_user_id():
    records = [_base_record(user_id=None), _base_record(user_id="U2", asin="A2")]
    stats = _stats()
    out = M._process_batch(records, _seen(), stats)
    assert len(out) == 1
    assert stats.removed_missing_user_id == 1

def test_process_batch_drops_missing_rating():
    records = [_base_record(rating=None), _base_record(rating=5, asin="A2")]
    stats = _stats()
    out = M._process_batch(records, _seen(), stats)
    assert len(out) == 1
    assert stats.removed_missing_rating == 1

def test_process_batch_removes_duplicates():
    r = _base_record()
    stats = _stats()
    seen = _seen()
    out = M._process_batch([r, r], seen, stats)
    assert len(out) == 1
    assert stats.duplicates_removed == 1

def test_process_batch_cross_batch_dedup():
    """Duplicates across batches are detected via shared seen set."""
    r = _base_record()
    seen = _seen()
    stats = _stats()
    M._process_batch([r], seen, stats)
    out2 = M._process_batch([r], seen, stats)
    assert len(out2) == 0

def test_process_batch_helpful_vote_defaults_to_zero():
    records = [_base_record(helpful_vote=None)]
    out = M._process_batch(records, _seen(), _stats())
    assert out.iloc[0]["helpful_vote"] == 0

def test_process_batch_verified_purchase_bool():
    records = [_base_record(verified_purchase="yes")]
    out = M._process_batch(records, _seen(), _stats())
    assert bool(out.iloc[0]["verified_purchase"]) is True

def test_process_batch_rating_coerced_to_float():
    records = [_base_record(rating="3")]
    out = M._process_batch(records, _seen(), _stats())
    assert isinstance(out.iloc[0]["rating"], float)


# =============================================================================
# 5) preprocess_review_data (integration)
# =============================================================================

def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def test_preprocess_review_data_creates_output(tmp_path):
    inp = tmp_path / "reviews.jsonl"
    out = tmp_path / "out.jsonl"
    _write_jsonl(inp, [_base_record()])
    M.preprocess_review_data(str(inp), str(out))
    assert out.exists()
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["user_id"] == "U1"
    assert row["product_id"] == "PA1"

def test_preprocess_review_data_skips_malformed_json(tmp_path):
    inp = tmp_path / "reviews.jsonl"
    out = tmp_path / "out.jsonl"
    with open(inp, "w") as f:
        f.write("{bad json}\n")
        f.write(json.dumps(_base_record()) + "\n")
    M.preprocess_review_data(str(inp), str(out))
    lines = [l for l in out.read_text().strip().split("\n") if l]
    assert len(lines) == 1

def test_preprocess_review_data_deduplicates(tmp_path):
    inp = tmp_path / "reviews.jsonl"
    out = tmp_path / "out.jsonl"
    r = _base_record()
    _write_jsonl(inp, [r, r])
    M.preprocess_review_data(str(inp), str(out))
    lines = [l for l in out.read_text().strip().split("\n") if l]
    assert len(lines) == 1

def test_preprocess_review_data_output_has_correct_columns(tmp_path):
    inp = tmp_path / "reviews.jsonl"
    out = tmp_path / "out.jsonl"
    _write_jsonl(inp, [_base_record()])
    M.preprocess_review_data(str(inp), str(out))
    row = json.loads(out.read_text().strip())
    for col in M.FINAL_COLUMNS:
        assert col in row

def test_preprocess_review_data_empty_file(tmp_path):
    inp = tmp_path / "reviews.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text("")
    M.preprocess_review_data(str(inp), str(out))
    assert out.exists()
    assert out.read_text().strip() == ""