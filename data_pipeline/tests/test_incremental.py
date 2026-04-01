"""
Tests for Incremental Merge Utilities — incremental.py.

Covers the shared merge functions that enable incremental processing
across all pipeline stages: merge_csv, merge_jsonl, file_checksum.
"""
import json
import os
import sys
import types
import importlib.util

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path constants  (sys.path set up by conftest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DAGS_SRC = os.path.join(PROJECT_ROOT, "dags", "src")


# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    fpath = os.path.join(DAGS_SRC, "incremental.py")
    if not os.path.isfile(fpath):
        raise ImportError(f"Could not find incremental.py at {fpath}")
    spec = importlib.util.spec_from_file_location("incremental", fpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["incremental"] = mod
    spec.loader.exec_module(mod)
    return mod

M = _load()


# =============================================================================
# 1) file_checksum tests
# =============================================================================

def test_file_checksum_deterministic(tmp_path):
    """Same content produces the same hash."""
    p = tmp_path / "test.txt"
    p.write_text("hello world\n")
    hash1 = M.file_checksum(str(p))
    hash2 = M.file_checksum(str(p))
    assert hash1 == hash2
    assert len(hash1) == 32  # MD5 hex digest length


def test_file_checksum_different_content(tmp_path):
    """Different content produces different hashes."""
    p1 = tmp_path / "a.txt"
    p2 = tmp_path / "b.txt"
    p1.write_text("hello")
    p2.write_text("world")
    assert M.file_checksum(str(p1)) != M.file_checksum(str(p2))


# =============================================================================
# 2) merge_csv tests
# =============================================================================

def test_merge_csv_first_run(tmp_path):
    """When no existing file, writes new data as-is."""
    new_path = tmp_path / "new.csv"
    existing_path = tmp_path / "existing.csv"
    pd.DataFrame([
        {"user_id": "u1", "income": 5000},
        {"user_id": "u2", "income": 3000},
    ]).to_csv(new_path, index=False)

    stats = M.merge_csv(str(new_path), str(existing_path), key_cols=["user_id"])
    assert stats["appended"] == 2
    assert stats["updated"] == 0
    assert stats["unchanged"] == 0
    assert stats["total"] == 2

    result = pd.read_csv(str(existing_path))
    assert len(result) == 2


def test_merge_csv_update_existing(tmp_path):
    """Records with same key are replaced by new version."""
    new_path = tmp_path / "new.csv"
    existing_path = tmp_path / "existing.csv"
    pd.DataFrame([
        {"user_id": "u1", "income": 5000},
    ]).to_csv(existing_path, index=False)

    pd.DataFrame([
        {"user_id": "u1", "income": 9999},  # Updated income
    ]).to_csv(new_path, index=False)

    stats = M.merge_csv(str(new_path), str(existing_path), key_cols=["user_id"])
    assert stats["updated"] == 1
    assert stats["appended"] == 0
    assert stats["unchanged"] == 0
    assert stats["total"] == 1

    result = pd.read_csv(str(existing_path))
    assert len(result) == 1
    assert result.loc[0, "income"] == 9999


def test_merge_csv_append_new(tmp_path):
    """Records with new keys are appended."""
    new_path = tmp_path / "new.csv"
    existing_path = tmp_path / "existing.csv"
    pd.DataFrame([
        {"user_id": "u1", "income": 5000},
    ]).to_csv(existing_path, index=False)

    pd.DataFrame([
        {"user_id": "u2", "income": 3000},  # New key
    ]).to_csv(new_path, index=False)

    stats = M.merge_csv(str(new_path), str(existing_path), key_cols=["user_id"])
    assert stats["updated"] == 0
    assert stats["appended"] == 1
    assert stats["unchanged"] == 1
    assert stats["total"] == 2

    result = pd.read_csv(str(existing_path))
    assert len(result) == 2
    user_ids = set(result["user_id"])
    assert user_ids == {"u1", "u2"}


def test_merge_csv_mixed_update_and_append(tmp_path):
    """Some records updated, some appended, some unchanged."""
    new_path = tmp_path / "new.csv"
    existing_path = tmp_path / "existing.csv"
    pd.DataFrame([
        {"user_id": "u1", "income": 5000},
        {"user_id": "u2", "income": 3000},
    ]).to_csv(existing_path, index=False)

    pd.DataFrame([
        {"user_id": "u1", "income": 8000},  # Update
        {"user_id": "u3", "income": 4000},  # Append
    ]).to_csv(new_path, index=False)

    stats = M.merge_csv(str(new_path), str(existing_path), key_cols=["user_id"])
    assert stats["updated"] == 1
    assert stats["appended"] == 1
    assert stats["unchanged"] == 1
    assert stats["total"] == 3

    result = pd.read_csv(str(existing_path))
    assert len(result) == 3
    u1_income = result[result["user_id"] == "u1"]["income"].values[0]
    assert u1_income == 8000  # New value wins


# =============================================================================
# 3) merge_jsonl tests
# =============================================================================

def _write_jsonl(path, records):
    """Helper to write records as JSONL."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _read_jsonl(path):
    """Helper to read JSONL into list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def test_merge_jsonl_first_run(tmp_path):
    """When no existing file, writes new data as-is."""
    new_path = str(tmp_path / "new.jsonl")
    existing_path = str(tmp_path / "existing.jsonl")
    _write_jsonl(new_path, [
        {"product_id": "p1", "price": 10.0},
        {"product_id": "p2", "price": 20.0},
    ])

    stats = M.merge_jsonl(new_path, existing_path, key_cols=["product_id"])
    assert stats["appended"] == 2
    assert stats["total"] == 2

    result = _read_jsonl(existing_path)
    assert len(result) == 2


def test_merge_jsonl_update_existing(tmp_path):
    """Records with same key are replaced."""
    new_path = str(tmp_path / "new.jsonl")
    existing_path = str(tmp_path / "existing.jsonl")
    _write_jsonl(existing_path, [
        {"product_id": "p1", "price": 10.0},
    ])
    _write_jsonl(new_path, [
        {"product_id": "p1", "price": 99.0},
    ])

    stats = M.merge_jsonl(new_path, existing_path, key_cols=["product_id"])
    assert stats["updated"] == 1
    assert stats["appended"] == 0
    assert stats["total"] == 1

    result = _read_jsonl(existing_path)
    assert len(result) == 1
    assert result[0]["price"] == 99.0


def test_merge_jsonl_append_new(tmp_path):
    """New keys are appended."""
    new_path = str(tmp_path / "new.jsonl")
    existing_path = str(tmp_path / "existing.jsonl")
    _write_jsonl(existing_path, [
        {"product_id": "p1", "price": 10.0},
    ])
    _write_jsonl(new_path, [
        {"product_id": "p2", "price": 20.0},
    ])

    stats = M.merge_jsonl(new_path, existing_path, key_cols=["product_id"])
    assert stats["appended"] == 1
    assert stats["unchanged"] == 1
    assert stats["total"] == 2

    result = _read_jsonl(existing_path)
    assert len(result) == 2


def test_merge_jsonl_composite_key(tmp_path):
    """Merge with composite key (user_id, product_id)."""
    new_path = str(tmp_path / "new.jsonl")
    existing_path = str(tmp_path / "existing.jsonl")
    _write_jsonl(existing_path, [
        {"user_id": "u1", "product_id": "p1", "rating": 3},
        {"user_id": "u1", "product_id": "p2", "rating": 4},
    ])
    _write_jsonl(new_path, [
        {"user_id": "u1", "product_id": "p1", "rating": 5},  # Update
        {"user_id": "u2", "product_id": "p1", "rating": 2},  # Append
    ])

    stats = M.merge_jsonl(
        new_path, existing_path, key_cols=["user_id", "product_id"]
    )
    assert stats["updated"] == 1
    assert stats["appended"] == 1
    assert stats["unchanged"] == 1
    assert stats["total"] == 3

    result = _read_jsonl(existing_path)
    assert len(result) == 3
    # Verify the u1+p1 record was updated to rating 5
    u1_p1 = [r for r in result if r["user_id"] == "u1" and r["product_id"] == "p1"]
    assert len(u1_p1) == 1
    assert u1_p1[0]["rating"] == 5
