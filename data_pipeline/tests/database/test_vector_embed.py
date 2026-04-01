"""
Tests for Load to Database — vector_embed.py.

Covers embedding generation and storage for products and reviews using
sentence-transformers and pgvector. Tests: _flatten_details, build_product_text,
build_review_text, _read_file, generate_embeddings, store_product_embeddings,
store_review_embeddings, embed_products, embed_reviews, run_embed.
"""
import os
import sys
import json
import types
import importlib
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path constants  (sys.path set up by conftest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# --- stubs ---
if "db_config" not in sys.modules:
    _db = types.ModuleType("db_config")
    _db.get_engine = lambda env="dev": MagicMock()
    _db.ensure_pgvector = lambda e: None
    sys.modules["db_config"] = _db

if "models" not in sys.modules:
    _mo = types.ModuleType("models")
    _mo.create_tables = lambda e: None
    sys.modules["models"] = _mo

if "sentence_transformers" not in sys.modules:
    _st = types.ModuleType("sentence_transformers")
    class _FakeST:
        def __init__(self, *a, **kw): pass
        def encode(self, sentences, **kw):
            return np.zeros((len(sentences), 384), dtype=np.float32)
    _st.SentenceTransformer = _FakeST
    sys.modules["sentence_transformers"] = _st

# --- load module ---
import importlib.util
_fpath = os.path.join(PROJECT_ROOT, "dags", "src", "database", "vector_embed.py")
_spec = importlib.util.spec_from_file_location("vector_embed", _fpath)
M = importlib.util.module_from_spec(_spec)
sys.modules["vector_embed"] = M
_spec.loader.exec_module(M)


# =============================================================================
# 1) _flatten_details tests
# =============================================================================

@pytest.mark.parametrize("input_val, expected_contains", [
    (None, ""),
    (float("nan"), ""),
    ("", ""),
    ("  hello  ", "hello"),
    ('{"a": "1", "b": "x"}', "a: 1"),
    ('{"a": "1", "b": "x"}', "b: x"),
    ("{bad json", "{bad json"),
    ({"color": "red", "size": "M"}, "color: red"),
    ({"color": "red", "size": "M"}, "size: M"),
    (["a", "b"], ""),
    ([{"k": "v"}], ""),
])
def test_flatten_details_various_inputs(input_val, expected_contains):
    out = M._flatten_details(input_val)
    assert isinstance(out, str)
    if expected_contains == "":
        assert out.strip() == ""
    else:
        assert expected_contains in out

def test_flatten_details_dict_ignores_empty_values():
    out = M._flatten_details({"a": "", "b": None, "c": "ok"})
    assert "c: ok" in out
    assert "a:" not in out
    assert "b:" not in out

# =============================================================================
# 2) build_product_text tests
# =============================================================================

def test_build_product_text_concatenates_fields_and_details():
    row = pd.Series({"product_name": "Widget", "category": "Tools",
                     "description": "A useful widget", "features": "Lightweight",
                     "details": {"color": "blue"}})
    out = M.build_product_text(row)
    assert "Widget" in out and "Tools" in out and "color: blue" in out and " | " in out

def test_build_product_text_skips_empty_fields():
    row = pd.Series({"product_name": "Widget", "category": "",
                     "description": float("nan"), "features": "   ", "details": None})
    assert M.build_product_text(row) == "Widget"

# =============================================================================
# 3) build_review_text tests
# =============================================================================

def test_build_review_text_title_and_body_joined():
    row = pd.Series({"review_title": "Great", "review_text": "Loved it"})
    assert M.build_review_text(row) == "Great | Loved it"

def test_build_review_text_skips_empty_values():
    row = pd.Series({"review_title": "  ", "review_text": "Body"})
    assert M.build_review_text(row) == "Body"

# =============================================================================
# 4) _read_file tests
# =============================================================================

def test_read_file_csv(tmp_path):
    p = tmp_path / "products.csv"
    pd.DataFrame([{"a": 1, "b": "x"}]).to_csv(p, index=False)
    df = M._read_file(str(p))
    assert df.shape == (1, 2) and df.loc[0, "a"] == 1

def test_read_file_jsonl(tmp_path):
    p = tmp_path / "reviews.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"a": 1, "b": "x"}) + "\n")
        f.write(json.dumps({"a": 2, "b": "y"}) + "\n")
    df = M._read_file(str(p))
    assert df.shape == (2, 2) and df.loc[1, "b"] == "y"

# =============================================================================
# 5) generate_embeddings tests
# =============================================================================

def test_generate_embeddings_calls_model_encode_with_expected_args():
    fake_model = MagicMock()
    dim = getattr(M, "EMBEDDING_DIM", 384)
    fake_model.encode.return_value = np.zeros((2, dim), dtype=np.float32)
    out = M.generate_embeddings(["a", "b"], fake_model)
    assert isinstance(out, np.ndarray) and out.shape == (2, dim)

# =============================================================================
# 6) store_* tests
# =============================================================================

def _mock_engine():
    engine, conn, cm = MagicMock(), MagicMock(), MagicMock()
    cm.__enter__.return_value = conn
    cm.__exit__.return_value = False
    engine.begin.return_value = cm
    return engine, conn

def test_store_product_embeddings_executes_db_calls():
    engine, conn = _mock_engine()
    dim = getattr(M, "EMBEDDING_DIM", 384)
    M.store_product_embeddings(engine, ["p1","p2","p3"], np.zeros((3,dim), dtype=np.float32))
    assert conn.execute.call_count == 3

def test_store_review_embeddings_executes_db_calls():
    engine, conn = _mock_engine()
    dim = getattr(M, "EMBEDDING_DIM", 384)
    M.store_review_embeddings(engine, ["p1","p2"], ["u1","u2"], np.zeros((2,dim), dtype=np.float32))
    assert conn.execute.call_count == 2

# =============================================================================
# 7) embed_products / embed_reviews tests
# =============================================================================

def test_embed_products_raises_without_product_id(tmp_path):
    p = tmp_path / "products.csv"
    pd.DataFrame([{"product_name": "x"}]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="product_id"):
        M.embed_products(MagicMock(), str(p), MagicMock())

def test_embed_products_filters_empty_text_and_stores(tmp_path):
    p = tmp_path / "products.csv"
    pd.DataFrame([
        {"product_id": "p1", "product_name": "Widget", "category": "Tools"},
        {"product_id": "p2", "product_name": "   ", "category": ""},
    ]).to_csv(p, index=False)
    dim = getattr(M, "EMBEDDING_DIM", 384)
    with patch.object(M, "generate_embeddings", return_value=np.zeros((1,dim), dtype=np.float32)) as mg, \
         patch.object(M, "store_product_embeddings") as ms:
        n = M.embed_products(MagicMock(), str(p), MagicMock())
        assert n == 1
        mg.assert_called_once()
        assert ms.call_args[0][1] == ["p1"]

def test_embed_reviews_raises_without_required_columns(tmp_path):
    p = tmp_path / "reviews.csv"
    pd.DataFrame([{"review_text": "x"}]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="product_id"):
        M.embed_reviews(MagicMock(), str(p), MagicMock())

def test_embed_reviews_skips_when_no_text(tmp_path):
    p = tmp_path / "reviews.csv"
    pd.DataFrame([
        {"product_id": "p1", "user_id": "u1", "review_title": "   ", "review_text": ""},
        {"product_id": "p2", "user_id": "u2", "review_title": None, "review_text": "   "},
    ]).to_csv(p, index=False)
    with patch.object(M, "generate_embeddings") as mg, \
         patch.object(M, "store_review_embeddings") as ms:
        assert M.embed_reviews(MagicMock(), str(p), MagicMock()) == 0
        mg.assert_not_called()
        ms.assert_not_called()

def test_embed_reviews_filters_and_stores(tmp_path):
    p = tmp_path / "reviews.csv"
    pd.DataFrame([
        {"product_id": "p1", "user_id": "u1", "review_title": "Nice", "review_text": ""},
        {"product_id": "p2", "user_id": "u2", "review_title": "   ", "review_text": "   "},
    ]).to_csv(p, index=False)
    dim = getattr(M, "EMBEDDING_DIM", 384)
    with patch.object(M, "generate_embeddings", return_value=np.zeros((1,dim), dtype=np.float32)) as mg, \
         patch.object(M, "store_review_embeddings") as ms:
        n = M.embed_reviews(MagicMock(), str(p), MagicMock())
        assert n == 1
        mg.assert_called_once()
        assert ms.call_args[0][1] == ["p1"]
        assert ms.call_args[0][2] == ["u1"]


# =============================================================================
# 8) run_embed (end-to-end orchestrator)
# =============================================================================

def test_run_embed_products_only():
    with patch.object(M, "get_engine") as mock_ge, \
         patch.object(M, "ensure_pgvector") as mock_pgv, \
         patch.object(M, "create_tables") as mock_ct, \
         patch.object(M, "_ensure_embedding_tables") as mock_et, \
         patch.object(M, "load_model") as mock_lm, \
         patch.object(M, "embed_products", return_value=5) as mock_ep, \
         patch.object(M, "embed_reviews") as mock_er:
        result = M.run_embed("/products.csv", reviews_path=None, env="dev")
    mock_ge.assert_called_once_with("dev")
    mock_pgv.assert_called_once()
    mock_ct.assert_called_once()
    mock_et.assert_called_once()
    mock_lm.assert_called_once()
    mock_ep.assert_called_once()
    mock_er.assert_not_called()
    assert result == {"products_embedded": 5, "reviews_embedded": 0}


def test_run_embed_with_reviews():
    with patch.object(M, "get_engine"), \
         patch.object(M, "ensure_pgvector"), \
         patch.object(M, "create_tables"), \
         patch.object(M, "_ensure_embedding_tables"), \
         patch.object(M, "load_model"), \
         patch.object(M, "embed_products", return_value=3), \
         patch.object(M, "embed_reviews", return_value=7):
        result = M.run_embed("/products.csv", "/reviews.csv", env="prod")
    assert result == {"products_embedded": 3, "reviews_embedded": 7}
