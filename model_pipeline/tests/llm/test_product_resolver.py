"""Tests for the product resolver — vector search and direct DB lookup."""

import pytest
from unittest.mock import MagicMock, patch

from llm.product_resolver import (
    ProductMatch,
    resolve_product,
    resolve_product_by_id,
)


class TestProductMatch:
    def test_dataclass_creation(self):
        match = ProductMatch(
            product_id="B0BX1234",
            product_name="Sony WH-1000XM5",
            price=349.99,
            category="Electronics",
            similarity_score=0.94,
        )
        assert match.product_id == "B0BX1234"
        assert match.product_name == "Sony WH-1000XM5"
        assert match.price == 349.99
        assert match.category == "Electronics"
        assert match.similarity_score == 0.94


class TestResolveProduct:
    def test_empty_text_returns_none(self):
        engine = MagicMock()
        result = resolve_product("", engine)
        assert result is None

    def test_whitespace_only_returns_none(self):
        engine = MagicMock()
        result = resolve_product("   ", engine)
        assert result is None

    @patch("llm.product_resolver._get_embedding_model")
    def test_no_results_returns_none(self, mock_model_fn):
        """When the DB returns no rows, should return None."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384)
        mock_model_fn.return_value = mock_model

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []

        engine = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        result = resolve_product("nonexistent product xyz", engine)
        assert result is None

    @patch("llm.product_resolver._get_embedding_model")
    def test_below_threshold_returns_none(self, mock_model_fn):
        """When the best match is below similarity threshold, should return None."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384)
        mock_model_fn.return_value = mock_model

        # Return a row with very low similarity
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("B0001", "Random Product", 10.0, "Unknown", 0.1),  # sim=0.1 < threshold
        ]

        engine = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        result = resolve_product("something", engine, threshold=0.3)
        assert result is None

    @patch("llm.product_resolver._get_embedding_model")
    def test_above_threshold_returns_match(self, mock_model_fn):
        """When the best match is above threshold, should return ProductMatch."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384)
        mock_model_fn.return_value = mock_model

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("B0BX1234", "Sony WH-1000XM5", 349.99, "Electronics", 0.94),
        ]

        engine = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        result = resolve_product("Sony headphones", engine)
        assert result is not None
        assert isinstance(result, ProductMatch)
        assert result.product_id == "B0BX1234"
        assert result.product_name == "Sony WH-1000XM5"
        assert result.price == 349.99
        assert result.similarity_score == 0.94

    @patch("llm.product_resolver._get_embedding_model")
    def test_db_error_returns_none(self, mock_model_fn):
        """When the DB query fails, should return None gracefully."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384)
        mock_model_fn.return_value = mock_model

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("DB connection lost")

        engine = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        result = resolve_product("Sony headphones", engine)
        assert result is None


class TestResolveProductById:
    def test_found_returns_match(self):
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (
            "B0BX1234", "Sony WH-1000XM5", 349.99, "Electronics"
        )

        engine = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        result = resolve_product_by_id("B0BX1234", engine)
        assert result is not None
        assert result.product_id == "B0BX1234"
        assert result.similarity_score == 1.0  # exact match

    def test_not_found_returns_none(self):
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None

        engine = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        result = resolve_product_by_id("NONEXISTENT", engine)
        assert result is None

    def test_db_error_returns_none(self):
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("DB error")

        engine = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        result = resolve_product_by_id("B0BX1234", engine)
        assert result is None
