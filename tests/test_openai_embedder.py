"""OpenAIEmbedder — fully mocked (no real API calls)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai", reason="openai not installed — pip install 'extremis[openai]'")

from extremis.embeddings.openai import OpenAIEmbedder, _DIMS


class TestOpenAIEmbedder:
    def _make_mock_response(self, embeddings: list[list[float]]):
        response = MagicMock()
        response.data = [MagicMock(embedding=e) for e in embeddings]
        return response

    def test_dim_small(self):
        with patch("openai.OpenAI"):
            e = OpenAIEmbedder("text-embedding-3-small")
        assert e.dim == 1536

    def test_dim_large(self):
        with patch("openai.OpenAI"):
            e = OpenAIEmbedder("text-embedding-3-large")
        assert e.dim == 3072

    def test_dim_ada(self):
        with patch("openai.OpenAI"):
            e = OpenAIEmbedder("text-embedding-ada-002")
        assert e.dim == 1536

    def test_embed_calls_api(self):
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = self._make_mock_response([[0.1] * 1536])
        with patch("openai.OpenAI", return_value=mock_client):
            e = OpenAIEmbedder("text-embedding-3-small")
        result = e.embed("hello world")
        assert result == [0.1] * 1536
        mock_client.embeddings.create.assert_called_once()

    def test_embed_batch_batches_correctly(self):
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = self._make_mock_response(
            [[0.1] * 1536, [0.2] * 1536]
        )
        with patch("openai.OpenAI", return_value=mock_client):
            e = OpenAIEmbedder("text-embedding-3-small", batch_size=2)
        result = e.embed_batch(["text 1", "text 2"])
        assert len(result) == 2
        assert result[0] == [0.1] * 1536
        assert result[1] == [0.2] * 1536

    def test_embed_batch_respects_batch_size(self):
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = self._make_mock_response([[0.1] * 1536])
        with patch("openai.OpenAI", return_value=mock_client):
            e = OpenAIEmbedder("text-embedding-3-small", batch_size=1)
        e.embed_batch(["a", "b", "c"])
        assert mock_client.embeddings.create.call_count == 3

    def test_import_error_without_package(self):
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="extremis\\[openai\\]"):
                OpenAIEmbedder("text-embedding-3-small")
