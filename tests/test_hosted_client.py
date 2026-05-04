"""HostedClient — all HTTP calls mocked with respx or httpx mock."""
from __future__ import annotations

import pytest
pytest.importorskip("httpx", reason="httpx not installed")

from unittest.mock import patch, MagicMock
from lore_ai.client import HostedClient
from lore_ai.types import MemoryLayer


def make_client(responses: dict | None = None) -> HostedClient:
    """Return a HostedClient with httpx.Client patched."""
    client = HostedClient.__new__(HostedClient)
    client._base = "http://localhost:8000"
    mock_http = MagicMock()
    client._http = mock_http
    return client, mock_http


class TestHostedClientInterface:
    def test_remember_posts_to_correct_path(self):
        client, http = make_client()
        http.post.return_value = MagicMock(status_code=204, content=b"")
        client.remember("test content", conversation_id="c1")
        http.post.assert_called_once()
        path, kwargs = http.post.call_args[0][0], http.post.call_args[1]
        assert path == "/v1/memories/remember"
        assert kwargs["json"]["content"] == "test content"

    def test_recall_returns_recall_results(self):
        client, http = make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [
            {
                "memory": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "namespace": "default",
                    "layer": "semantic",
                    "content": "User is a developer",
                    "score": 0.0,
                    "confidence": 0.9,
                    "metadata": {},
                    "source_memory_ids": [],
                    "validity_start": "2026-01-01T00:00:00",
                    "validity_end": None,
                    "created_at": "2026-01-01T00:00:00",
                    "last_accessed_at": None,
                    "access_count": 0,
                    "do_not_consolidate": False,
                },
                "relevance": 0.87,
                "final_rank": 0.73,
            }
        ]}
        http.post.return_value = mock_resp
        results = client.recall("developer", limit=5)
        assert len(results) == 1
        assert results[0].memory.content == "User is a developer"
        assert results[0].relevance == pytest.approx(0.87)

    def test_report_posts_to_correct_path(self):
        from uuid import uuid4
        client, http = make_client()
        http.post.return_value = MagicMock(status_code=204, content=b"")
        client.report_outcome([uuid4()], success=True)
        path = http.post.call_args[0][0]
        assert path == "/v1/memories/report"

    def test_context_manager(self):
        with patch("httpx.Client") as mock_cls:
            mock_http = MagicMock()
            mock_cls.return_value = mock_http
            with HostedClient(api_key="lore_sk_test") as c:
                pass
            mock_http.close.assert_called_once()
