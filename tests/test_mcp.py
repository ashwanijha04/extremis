"""MCP server tool tests — no network, no LLM."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from friday_memory.api import FridayMemory
from friday_memory.config import Config
from friday_memory.storage.log import FileLogStore
from friday_memory.storage.sqlite import SQLiteMemoryStore
from friday_memory.types import Memory, MemoryLayer


@pytest.fixture
def config(tmp_path):
    return Config(
        friday_home=str(tmp_path),
        log_dir=str(tmp_path / "log"),
        local_db_path=str(tmp_path / "local.db"),
    )


@pytest.fixture
def mock_embedder():
    e = MagicMock()
    e.embed.return_value = [0.1] * 384
    e.embed_batch.return_value = [[0.1] * 384]
    e.dim = 384
    return e


@pytest.fixture
def memory_api(config, mock_embedder):
    return FridayMemory(config=config, embedder=mock_embedder)


class TestMemoryAPIViaTools:
    """Test the FridayMemory API directly (same logic the MCP tools call)."""

    def test_remember_and_recall(self, memory_api):
        memory_api.remember("User is building a WhatsApp AI", conversation_id="c1")
        results = memory_api.recall("WhatsApp", limit=5)
        assert len(results) >= 1
        assert any("WhatsApp" in r.memory.content for r in results)

    def test_remember_now_direct_write(self, memory_api):
        mem = memory_api.remember_now(
            "User's flight departs Thursday at 06:00",
            layer=MemoryLayer.EPISODIC,
            confidence=0.99,
        )
        assert mem.layer == MemoryLayer.EPISODIC
        assert mem.confidence == 0.99

        retrieved = memory_api.get_local_store().get(mem.id)
        assert retrieved is not None

    def test_report_outcome_adjusts_score(self, memory_api):
        mem = memory_api.remember_now("Prefer concise answers", layer=MemoryLayer.PROCEDURAL)
        original_score = mem.score

        memory_api.report_outcome([mem.id], success=True, weight=2.0)

        updated = memory_api.get_local_store().get(mem.id)
        assert updated is not None
        assert updated.score == original_score + 2.0

    def test_report_negative_outcome(self, memory_api):
        mem = memory_api.remember_now("User likes long answers", layer=MemoryLayer.PROCEDURAL)
        memory_api.report_outcome([mem.id], success=False, weight=1.0)

        updated = memory_api.get_local_store().get(mem.id)
        assert updated is not None
        assert updated.score == -1.0

    def test_recall_returns_procedural_always(self, memory_api):
        memory_api.remember_now("Always check deadline before suggesting", layer=MemoryLayer.PROCEDURAL)
        memory_api.remember("Some unrelated episodic fact", conversation_id="c2")

        # Query something unrelated — procedural should still come back
        results = memory_api.recall("weather in Dubai", limit=5)
        layers = [r.memory.layer for r in results]
        assert MemoryLayer.PROCEDURAL in layers

    def test_remember_writes_to_log(self, memory_api, config):
        memory_api.remember("User mentioned they hate meetings", conversation_id="c3")

        log_store = FileLogStore(config.resolved_log_dir())
        entries = log_store.read_since(None)
        assert len(entries) >= 1
        assert any("hate meetings" in e.content for e in entries)

    def test_remember_now_with_expiry(self, memory_api):
        from datetime import timedelta
        expiry = datetime.now(tz=timezone.utc) + timedelta(hours=2)
        mem = memory_api.remember_now(
            "User is in a meeting until 3pm",
            layer=MemoryLayer.WORKING,
            expires_at=expiry,
        )
        assert mem.validity_end is not None
