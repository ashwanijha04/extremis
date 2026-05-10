"""Tests for #1 auto-consolidation, #4 chunking, #5 AsyncExtremis, #6 contradiction detection."""

from __future__ import annotations

import asyncio

from extremis import AsyncExtremis, Extremis
from extremis.api import _chunk_content
from extremis.config import Config
from extremis.types import MemoryLayer

# ── #4 Chunking ───────────────────────────────────────────────────────────────


class TestChunking:
    def test_short_content_not_chunked(self):
        content = "Short sentence."
        assert _chunk_content(content, chunk_size=200) == [content]

    def test_long_content_split_at_sentences(self):
        content = " ".join([f"Sentence number {i} is here." for i in range(60)])
        chunks = _chunk_content(content, chunk_size=50)
        assert len(chunks) > 1
        rejoined = " ".join(chunks)
        for i in range(60):
            assert f"Sentence number {i}" in rejoined

    def test_chunk_size_zero_disables_chunking(self):
        content = " ".join([f"Sentence {i}." for i in range(100)])
        assert _chunk_content(content, chunk_size=0) == [content]

    def test_remember_stores_multiple_memories_for_long_content(self, tmp_path, mock_embedder):
        config = Config(extremis_home=str(tmp_path), chunk_size=20)
        mem = Extremis(config=config, embedder=mock_embedder)
        long_content = " ".join([f"The agent learned fact number {i} today." for i in range(30)])
        mem.remember(long_content, conversation_id="c1")
        results = mem.recall("fact", limit=50)
        episodic = [r for r in results if r.memory.layer == MemoryLayer.EPISODIC]
        assert len(episodic) > 1

    def test_remember_stores_single_memory_for_short_content(self, api):
        api.remember("Short fact.", conversation_id="c1")
        results = api.recall("fact", limit=10)
        episodic = [r for r in results if r.memory.layer == MemoryLayer.EPISODIC]
        assert len(episodic) == 1


# ── #1 Auto-consolidation ─────────────────────────────────────────────────────


class TestAutoConsolidation:
    def test_background_consolidation_triggered_after_threshold(self, tmp_path, mock_embedder, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        config = Config(
            extremis_home=str(tmp_path),
            auto_consolidate=True,
            auto_consolidate_every=5,
        )
        mem = Extremis(config=config, embedder=mock_embedder)

        triggered = []

        def fake_consolidate():
            triggered.append(1)

        mem._background_consolidate = fake_consolidate

        for i in range(5):
            mem.remember(f"fact {i}", conversation_id="c1")

        assert len(triggered) == 1

    def test_auto_consolidation_disabled_when_flag_false(self, tmp_path, mock_embedder, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        config = Config(
            extremis_home=str(tmp_path),
            auto_consolidate=False,
            auto_consolidate_every=5,
        )
        mem = Extremis(config=config, embedder=mock_embedder)
        triggered = []
        mem._background_consolidate = lambda: triggered.append(1)

        for i in range(10):
            mem.remember(f"fact {i}", conversation_id="c1")

        assert len(triggered) == 0

    def test_auto_consolidation_skipped_without_api_key(self, tmp_path, mock_embedder, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = Config(
            extremis_home=str(tmp_path),
            auto_consolidate=True,
            auto_consolidate_every=5,
        )
        mem = Extremis(config=config, embedder=mock_embedder)
        triggered = []
        mem._background_consolidate = lambda: triggered.append(1)

        for i in range(5):
            mem.remember(f"fact {i}", conversation_id="c1")

        assert len(triggered) == 0


# ── #5 AsyncExtremis ──────────────────────────────────────────────────────────


class TestAsyncExtremis:
    def test_async_remember_and_recall(self, tmp_path, mock_embedder):
        config = Config(extremis_home=str(tmp_path))
        mem = AsyncExtremis(config=config, embedder=mock_embedder)

        async def run():
            await mem.remember("User builds AI agents", conversation_id="c1")
            results = await mem.recall("AI agents", limit=5)
            return results

        results = asyncio.run(run())
        assert any("AI agents" in r.memory.content for r in results)

    def test_async_report_outcome(self, tmp_path, mock_embedder):
        config = Config(extremis_home=str(tmp_path))
        mem = AsyncExtremis(config=config, embedder=mock_embedder)

        async def run():
            await mem.remember("Some fact", conversation_id="c1")
            results = await mem.recall("fact", limit=5)
            ids = [r.memory.id for r in results]
            await mem.report_outcome(ids, success=True)
            updated = await mem.recall("fact", limit=5)
            return updated

        results = asyncio.run(run())
        assert results

    def test_async_remember_now(self, tmp_path, mock_embedder):
        config = Config(extremis_home=str(tmp_path))
        mem = AsyncExtremis(config=config, embedder=mock_embedder)

        async def run():
            await mem.remember_now("Always ask deadline first", layer=MemoryLayer.PROCEDURAL)
            results = await mem.recall("unrelated query", limit=10)
            return results

        results = asyncio.run(run())
        layers = [r.memory.layer for r in results]
        assert MemoryLayer.PROCEDURAL in layers

    def test_async_exposes_sync_store(self, tmp_path, mock_embedder):
        config = Config(extremis_home=str(tmp_path))
        mem = AsyncExtremis(config=config, embedder=mock_embedder)
        assert mem.get_local_store() is not None
        assert mem.get_log() is not None
