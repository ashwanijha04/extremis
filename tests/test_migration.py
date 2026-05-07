"""Migration tests — SQLite ↔ Chroma round-trip, mocked embedder."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from extremis.migrate import Migrator, MigrationResult
from extremis.storage.sqlite import SQLiteMemoryStore
from extremis.types import MemoryLayer
from .conftest import make_memory

pytest.importorskip("chromadb", reason="chromadb not installed")


@pytest.fixture
def sqlite_source(tmp_config):
    return SQLiteMemoryStore(tmp_config.resolved_local_db_path(), tmp_config)


@pytest.fixture
def chroma_dest(tmp_path, tmp_config):
    from extremis.storage.chroma import ChromaMemoryStore
    store = ChromaMemoryStore(str(tmp_path / "chroma_dest"), tmp_config)
    yield store
    store.close()


@pytest.fixture
def chroma_source(tmp_path, tmp_config):
    from extremis.storage.chroma import ChromaMemoryStore
    store = ChromaMemoryStore(str(tmp_path / "chroma_source"), tmp_config)
    yield store
    store.close()


class TestMigratorBasics:
    def test_empty_source_returns_zero(self, sqlite_source, chroma_dest):
        result = Migrator().run(sqlite_source, chroma_dest)
        assert result.memories_migrated == 0
        assert result.memories_skipped == 0

    def test_migrates_all_memories(self, sqlite_source, chroma_dest):
        for i in range(5):
            sqlite_source.store(make_memory(f"fact {i}", MemoryLayer.SEMANTIC))

        result = Migrator().run(sqlite_source, chroma_dest)
        assert result.memories_migrated == 5
        assert result.memories_skipped == 0

    def test_migrated_memories_queryable_in_dest(self, sqlite_source, chroma_dest):
        mem = sqlite_source.store(make_memory("Python developer", MemoryLayer.SEMANTIC))
        Migrator().run(sqlite_source, chroma_dest)

        retrieved = chroma_dest.get(mem.id)
        assert retrieved is not None
        assert retrieved.content == "Python developer"

    def test_dry_run_writes_nothing(self, sqlite_source, chroma_dest):
        sqlite_source.store(make_memory("fact", MemoryLayer.SEMANTIC))
        result = Migrator().run(sqlite_source, chroma_dest, dry_run=True)
        assert result.memories_migrated == 1  # counted
        assert len(chroma_dest.list_recent()) == 0  # not written

    def test_preserves_layer(self, sqlite_source, chroma_dest):
        sqlite_source.store(make_memory("rule", MemoryLayer.PROCEDURAL))
        Migrator().run(sqlite_source, chroma_dest)
        memories = chroma_dest.list_recent(layer=MemoryLayer.PROCEDURAL)
        assert len(memories) == 1
        assert memories[0].layer == MemoryLayer.PROCEDURAL


class TestReembedding:
    def test_reembed_when_embedders_differ(self, sqlite_source, chroma_dest):
        sqlite_source.store(make_memory("some content", MemoryLayer.SEMANTIC))

        source_emb = MagicMock()
        source_emb._model_name = "all-MiniLM-L6-v2"
        source_emb.dim = 384

        dest_emb = MagicMock()
        dest_emb._model_name = "text-embedding-3-small"
        dest_emb.embed_batch.return_value = [[0.2] * 384]
        dest_emb.dim = 384

        result = Migrator().run(sqlite_source, chroma_dest, source_emb, dest_emb)
        assert result.re_embedded == 1
        dest_emb.embed_batch.assert_called_once()

    def test_no_reembed_when_same_model(self, sqlite_source, chroma_dest):
        sqlite_source.store(make_memory("some content", MemoryLayer.SEMANTIC))

        emb = MagicMock()
        emb._model_name = "all-MiniLM-L6-v2"
        emb.dim = 384

        result = Migrator().run(sqlite_source, chroma_dest, emb, emb)
        assert result.re_embedded == 0


class TestRoundTrip:
    def test_sqlite_to_chroma_to_sqlite(self, tmp_path, tmp_config):
        from extremis.storage.chroma import ChromaMemoryStore

        source = SQLiteMemoryStore(str(tmp_path / "source.db"), tmp_config)
        mid = ChromaMemoryStore(str(tmp_path / "chroma"), tmp_config)
        dest = SQLiteMemoryStore(str(tmp_path / "dest.db"), tmp_config)

        for i in range(3):
            source.store(make_memory(f"memory {i}", MemoryLayer.SEMANTIC))

        Migrator().run(source, mid)
        Migrator().run(mid, dest)

        dest_memories = dest.list_recent(layer=MemoryLayer.SEMANTIC)
        assert len(dest_memories) == 3
        contents = {m.content for m in dest_memories}
        assert contents == {"memory 0", "memory 1", "memory 2"}

        mid.close()
