"""Smoke tests — no LLM calls, no network. Uses a temp SQLite DB."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from friday_memory.config import Config
from friday_memory.storage.log import FileLogStore
from friday_memory.storage.sqlite import SQLiteMemoryStore
from friday_memory.types import LogEntry, Memory, MemoryLayer


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def config(tmp_dir):
    cfg = Config(
        friday_home=str(tmp_dir),
        log_dir=str(tmp_dir / "log"),
        local_db_path=str(tmp_dir / "local.db"),
        embedder="all-MiniLM-L6-v2",
    )
    return cfg


@pytest.fixture
def log(config):
    return FileLogStore(config.resolved_log_dir())


@pytest.fixture
def store(config):
    return SQLiteMemoryStore(config.resolved_local_db_path(), config)


def make_memory(content: str, layer: MemoryLayer = MemoryLayer.EPISODIC) -> Memory:
    return Memory(
        layer=layer,
        content=content,
        embedding=[0.1] * 384,
        validity_start=datetime.now(tz=timezone.utc),
    )


class TestFileLogStore:
    def test_append_and_read(self, log):
        entry = LogEntry(
            role="user",
            content="Hello friday",
            conversation_id="conv1",
        )
        log.append(entry)

        entries = log.read_since(None)
        assert len(entries) == 1
        assert entries[0].content == "Hello friday"

    def test_checkpoint_round_trip(self, log):
        entry = LogEntry(role="assistant", content="Hi!", conversation_id="conv1")
        log.append(entry)

        cp = log.current_checkpoint()
        log.set_checkpoint(cp)

        assert log.get_checkpoint() == cp

    def test_read_since_checkpoint(self, log):
        e1 = LogEntry(role="user", content="msg1", conversation_id="c1")
        log.append(e1)
        cp = log.current_checkpoint()
        log.set_checkpoint(cp)

        e2 = LogEntry(role="user", content="msg2", conversation_id="c1")
        log.append(e2)

        new_entries = log.read_since(cp)
        assert len(new_entries) == 1
        assert new_entries[0].content == "msg2"


class TestSQLiteMemoryStore:
    def test_store_and_get(self, store):
        mem = make_memory("User is a Python developer")
        store.store(mem)
        retrieved = store.get(mem.id)
        assert retrieved is not None
        assert retrieved.content == "User is a Python developer"

    def test_search_returns_ranked_results(self, store):
        m1 = make_memory("User prefers dark mode", MemoryLayer.SEMANTIC)
        m2 = make_memory("User lives in Dubai", MemoryLayer.SEMANTIC)
        store.store(m1)
        store.store(m2)

        # Use same embedding vector to ensure deterministic cosine match
        results = store.search(query_embedding=[0.1] * 384, limit=5)
        assert len(results) == 2
        for r in results:
            assert r.relevance > 0

    def test_update_score(self, store):
        mem = make_memory("Important rule")
        store.store(mem)
        store.update_score(mem.id, 2.0)
        updated = store.get(mem.id)
        assert updated is not None
        assert updated.score == 2.0

    def test_supersede(self, store):
        old = make_memory("User works at Company A", MemoryLayer.SEMANTIC)
        store.store(old)

        new = make_memory("User works at Company B", MemoryLayer.SEMANTIC)
        store.supersede(old.id, new)

        old_retrieved = store.get(old.id)
        assert old_retrieved is not None
        assert old_retrieved.validity_end is not None  # closed

        current = store.list_recent(layer=MemoryLayer.SEMANTIC)
        contents = [m.content for m in current]
        assert "User works at Company B" in contents
        assert "User works at Company A" not in contents

    def test_list_recent(self, store):
        for i in range(5):
            store.store(make_memory(f"fact {i}", MemoryLayer.SEMANTIC))
        results = store.list_recent(layer=MemoryLayer.SEMANTIC, limit=3)
        assert len(results) == 3
