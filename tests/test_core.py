"""FileLogStore + SQLiteMemoryStore — core storage tests."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lore_ai.types import LogEntry, MemoryLayer
from .conftest import make_memory


class TestFileLogStore:
    def test_append_and_read_all(self, log_store):
        log_store.append(LogEntry(role="user", content="Hello friday", conversation_id="c1"))
        entries = log_store.read_since(None)
        assert len(entries) == 1
        assert entries[0].content == "Hello friday"

    def test_multiple_entries_preserved(self, log_store):
        for i in range(5):
            log_store.append(LogEntry(role="user", content=f"message {i}", conversation_id="c1"))
        assert len(log_store.read_since(None)) == 5

    def test_checkpoint_round_trip(self, log_store):
        log_store.append(LogEntry(role="assistant", content="Hi!", conversation_id="c1"))
        cp = log_store.current_checkpoint()
        log_store.set_checkpoint(cp)
        assert log_store.get_checkpoint() == cp

    def test_get_checkpoint_returns_none_initially(self, log_store):
        assert log_store.get_checkpoint() is None

    def test_read_since_checkpoint_only_returns_new(self, log_store):
        log_store.append(LogEntry(role="user", content="old msg", conversation_id="c1"))
        cp = log_store.current_checkpoint()
        log_store.set_checkpoint(cp)

        log_store.append(LogEntry(role="user", content="new msg", conversation_id="c1"))
        new_entries = log_store.read_since(cp)

        assert len(new_entries) == 1
        assert new_entries[0].content == "new msg"

    def test_read_since_none_returns_all(self, log_store):
        for msg in ["a", "b", "c"]:
            log_store.append(LogEntry(role="user", content=msg, conversation_id="c1"))
        assert len(log_store.read_since(None)) == 3

    def test_entries_survive_reopen(self, tmp_config):
        from lore_ai.storage.log import FileLogStore
        store1 = FileLogStore(tmp_config.resolved_log_dir(), namespace=tmp_config.namespace)
        store1.append(LogEntry(role="user", content="persisted", conversation_id="c1"))

        store2 = FileLogStore(tmp_config.resolved_log_dir(), namespace=tmp_config.namespace)
        entries = store2.read_since(None)
        assert any(e.content == "persisted" for e in entries)


class TestSQLiteMemoryStore:
    def test_store_and_retrieve(self, mem_store):
        mem = make_memory("User is a Python developer")
        mem_store.store(mem)
        retrieved = mem_store.get(mem.id)
        assert retrieved is not None
        assert retrieved.content == "User is a Python developer"

    def test_get_unknown_id_returns_none(self, mem_store):
        from uuid import uuid4
        assert mem_store.get(uuid4()) is None

    def test_search_returns_results(self, mem_store):
        mem_store.store(make_memory("User prefers dark mode", MemoryLayer.SEMANTIC))
        mem_store.store(make_memory("User lives in Dubai", MemoryLayer.SEMANTIC))
        results = mem_store.search(query_embedding=[0.1] * 384, limit=5)
        assert len(results) == 2
        assert all(r.relevance > 0 for r in results)

    def test_search_layer_filter(self, mem_store):
        mem_store.store(make_memory("semantic fact", MemoryLayer.SEMANTIC))
        mem_store.store(make_memory("procedural rule", MemoryLayer.PROCEDURAL))
        results = mem_store.search([0.1] * 384, layers=[MemoryLayer.SEMANTIC])
        assert all(r.memory.layer == MemoryLayer.SEMANTIC for r in results)
        assert len(results) == 1

    def test_search_respects_limit(self, mem_store):
        for i in range(10):
            mem_store.store(make_memory(f"fact {i}", MemoryLayer.SEMANTIC))
        results = mem_store.search([0.1] * 384, limit=3)
        assert len(results) == 3

    def test_update_score(self, mem_store):
        mem = make_memory("Important rule")
        mem_store.store(mem)
        mem_store.update_score(mem.id, 3.5)
        assert mem_store.get(mem.id).score == pytest.approx(3.5)

    def test_update_score_accumulates(self, mem_store):
        mem = make_memory("Rule")
        mem_store.store(mem)
        mem_store.update_score(mem.id, 1.0)
        mem_store.update_score(mem.id, 1.0)
        assert mem_store.get(mem.id).score == pytest.approx(2.0)

    def test_supersede_closes_old_memory(self, mem_store):
        old = make_memory("User works at Company A", MemoryLayer.SEMANTIC)
        mem_store.store(old)

        new = make_memory("User works at Company B", MemoryLayer.SEMANTIC)
        mem_store.supersede(old.id, new)

        old_retrieved = mem_store.get(old.id)
        assert old_retrieved.validity_end is not None

    def test_supersede_new_memory_visible(self, mem_store):
        old = make_memory("User works at Company A", MemoryLayer.SEMANTIC)
        mem_store.store(old)
        new = make_memory("User works at Company B", MemoryLayer.SEMANTIC)
        mem_store.supersede(old.id, new)

        current = mem_store.list_recent(layer=MemoryLayer.SEMANTIC)
        contents = [m.content for m in current]
        assert "User works at Company B" in contents
        assert "User works at Company A" not in contents

    def test_list_recent_layer_filter(self, mem_store):
        mem_store.store(make_memory("semantic fact", MemoryLayer.SEMANTIC))
        mem_store.store(make_memory("procedural rule", MemoryLayer.PROCEDURAL))
        results = mem_store.list_recent(layer=MemoryLayer.SEMANTIC)
        assert all(m.layer == MemoryLayer.SEMANTIC for m in results)

    def test_list_recent_limit(self, mem_store):
        for i in range(10):
            mem_store.store(make_memory(f"fact {i}", MemoryLayer.SEMANTIC))
        assert len(mem_store.list_recent(limit=4)) == 4

    def test_memories_survive_reopen(self, tmp_config):
        from lore_ai.storage.sqlite import SQLiteMemoryStore
        s1 = SQLiteMemoryStore(tmp_config.resolved_local_db_path(), tmp_config)
        mem = make_memory("persisted content")
        s1.store(mem)

        s2 = SQLiteMemoryStore(tmp_config.resolved_local_db_path(), tmp_config)
        retrieved = s2.get(mem.id)
        assert retrieved is not None
        assert retrieved.content == "persisted content"
