"""ChromaMemoryStore — adapter tests (requires chromadb installed)."""

from __future__ import annotations

import pytest

from extremis.storage.chroma import ChromaMemoryStore
from extremis.types import MemoryLayer

from .conftest import make_memory

pytest.importorskip("chromadb", reason="chromadb not installed")


@pytest.fixture
def chroma_store(tmp_path, tmp_config):
    store = ChromaMemoryStore(str(tmp_path / "chroma"), tmp_config)
    yield store
    store.close()


class TestChromaStore:
    def test_store_and_get(self, chroma_store):
        mem = make_memory("User is a Python developer", MemoryLayer.SEMANTIC)
        chroma_store.store(mem)
        retrieved = chroma_store.get(mem.id)
        assert retrieved is not None
        assert retrieved.content == "User is a Python developer"
        assert retrieved.layer == MemoryLayer.SEMANTIC

    def test_get_unknown_returns_none(self, chroma_store):
        from uuid import uuid4

        assert chroma_store.get(uuid4()) is None

    def test_search_returns_results(self, chroma_store):
        chroma_store.store(make_memory("User prefers dark mode", MemoryLayer.SEMANTIC))
        chroma_store.store(make_memory("User lives in Dubai", MemoryLayer.SEMANTIC))
        results = chroma_store.search([0.1] * 384, limit=5)
        assert len(results) == 2
        assert all(r.relevance >= 0 for r in results)

    def test_search_layer_filter(self, chroma_store):
        chroma_store.store(make_memory("semantic fact", MemoryLayer.SEMANTIC))
        chroma_store.store(make_memory("procedural rule", MemoryLayer.PROCEDURAL))
        results = chroma_store.search([0.1] * 384, layers=[MemoryLayer.SEMANTIC])
        assert all(r.memory.layer == MemoryLayer.SEMANTIC for r in results)

    def test_update_score_via_score_index(self, chroma_store):
        mem = make_memory("Important rule")
        chroma_store.store(mem)
        chroma_store.update_score(mem.id, 3.0)
        retrieved = chroma_store.get(mem.id)
        assert retrieved.score == pytest.approx(3.0)

    def test_supersede_closes_old(self, chroma_store):
        old = make_memory("User works at Company A", MemoryLayer.SEMANTIC)
        chroma_store.store(old)
        new = make_memory("User works at Company B", MemoryLayer.SEMANTIC)
        chroma_store.supersede(old.id, new)

        old_retrieved = chroma_store.get(old.id)
        assert old_retrieved.validity_end is not None

    def test_supersede_new_visible_in_search(self, chroma_store):
        old = make_memory("User at A", MemoryLayer.SEMANTIC)
        chroma_store.store(old)
        new = make_memory("User at B", MemoryLayer.SEMANTIC)
        chroma_store.supersede(old.id, new)

        current = chroma_store.list_recent(layer=MemoryLayer.SEMANTIC)
        contents = [m.content for m in current]
        assert "User at B" in contents
        assert "User at A" not in contents

    def test_list_recent(self, chroma_store):
        for i in range(5):
            chroma_store.store(make_memory(f"fact {i}", MemoryLayer.SEMANTIC))
        results = chroma_store.list_recent(layer=MemoryLayer.SEMANTIC, limit=3)
        assert len(results) == 3

    def test_namespace_isolation(self, tmp_path):
        from extremis.config import Config

        cfg_a = Config(
            extremis_home=str(tmp_path),
            local_db_path=str(tmp_path / "local.db"),
            namespace="alice",
        )
        cfg_b = cfg_a.model_copy(update={"namespace": "bob"})

        store_a = ChromaMemoryStore(str(tmp_path / "chroma"), cfg_a)
        store_b = ChromaMemoryStore(str(tmp_path / "chroma"), cfg_b)

        store_a.store(make_memory("Alice's secret", MemoryLayer.SEMANTIC, namespace="alice"))

        results = store_b.search([0.1] * 384, limit=10)
        assert not any("Alice" in r.memory.content for r in results)

        store_a.close()
        store_b.close()
