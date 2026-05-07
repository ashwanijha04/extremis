"""Namespace isolation — two namespaces on the same DB must not see each other's data."""
from __future__ import annotations

import pytest

from extremis.api import Memory
from extremis.config import Config
from extremis.storage.kg import SQLiteKGStore
from extremis.types import EntityType, MemoryLayer
from .conftest import make_memory


@pytest.fixture
def ns_a(tmp_path, mock_embedder):
    cfg = Config(
        extremis_home=str(tmp_path),
        log_dir=str(tmp_path / "log"),
        local_db_path=str(tmp_path / "shared.db"),
        namespace="alice",
    )
    return Memory(config=cfg, embedder=mock_embedder)


@pytest.fixture
def ns_b(tmp_path, mock_embedder):
    cfg = Config(
        extremis_home=str(tmp_path),
        log_dir=str(tmp_path / "log"),
        local_db_path=str(tmp_path / "shared.db"),
        namespace="bob",
    )
    return Memory(config=cfg, embedder=mock_embedder)


class TestMemoryNamespaceIsolation:
    def test_alice_memory_not_visible_to_bob(self, ns_a, ns_b):
        ns_a.remember("Alice's secret project", conversation_id="a1")
        results = ns_b.recall("secret project", limit=10)
        assert not any("Alice" in r.memory.content for r in results)

    def test_bob_memory_not_visible_to_alice(self, ns_a, ns_b):
        ns_b.remember("Bob's secret project", conversation_id="b1")
        results = ns_a.recall("secret project", limit=10)
        assert not any("Bob" in r.memory.content for r in results)

    def test_each_namespace_only_sees_own_data(self, ns_a, ns_b):
        ns_a.remember("Alice fact", conversation_id="a1")
        ns_b.remember("Bob fact", conversation_id="b1")

        alice_results = ns_a.recall("fact", limit=10)
        bob_results = ns_b.recall("fact", limit=10)

        alice_contents = {r.memory.content for r in alice_results}
        bob_contents = {r.memory.content for r in bob_results}

        assert "Alice fact" in alice_contents
        assert "Bob fact" not in alice_contents
        assert "Bob fact" in bob_contents
        assert "Alice fact" not in bob_contents

    def test_scores_isolated(self, ns_a, ns_b):
        mem_a = ns_a.remember_now("Shared topic", layer=MemoryLayer.SEMANTIC)
        mem_b = ns_b.remember_now("Shared topic", layer=MemoryLayer.SEMANTIC)

        ns_a.report_outcome([mem_a.id], success=True, weight=5.0)

        # Bob's memory is untouched
        bob_mem = ns_b.get_local_store().get(mem_b.id)
        assert bob_mem.score == pytest.approx(0.0)

    def test_remember_now_isolated(self, ns_a, ns_b):
        ns_a.remember_now("Alice procedural rule", layer=MemoryLayer.PROCEDURAL)
        bob_results = ns_b.recall("Alice", limit=10)
        assert not any("Alice procedural" in r.memory.content for r in bob_results)


class TestLogNamespaceIsolation:
    def test_logs_in_separate_directories(self, tmp_path, mock_embedder):
        from extremis.storage.log import FileLogStore
        from extremis.types import LogEntry

        cfg_a = Config(
            extremis_home=str(tmp_path),
            log_dir=str(tmp_path / "log"),
            local_db_path=str(tmp_path / "shared.db"),
            namespace="alice",
        )
        cfg_b = cfg_a.model_copy(update={"namespace": "bob"})

        log_a = FileLogStore(cfg_a.resolved_log_dir(), namespace="alice")
        log_b = FileLogStore(cfg_b.resolved_log_dir(), namespace="bob")

        log_a.append(LogEntry(role="user", content="alice log", conversation_id="a1"))
        log_b.append(LogEntry(role="user", content="bob log", conversation_id="b1"))

        alice_entries = log_a.read_since(None)
        bob_entries = log_b.read_since(None)

        assert not any("bob log" in e.content for e in alice_entries)
        assert not any("alice log" in e.content for e in bob_entries)


class TestKGNamespaceIsolation:
    def test_entities_isolated_by_namespace(self, tmp_path):
        from extremis.config import Config

        cfg_a = Config(
            extremis_home=str(tmp_path),
            local_db_path=str(tmp_path / "shared.db"),
            namespace="alice",
        )
        cfg_b = cfg_a.model_copy(update={"namespace": "bob"})

        kg_a = SQLiteKGStore(str(tmp_path / "shared.db"), cfg_a)
        kg_b = SQLiteKGStore(str(tmp_path / "shared.db"), cfg_b)

        kg_a.add_entity("Alice Corp", EntityType.ORG)
        assert kg_b.query_entity("Alice Corp") is None

    def test_relationships_isolated(self, tmp_path):
        from extremis.config import Config

        cfg_a = Config(
            extremis_home=str(tmp_path),
            local_db_path=str(tmp_path / "shared.db"),
            namespace="alice",
        )
        cfg_b = cfg_a.model_copy(update={"namespace": "bob"})

        kg_a = SQLiteKGStore(str(tmp_path / "shared.db"), cfg_a)
        kg_b = SQLiteKGStore(str(tmp_path / "shared.db"), cfg_b)

        kg_a.add_entity("Alice", EntityType.PERSON)
        kg_a.add_entity("Corp", EntityType.ORG)
        kg_a.add_relationship("Alice", "Corp", "works_at")

        result = kg_b.query_entity("Alice")
        assert result is None
