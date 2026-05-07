"""Memory public API — remember / recall / report_outcome / remember_now."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from extremis.storage.log import FileLogStore
from extremis.types import MemoryLayer


class TestRemember:
    def test_remember_creates_episodic_memory(self, api):
        api.remember("User likes Python", conversation_id="c1")
        results = api.recall("Python", limit=5)
        assert any("Python" in r.memory.content for r in results)

    def test_remember_writes_to_log(self, api, tmp_config):
        api.remember("User hates meetings", conversation_id="c2")
        log = FileLogStore(tmp_config.resolved_log_dir(), namespace=tmp_config.namespace)
        entries = log.read_since(None)
        assert any("meetings" in e.content for e in entries)

    def test_remember_sets_role(self, api, tmp_config):
        api.remember("assistant said something", role="assistant", conversation_id="c3")
        log = FileLogStore(tmp_config.resolved_log_dir(), namespace=tmp_config.namespace)
        entries = log.read_since(None)
        assert any(e.role == "assistant" for e in entries)


class TestRecall:
    def test_procedural_always_returned(self, api):
        api.remember_now("Always check deadline first", layer=MemoryLayer.PROCEDURAL)
        api.remember("Some completely unrelated fact", conversation_id="c1")

        results = api.recall("weather forecast tomorrow", limit=10)
        layers = [r.memory.layer for r in results]
        assert MemoryLayer.PROCEDURAL in layers

    def test_identity_always_returned(self, api):
        api.remember_now("User is a software engineer", layer=MemoryLayer.IDENTITY, confidence=1.0)
        results = api.recall("random unrelated query", limit=10)
        layers = [r.memory.layer for r in results]
        assert MemoryLayer.IDENTITY in layers

    def test_results_deduplicated(self, api):
        api.remember_now("Single fact", layer=MemoryLayer.PROCEDURAL)
        results = api.recall("fact", limit=10)
        ids = [r.memory.id for r in results]
        assert len(ids) == len(set(ids))

    def test_recall_respects_limit(self, api):
        for i in range(20):
            api.remember(f"fact number {i}", conversation_id="c1")
        results = api.recall("fact", limit=5)
        assert len(results) <= 5

    def test_layer_filter_passed_through(self, api):
        api.remember_now("semantic thing", layer=MemoryLayer.SEMANTIC)
        api.remember("episodic thing", conversation_id="c1")
        results = api.recall("thing", layers=[MemoryLayer.EPISODIC])
        non_pinned = [r for r in results if r.memory.layer not in (MemoryLayer.IDENTITY, MemoryLayer.PROCEDURAL)]
        assert all(r.memory.layer == MemoryLayer.EPISODIC for r in non_pinned)


class TestReportOutcome:
    def test_positive_signal_increases_score(self, api):
        mem = api.remember_now("Concise answers work", layer=MemoryLayer.PROCEDURAL)
        api.report_outcome([mem.id], success=True, weight=2.0)
        updated = api.get_local_store().get(mem.id)
        assert updated.score == pytest.approx(2.0)

    def test_negative_signal_has_asymmetric_weight(self, api):
        """Negative signals apply 1.5× weight (matches human memory asymmetry)."""
        mem = api.remember_now("Bad approach", layer=MemoryLayer.PROCEDURAL)
        api.report_outcome([mem.id], success=False, weight=1.0)
        updated = api.get_local_store().get(mem.id)
        assert updated.score == pytest.approx(-1.5)

    def test_negative_weight_two(self, api):
        mem = api.remember_now("Another bad approach", layer=MemoryLayer.PROCEDURAL)
        api.report_outcome([mem.id], success=False, weight=2.0)
        updated = api.get_local_store().get(mem.id)
        assert updated.score == pytest.approx(-3.0)

    def test_multiple_ids_all_updated(self, api):
        m1 = api.remember_now("fact 1", layer=MemoryLayer.SEMANTIC)
        m2 = api.remember_now("fact 2", layer=MemoryLayer.SEMANTIC)
        api.report_outcome([m1.id, m2.id], success=True, weight=1.0)
        assert api.get_local_store().get(m1.id).score == pytest.approx(1.0)
        assert api.get_local_store().get(m2.id).score == pytest.approx(1.0)


class TestRememberNow:
    def test_writes_to_correct_layer(self, api):
        mem = api.remember_now("Procedural rule", layer=MemoryLayer.PROCEDURAL)
        assert mem.layer == MemoryLayer.PROCEDURAL

    def test_confidence_stored(self, api):
        mem = api.remember_now("High confidence fact", layer=MemoryLayer.SEMANTIC, confidence=0.95)
        retrieved = api.get_local_store().get(mem.id)
        assert retrieved.confidence == pytest.approx(0.95)

    def test_expiry_stored(self, api):
        expiry = datetime.now(tz=timezone.utc) + timedelta(hours=2)
        mem = api.remember_now("Temp fact", layer=MemoryLayer.WORKING, expires_at=expiry)
        retrieved = api.get_local_store().get(mem.id)
        assert retrieved.validity_end is not None

    def test_does_not_write_to_log(self, api, tmp_config):
        api.remember_now("Direct write, skip log", layer=MemoryLayer.SEMANTIC)
        log = FileLogStore(tmp_config.resolved_log_dir(), namespace=tmp_config.namespace)
        entries = log.read_since(None)
        assert not any("Direct write" in e.content for e in entries)
