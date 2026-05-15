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


class TestRecallEffectiveConfidence:
    """Verify the hedging signal exposed on every RecallResult."""

    def test_effective_confidence_populated(self, api):
        api.remember_now("identity fact", layer=MemoryLayer.IDENTITY, confidence=1.0)
        results = api.recall("identity", limit=5)
        assert results
        assert results[0].effective_confidence is not None

    def test_identity_outweighs_episodic_at_same_age(self, api):
        api.remember_now("durable identity", layer=MemoryLayer.IDENTITY, confidence=1.0)
        api.remember("transient episodic", conversation_id="c1")
        results = api.recall("identity", limit=10)

        by_layer = {r.memory.layer: r.effective_confidence for r in results}
        # IDENTITY weight 0.95 vs EPISODIC weight 0.60 with both fresh
        if MemoryLayer.IDENTITY in by_layer and MemoryLayer.EPISODIC in by_layer:
            assert by_layer[MemoryLayer.IDENTITY] > by_layer[MemoryLayer.EPISODIC]

    def test_expired_memory_decays_to_zero(self, api):
        past = datetime.now(tz=timezone.utc) - timedelta(hours=1)
        api.remember_now("expired", layer=MemoryLayer.WORKING, expires_at=past)
        results = api.recall("expired", limit=10)
        for r in results:
            if r.memory.content == "expired":
                assert r.effective_confidence == 0.0


class TestRecallSources:
    """Every RecallResult carries a structured provenance trail."""

    def test_sources_attached(self, api):
        mem = api.remember_now("fact with source", layer=MemoryLayer.SEMANTIC)
        results = api.recall("fact", limit=5)
        match = next((r for r in results if r.memory.id == mem.id), None)
        assert match is not None
        assert match.sources is not None
        assert match.sources["layer"] == MemoryLayer.SEMANTIC.value
        assert "source_memory_ids" in match.sources
        assert "created_at" in match.sources

    def test_sources_exposes_conversation_id_for_episodic(self, api):
        api.remember("episodic from conv", conversation_id="conv-xyz")
        results = api.recall("episodic", limit=5)
        for r in results:
            if r.memory.layer == MemoryLayer.EPISODIC and "episodic from conv" in r.memory.content:
                assert r.sources["conversation_id"] == "conv-xyz"
                return
        pytest.fail("episodic memory not surfaced in recall")

    def test_contradicted_memory_recall_carries_recommendation(self, api):
        """If a memory was tagged CONTRADICTED at write time, recall flags it."""
        mem = api.remember_now(
            "questionable fact",
            layer=MemoryLayer.SEMANTIC,
            metadata={
                "verification": {
                    "verdict": "CONTRADICTED",
                    "score": 0.1,
                    "method": "nli",
                }
            },
        )
        results = api.recall("questionable", limit=10)
        match = next((r for r in results if r.memory.id == mem.id), None)
        assert match is not None
        issues = [rec["issue"] for rec in (match.sources.get("recommendations") or [])]
        assert "surfacing_contradicted_memory" in issues


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
