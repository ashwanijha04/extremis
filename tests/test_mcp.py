"""MCP server tool tests — no network, no LLM. Uses shared conftest fixtures."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lore_ai.types import MemoryLayer


class TestMCPTools:
    """Verify the logic the MCP tool functions delegate to."""

    def test_remember_and_recall(self, api):
        api.remember("User is building a WhatsApp AI", conversation_id="c1")
        results = api.recall("WhatsApp", limit=5)
        assert any("WhatsApp" in r.memory.content for r in results)

    def test_remember_now_direct_write(self, api):
        mem = api.remember_now(
            "User's flight departs Thursday at 06:00",
            layer=MemoryLayer.EPISODIC,
            confidence=0.99,
        )
        assert mem.layer == MemoryLayer.EPISODIC
        assert mem.confidence == pytest.approx(0.99)
        assert api.get_local_store().get(mem.id) is not None

    def test_report_positive_outcome_adjusts_score(self, api):
        mem = api.remember_now("Prefer concise answers", layer=MemoryLayer.PROCEDURAL)
        api.report_outcome([mem.id], success=True, weight=2.0)
        updated = api.get_local_store().get(mem.id)
        assert updated.score == pytest.approx(2.0)

    def test_report_negative_outcome_has_asymmetric_weight(self, api):
        """Negative signals use 1.5× multiplier — not −1.0 but −1.5."""
        mem = api.remember_now("Bad approach", layer=MemoryLayer.PROCEDURAL)
        api.report_outcome([mem.id], success=False, weight=1.0)
        updated = api.get_local_store().get(mem.id)
        assert updated.score == pytest.approx(-1.5)

    def test_procedural_returned_regardless_of_query(self, api):
        api.remember_now("Always check deadline before suggesting", layer=MemoryLayer.PROCEDURAL)
        api.remember("Some unrelated episodic fact", conversation_id="c2")
        results = api.recall("weather in Dubai", limit=5)
        assert MemoryLayer.PROCEDURAL in [r.memory.layer for r in results]

    def test_remember_writes_to_log(self, api, tmp_config):
        from lore_ai.storage.log import FileLogStore
        api.remember("User mentioned they hate meetings", conversation_id="c3")
        log = FileLogStore(tmp_config.resolved_log_dir(), namespace=tmp_config.namespace)
        entries = log.read_since(None)
        assert any("hate meetings" in e.content for e in entries)

    def test_remember_now_with_expiry_stored(self, api):
        from datetime import timedelta
        expiry = datetime.now(tz=timezone.utc) + timedelta(hours=2)
        mem = api.remember_now("User is in a meeting until 3pm", layer=MemoryLayer.WORKING, expires_at=expiry)
        assert mem.validity_end is not None

    def test_kg_roundtrip(self, api):
        from lore_ai.types import EntityType
        api.kg_add_entity("Alice", EntityType.PERSON)
        api.kg_add_attribute("Alice", "timezone", "Asia/Dubai")
        result = api.kg_query("Alice")
        assert result is not None
        assert result.entity.name == "Alice"
        attr_map = {a.key: a.value for a in result.attributes}
        assert attr_map["timezone"] == "Asia/Dubai"

    def test_attention_scorer_via_api(self, api):
        result = api.score_attention("URGENT the server is down!", channel="dm")
        assert result.score > 50
        assert result.level in ("full", "standard")

    def test_observe_compresses_log(self, api):
        api.remember("We decided to launch tomorrow", conversation_id="obs_test")
        api.remember("The build failed with an error", conversation_id="obs_test")
        api.remember("Nice weather today", conversation_id="obs_test")

        observations = api.observe("obs_test")
        priorities = {o.priority.value for o in observations}
        assert "critical" in priorities
