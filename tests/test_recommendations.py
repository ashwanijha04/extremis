"""Tests for the verification recommendations layer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from extremis.types import Memory, MemoryLayer
from extremis.verification.recommendations import (
    recommend_for_recall,
    recommend_for_verification,
)


def _now():
    return datetime.now(tz=timezone.utc)


def _memory(
    *,
    layer=MemoryLayer.SEMANTIC,
    confidence=0.9,
    metadata=None,
    age_days=0,
    validity_end=None,
) -> Memory:
    created = _now() - timedelta(days=age_days)
    return Memory(
        layer=layer,
        content="test",
        embedding=[0.1] * 384,
        confidence=confidence,
        metadata=metadata or {},
        validity_start=created,
        validity_end=validity_end,
        created_at=created,
    )


class TestVerificationRecommendations:
    def test_contradicted_returns_high_severity_action(self):
        verification = {
            "verdict": "CONTRADICTED",
            "score": 0.1,
            "method": "nli",
            "judge_reason": "",
        }
        recs = recommend_for_verification(
            verification,
            memory_refs={"conversation_id": "c1", "source_message_ids": ["m1", "m2"]},
        )
        assert len(recs) == 1
        rec = recs[0]
        assert rec.issue == "claim_contradicts_source"
        assert rec.severity == "high"
        assert "source_message_ids" in rec.action
        assert rec.refs["source_message_ids"] == ["m1", "m2"]
        assert rec.refs["conversation_id"] == "c1"
        assert rec.suggestion  # systemic fix included

    def test_unverifiable_returns_medium_severity(self):
        verification = {"verdict": "UNVERIFIABLE", "score": 0.4, "method": "nli+judge"}
        recs = recommend_for_verification(verification, memory_refs={"conversation_id": "c1"})
        assert len(recs) == 1
        assert recs[0].issue == "claim_unverifiable"
        assert recs[0].severity == "medium"
        assert recs[0].refs["method"] == "nli+judge"

    def test_borderline_support_returns_low_severity(self):
        verification = {"verdict": "SUPPORTED", "score": 0.65, "method": "nli"}
        recs = recommend_for_verification(verification, memory_refs={})
        assert len(recs) == 1
        assert recs[0].issue == "borderline_support"
        assert recs[0].severity == "low"

    def test_clean_support_returns_no_recommendations(self):
        verification = {"verdict": "SUPPORTED", "score": 0.95, "method": "nli"}
        recs = recommend_for_verification(verification, memory_refs={})
        assert recs == []

    def test_judge_reason_propagated_into_refs(self):
        verification = {
            "verdict": "CONTRADICTED",
            "score": 0.1,
            "method": "nli+judge",
            "judge_reason": "Says X but source says Y",
        }
        recs = recommend_for_verification(verification, memory_refs={})
        assert recs[0].refs["judge_reason"] == "Says X but source says Y"


class TestRecallRecommendations:
    def test_expired_memory_high_severity(self):
        past = _now() - timedelta(hours=1)
        memory = _memory(validity_end=past)
        recs = recommend_for_recall(memory, effective_confidence=0.0)
        issues = [r.issue for r in recs]
        assert "memory_expired" in issues
        expired = next(r for r in recs if r.issue == "memory_expired")
        assert expired.severity == "high"
        assert "validity_end" in expired.refs

    def test_contradicted_memory_still_surfacing(self):
        memory = _memory(
            metadata={
                "verification": {
                    "verdict": "CONTRADICTED",
                    "score": 0.1,
                    "method": "nli",
                }
            }
        )
        recs = recommend_for_recall(memory, effective_confidence=0.5)
        issues = [r.issue for r in recs]
        assert "surfacing_contradicted_memory" in issues
        rec = next(r for r in recs if r.issue == "surfacing_contradicted_memory")
        assert rec.severity == "high"
        assert rec.refs["verification_method"] == "nli"

    def test_stale_confidence_includes_age_hint(self):
        memory = _memory(confidence=0.5, age_days=365)
        # effective_confidence below 0.3 triggers stale
        recs = recommend_for_recall(memory, effective_confidence=0.1)
        issues = [r.issue for r in recs]
        assert "stale_confidence" in issues
        stale = next(r for r in recs if r.issue == "stale_confidence")
        assert stale.severity == "medium"
        assert "as of" in stale.action.lower()
        assert "age_days" in stale.refs

    def test_fresh_high_confidence_yields_no_recommendations(self):
        memory = _memory(confidence=0.95)
        recs = recommend_for_recall(memory, effective_confidence=0.9)
        assert recs == []

    def test_zero_effective_confidence_not_flagged_as_stale(self):
        """effective_confidence == 0 means expired; stale check must not double-fire."""
        past = _now() - timedelta(days=1)
        memory = _memory(validity_end=past)
        recs = recommend_for_recall(memory, effective_confidence=0.0)
        issues = [r.issue for r in recs]
        assert "memory_expired" in issues
        # No stale_confidence because effective_confidence is exactly 0
        assert "stale_confidence" not in issues

    def test_refs_include_memory_id(self):
        memory_id = uuid4()
        memory = Memory(
            id=memory_id,
            layer=MemoryLayer.SEMANTIC,
            content="x",
            embedding=[0.1] * 384,
            confidence=0.5,
            metadata={"verification": {"verdict": "CONTRADICTED", "score": 0.1, "method": "nli"}},
            validity_start=_now(),
            created_at=_now(),
        )
        recs = recommend_for_recall(memory, effective_confidence=0.5)
        assert recs[0].refs["memory_id"] == str(memory_id)
