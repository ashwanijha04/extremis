"""AttentionScorer — heuristic message priority tests."""

from __future__ import annotations

import pytest

from extremis.config import Config
from extremis.scorer.attention import AttentionScorer


@pytest.fixture
def scorer(tmp_config):
    return AttentionScorer(tmp_config)


@pytest.fixture
def strict_scorer():
    cfg = Config(
        attention_full_threshold=75,
        attention_standard_threshold=50,
        attention_minimal_threshold=25,
    )
    return AttentionScorer(cfg)


class TestOwnerOverride:
    def test_owner_always_gets_full(self, scorer):
        result = scorer.score("hi", sender="owner_1", owner_ids={"owner_1"})
        assert result.score == 100
        assert result.level == "full"

    def test_owner_override_ignores_channel(self, scorer):
        result = scorer.score("hi", sender="owner_1", channel="broadcast", owner_ids={"owner_1"})
        assert result.level == "full"

    def test_non_owner_not_overridden(self, scorer):
        result = scorer.score("hi", sender="stranger", owner_ids={"owner_1"})
        assert result.score < 100


class TestLevels:
    def test_unknown_sender_dm_no_content_is_standard_or_below(self, scorer):
        result = scorer.score("hello", sender="unknown", channel="dm")
        assert result.level in ("standard", "minimal", "ignore")

    def test_urgent_keyword_raises_score(self, scorer):
        low = scorer.score("hi there", sender="user", channel="dm")
        high = scorer.score("URGENT the system is broken", sender="user", channel="dm")
        assert high.score > low.score

    def test_action_keyword_raises_score(self, scorer):
        low = scorer.score("thanks", sender="user", channel="dm")
        high = scorer.score("can you please fix this bug", sender="user", channel="dm")
        assert high.score > low.score

    def test_question_mark_raises_score(self, scorer):
        low = scorer.score("noted", sender="user", channel="dm")
        high = scorer.score("can you check the deployment?", sender="user", channel="dm")
        assert high.score > low.score

    def test_already_answered_reduces_score(self, scorer):
        without = scorer.score("hi", sender="user", channel="group")
        with_flag = scorer.score("hi", sender="user", channel="group", context={"already_answered": True})
        assert with_flag.score < without.score

    def test_ongoing_conversation_raises_score(self, scorer):
        without = scorer.score("hi", sender="user", channel="group")
        with_flag = scorer.score("hi", sender="user", channel="group", context={"ongoing": True})
        assert with_flag.score >= without.score


class TestChannels:
    def test_dm_scores_higher_than_group(self, scorer):
        dm = scorer.score("hello", sender="user", channel="dm")
        group = scorer.score("hello", sender="user", channel="group")
        assert dm.score > group.score

    def test_group_scores_higher_than_broadcast(self, scorer):
        group = scorer.score("hello", sender="user", channel="group")
        broadcast = scorer.score("hello", sender="user", channel="broadcast")
        assert group.score > broadcast.score


class TestAllowlist:
    def test_allowlisted_sender_higher_than_unknown(self, scorer):
        unknown = scorer.score("hello", sender="stranger", channel="dm")
        known = scorer.score("hello", sender="known_user", channel="dm", allowlist={"known_user"})
        assert known.score > unknown.score


class TestOutputFormat:
    def test_result_has_score(self, scorer):
        result = scorer.score("test")
        assert isinstance(result.score, int)
        assert 0 <= result.score <= 100

    def test_result_has_valid_level(self, scorer):
        result = scorer.score("test")
        assert result.level in ("full", "standard", "minimal", "ignore")

    def test_result_has_reason(self, scorer):
        result = scorer.score("urgent request!", sender="user", channel="dm")
        assert len(result.reason) > 0

    def test_result_has_breakdown(self, scorer):
        result = scorer.score("test", channel="dm")
        assert "sender" in result.breakdown
        assert "channel" in result.breakdown


class TestThresholdConfiguration:
    def test_custom_thresholds_affect_levels(self):
        """Very permissive config makes everything 'full'."""
        cfg = Config(
            attention_full_threshold=0,
            attention_standard_threshold=0,
            attention_minimal_threshold=0,
        )
        scorer = AttentionScorer(cfg)
        result = scorer.score("whatever", sender="unknown", channel="broadcast")
        assert result.level == "full"

    def test_very_strict_config_produces_ignore(self):
        """Extremely strict config: only owner (100) gets through."""
        cfg = Config(
            attention_full_threshold=100,
            attention_standard_threshold=100,
            attention_minimal_threshold=100,
        )
        scorer = AttentionScorer(cfg)
        result = scorer.score("hello", sender="unknown", channel="dm")
        assert result.level == "ignore"
