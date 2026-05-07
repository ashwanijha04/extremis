"""HeuristicObserver — log compression and priority classification tests."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from extremis.observer.observer import _CAPS_DAILY, HeuristicObserver
from extremis.types import LogEntry, ObservationPriority


def make_entry(content: str, conversation_id: str = "c1") -> LogEntry:
    return LogEntry(
        role="user",
        content=content,
        conversation_id=conversation_id,
        timestamp=datetime.now(tz=timezone.utc),
    )


@pytest.fixture
def observer():
    return HeuristicObserver(namespace="test_ns")


class TestPriorityClassification:
    def test_decision_keywords_are_critical(self, observer):
        entries = [make_entry("We decided to ship the feature tomorrow")]
        obs = observer.compress(entries)
        assert obs[0].priority == ObservationPriority.CRITICAL

    def test_error_keywords_are_critical(self, observer):
        entries = [make_entry("The deployment failed with an error")]
        obs = observer.compress(entries)
        assert obs[0].priority == ObservationPriority.CRITICAL

    def test_urgent_keyword_is_critical(self, observer):
        entries = [make_entry("This is urgent, the API is broken")]
        obs = observer.compress(entries)
        assert obs[0].priority == ObservationPriority.CRITICAL

    def test_launched_is_critical(self, observer):
        entries = [make_entry("We launched the new version today")]
        obs = observer.compress(entries)
        assert obs[0].priority == ObservationPriority.CRITICAL

    def test_reason_keyword_is_context(self, observer):
        entries = [make_entry("The reason we chose Postgres is scalability")]
        obs = observer.compress(entries)
        assert obs[0].priority == ObservationPriority.CONTEXT

    def test_learned_keyword_is_context(self, observer):
        entries = [make_entry("I learned that vector search is faster with pgvector")]
        obs = observer.compress(entries)
        assert obs[0].priority == ObservationPriority.CONTEXT

    def test_generic_message_is_info(self, observer):
        entries = [make_entry("The weather in Dubai is 38 degrees")]
        obs = observer.compress(entries)
        assert obs[0].priority == ObservationPriority.INFO

    def test_because_is_context(self, observer):
        entries = [make_entry("We chose SQLite because it's simpler")]
        obs = observer.compress(entries)
        assert obs[0].priority == ObservationPriority.CONTEXT


class TestDeduplication:
    def test_exact_duplicate_skipped(self, observer):
        entry = make_entry("Same message")
        obs = observer.compress([entry, entry])
        assert len(obs) == 1

    def test_near_duplicate_by_first_80_chars_skipped(self, observer):
        long_prefix = "x" * 80
        e1 = make_entry(long_prefix + " suffix one")
        e2 = make_entry(long_prefix + " suffix two")
        obs = observer.compress([e1, e2])
        assert len(obs) == 1

    def test_different_messages_not_deduped(self, observer):
        entries = [make_entry("Message A"), make_entry("Message B")]
        obs = observer.compress(entries)
        assert len(obs) == 2

    def test_empty_content_skipped(self, observer):
        entries = [make_entry(""), make_entry("  "), make_entry("Real content")]
        obs = observer.compress(entries)
        assert len(obs) == 1


class TestCapEnforcement:
    def test_critical_cap_respected(self, observer):
        entries = [make_entry(f"We decided to do thing {i}") for i in range(20)]
        caps = {ObservationPriority.CRITICAL: 3, ObservationPriority.CONTEXT: 10, ObservationPriority.INFO: 5}
        obs = observer.compress(entries, caps=caps)
        critical = [o for o in obs if o.priority == ObservationPriority.CRITICAL]
        assert len(critical) <= 3

    def test_info_cap_respected(self, observer):
        entries = [make_entry(f"Generic message number {i}") for i in range(20)]
        caps = {ObservationPriority.CRITICAL: 15, ObservationPriority.CONTEXT: 10, ObservationPriority.INFO: 2}
        obs = observer.compress(entries, caps=caps)
        info = [o for o in obs if o.priority == ObservationPriority.INFO]
        assert len(info) <= 2

    def test_default_caps_applied(self, observer):
        critical = [make_entry(f"We decided {i}") for i in range(30)]
        obs = observer.compress(critical)
        crit_obs = [o for o in obs if o.priority == ObservationPriority.CRITICAL]
        assert len(crit_obs) <= _CAPS_DAILY[ObservationPriority.CRITICAL]


class TestFormatMarkdown:
    def test_format_contains_content(self, observer):
        entries = [make_entry("We decided to ship")]
        obs = observer.compress(entries)
        md = HeuristicObserver.format_markdown(obs)
        assert "We decided to ship" in md

    def test_format_contains_critical_icon(self, observer):
        entries = [make_entry("System crashed with error")]
        obs = observer.compress(entries)
        md = HeuristicObserver.format_markdown(obs)
        assert "🔴" in md

    def test_format_contains_date(self, observer):
        entries = [make_entry("Something happened")]
        obs = observer.compress(entries)
        md = HeuristicObserver.format_markdown(obs)
        assert "Date:" in md

    def test_empty_observations_gives_date_only(self, observer):
        md = HeuristicObserver.format_markdown([])
        assert "Date:" in md


class TestMetadata:
    def test_namespace_set_on_observation(self, observer):
        entries = [make_entry("Something")]
        obs = observer.compress(entries)
        assert obs[0].namespace == "test_ns"

    def test_conversation_id_preserved(self, observer):
        entries = [make_entry("Message", conversation_id="conv_xyz")]
        obs = observer.compress(entries)
        assert obs[0].conversation_id == "conv_xyz"

    def test_role_preserved_as_tag(self, observer):
        entry = LogEntry(role="assistant", content="I decided to help", conversation_id="c1",
                         timestamp=datetime.now(tz=timezone.utc))
        obs = observer.compress([entry])
        assert "assistant" in obs[0].tags
