"""
user_id + agent_id recall filtering.

Pinned identity/procedural memories always surface, but everything else
is scoped: passing ``user_id="alice"`` to recall() should return only
memories Alice stored, and only ones the matching agent stored when
``agent_id`` is also specified. Memories without the corresponding
metadata key are excluded — they're tenant-default and shouldn't bleed
into a per-user lens.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from extremis import Config, Extremis


@pytest.fixture
def mem():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(
            namespace="test_identity_filter",
            extremis_home=tmp,
            enable_faithfulness_check=False,
        )
        instance = Extremis(config=cfg)
        yield instance


def _contents(results) -> list[str]:
    return [r.memory.content for r in results]


def test_recall_filters_by_user_id(mem: Extremis) -> None:
    mem.remember("Alice asked for a refund on order 42", user_id="alice", agent_id="bot")
    mem.remember("Bob asked for help with login", user_id="bob", agent_id="bot")
    mem.remember("Default memory — no user attached")  # no user_id metadata

    alice_only = _contents(mem.recall("help with order", user_id="alice"))
    assert any("Alice" in c for c in alice_only)
    assert all("Bob" not in c for c in alice_only)
    assert all("Default" not in c for c in alice_only), (
        "memories without user_id should NOT surface in a user-scoped lens"
    )


def test_recall_filters_by_agent_id(mem: Extremis) -> None:
    mem.remember("Support handled refund for customer", user_id="alice", agent_id="support-bot")
    mem.remember("Sales pitched upgrade to customer", user_id="alice", agent_id="sales-bot")

    support_only = _contents(
        mem.recall("customer interaction", agent_id="support-bot", min_score=0.0)
    )
    assert any("Support" in c for c in support_only)
    assert all("Sales" not in c for c in support_only)


def test_recall_filters_by_user_and_agent(mem: Extremis) -> None:
    mem.remember("Alice + support: refund processed", user_id="alice", agent_id="support-bot")
    mem.remember("Alice + sales: tried to upsell", user_id="alice", agent_id="sales-bot")
    mem.remember("Bob + support: account locked", user_id="bob", agent_id="support-bot")

    alice_support = _contents(mem.recall("interaction", user_id="alice", agent_id="support-bot"))
    assert any("Alice + support" in c for c in alice_support)
    assert all("Alice + sales" not in c for c in alice_support)
    assert all("Bob" not in c for c in alice_support)


def test_recall_with_no_filter_returns_everything(mem: Extremis) -> None:
    mem.remember("Alice memory", user_id="alice")
    mem.remember("Bob memory", user_id="bob")
    mem.remember("Untagged memory")

    everything = _contents(mem.recall("memory", limit=20))
    assert any("Alice" in c for c in everything)
    assert any("Bob" in c for c in everything)
    assert any("Untagged" in c for c in everything)
