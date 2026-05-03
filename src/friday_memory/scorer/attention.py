"""
Heuristic attention scorer — grades incoming messages before the LLM sees them.
Ported and generalised from friday-saas/base/scripts/attention-scorer.py.

Returns a score 0–100 and a processing level (full / standard / minimal / ignore).
Zero LLM calls. Runs in microseconds.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

from ..config import Config
from ..types import AttentionResult

_URGENT_KEYWORDS = [
    "urgent", "asap", "emergency", "critical", "help", "broken", "down",
    "p0", "p1", "outage", "fire", "immediately", "now",
]
_ACTION_KEYWORDS = [
    "do", "please", "can you", "send", "build", "check", "fix", "create",
    "review", "write", "update", "deploy", "run",
]
_CASUAL_PATTERNS = [
    "haha", "lol", "nice", "cool", "ok", "okay", "sure", "yep", "yeah", "nah",
    "hm", "hmm",
]


def _is_single_emoji(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    chars = [c for c in text if not unicodedata.category(c).startswith(("Cf", "Mn"))]
    return len(chars) <= 2 and all(
        unicodedata.category(c) in ("So", "Sk", "Cn") or ord(c) > 0x1F000
        for c in chars if c.strip()
    )


def _has(text: str, patterns: list[str]) -> bool:
    lower = text.lower()
    return any(p in lower for p in patterns)


class AttentionScorer:
    """
    Scores a message on a 0–100 scale and returns a processing level.

    Levels:
      full     (≥ full_threshold)    — engage fully, rich response
      standard (≥ standard_threshold) — engage, balanced response
      minimal  (≥ minimal_threshold)  — brief reply
      ignore   (< minimal_threshold)  — skip

    Configure thresholds via Config or pass directly to score().
    """

    def __init__(self, config: Config) -> None:
        self._full = config.attention_full_threshold
        self._standard = config.attention_standard_threshold
        self._minimal = config.attention_minimal_threshold

    def score(
        self,
        message: str,
        sender: str = "",
        channel: str = "dm",
        owner_ids: Optional[set[str]] = None,
        allowlist: Optional[set[str]] = None,
        context: Optional[dict] = None,
    ) -> AttentionResult:
        """
        Score a message.

        Args:
            message:    The message text.
            sender:     Sender identifier (user ID, phone number, etc.)
            channel:    "dm" | "group" | "broadcast"
            owner_ids:  Set of sender IDs that always get full attention.
            allowlist:  Set of sender IDs with elevated base score.
            context:    Optional dict with keys: ongoing (bool), new_thread (bool),
                        already_answered (bool).
        """
        owner_ids = owner_ids or set()
        allowlist = allowlist or set()
        context = context or {}
        breakdown: dict = {}
        reasons: list[str] = []

        # Owner override — always full
        if sender in owner_ids:
            return AttentionResult(
                score=100,
                level="full",
                reason="owner — always full attention",
                breakdown={"sender": 100, "override": True},
            )

        # Sender score
        if sender in allowlist:
            sender_score = 60
            reasons.append("allowlisted sender")
        else:
            sender_score = 10
            reasons.append("unknown sender")
        breakdown["sender"] = sender_score

        # Channel score
        channel_scores = {"dm": 25, "group": 15, "broadcast": 5}
        channel_score = channel_scores.get(channel.lower(), 15)
        breakdown["channel"] = channel_score
        reasons.append(f"channel={channel}")

        # Content score
        content_score = 0
        if _is_single_emoji(message):
            content_score += -5
            reasons.append("single emoji")
        else:
            if _has(message, _URGENT_KEYWORDS):
                content_score += 20
                reasons.append("urgent keyword")
            if _has(message, _ACTION_KEYWORDS):
                content_score += 15
                reasons.append("action request")
            if "?" in message:
                content_score += 10
                reasons.append("question")
            if len(message.split()) <= 4 and _has(message, _CASUAL_PATTERNS):
                content_score += 8
                reasons.append("casual banter")
        breakdown["content"] = content_score

        # Context score
        context_score = 0
        if context.get("ongoing"):
            context_score += 10
            reasons.append("ongoing conversation")
        if context.get("new_thread"):
            context_score += 10
            reasons.append("new thread")
        if context.get("already_answered"):
            context_score -= 15
            reasons.append("already answered")
        breakdown["context"] = context_score

        raw = sender_score + channel_score + content_score + context_score
        final_score = max(0, min(100, raw))
        breakdown["raw"] = raw

        if final_score >= self._full:
            level = "full"
        elif final_score >= self._standard:
            level = "standard"
        elif final_score >= self._minimal:
            level = "minimal"
        else:
            level = "ignore"

        return AttentionResult(
            score=final_score,
            level=level,
            reason="; ".join(reasons),
            breakdown=breakdown,
        )
