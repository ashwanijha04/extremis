"""
Heuristic log observer — compresses raw log entries into priority-tagged observations.
Ported and generalised from friday-saas/base/scripts/observer.py.

Priority bands:
  🔴 CRITICAL — decisions, errors, deadlines, blockers, reward signals
  🟡 CONTEXT  — reasons, insights, learnings, explanations
  🟢 INFO     — everything else

No LLM required. Runs fast, zero cost.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from ..types import LogEntry, Observation, ObservationPriority

_CRITICAL_PATTERNS = [
    r"\bdecid(ed|e|ing)\b",
    r"\bapproved?\b",
    r"\bdeadline\b",
    r"\burgent\b",
    r"\basap\b",
    r"\berror\b",
    r"\bfail(ed|ure|ing)?\b",
    r"\bbreak(s|ing)?\b|\bbroke\b|\bbroken\b",
    r"\bblocker?\b|\bblocked\b",
    r"\bcritical\b",
    r"\bcrash(ed|ing)?\b",
    r"\bdown\b",                     # "service is down"
    r"\+1\b|-1\b",                   # reward signals
    r"\bship(ped|ping)?\b",          # shipping decisions
    r"\blaunch(ed|ing)?\b",
    r"\bmerged?\b",
    r"\bdeployed?\b",
]

_CONTEXT_PATTERNS = [
    r"\bbecause\b",
    r"\breason\b",
    r"\blearned?\b",
    r"\binsight\b",
    r"\bdiscover(ed|y)\b",
    r"\bnoticed?\b",
    r"\brealize[ds]?\b|\brealised?\b",
    r"\bunderstand\b|\bunderstood\b",
    r"\bexplain(ed|s)?\b",
    r"\bcontext\b",
    r"\bbackground\b",
    r"\bactually\b",
]

_CAPS_DAILY: dict[ObservationPriority, int] = {
    ObservationPriority.CRITICAL: 15,
    ObservationPriority.CONTEXT: 10,
    ObservationPriority.INFO: 5,
}


def _classify(content: str) -> ObservationPriority:
    lower = content.lower()
    for pattern in _CRITICAL_PATTERNS:
        if re.search(pattern, lower):
            return ObservationPriority.CRITICAL
    for pattern in _CONTEXT_PATTERNS:
        if re.search(pattern, lower):
            return ObservationPriority.CONTEXT
    return ObservationPriority.INFO


class HeuristicObserver:
    """
    Compresses a list of LogEntry objects into Observation objects.

    Usage:
        observer = HeuristicObserver(namespace="user_123")
        observations = observer.compress(log_entries, max_per_priority=_CAPS_DAILY)
    """

    def __init__(self, namespace: str = "default") -> None:
        self._namespace = namespace

    def compress(
        self,
        entries: list[LogEntry],
        caps: Optional[dict[ObservationPriority, int]] = None,
    ) -> list[Observation]:
        """
        Classify entries by priority, deduplicate, and enforce per-priority caps.
        Returns observations sorted critical → context → info.
        """
        caps = caps or _CAPS_DAILY
        buckets: dict[ObservationPriority, list[Observation]] = {
            ObservationPriority.CRITICAL: [],
            ObservationPriority.CONTEXT: [],
            ObservationPriority.INFO: [],
        }
        seen_content: set[str] = set()

        for entry in entries:
            content = entry.content.strip()
            if not content:
                continue
            # Deduplicate by first 80 chars (case-insensitive)
            dedup_key = content[:80].lower()
            if dedup_key in seen_content:
                continue
            seen_content.add(dedup_key)

            priority = _classify(content)
            obs = Observation(
                namespace=self._namespace,
                content=content,
                priority=priority,
                timestamp=entry.timestamp,
                conversation_id=entry.conversation_id,
                tags=[entry.role],
            )
            buckets[priority].append(obs)

        result: list[Observation] = []
        for priority in (ObservationPriority.CRITICAL, ObservationPriority.CONTEXT, ObservationPriority.INFO):
            cap = caps.get(priority, 999)
            result.extend(buckets[priority][:cap])

        return result

    @staticmethod
    def format_markdown(observations: list[Observation], date: Optional[datetime] = None) -> str:
        """Format observations as a markdown block for injection into context."""
        date_str = (date or datetime.now(tz=timezone.utc)).strftime("%Y-%m-%d")
        icons = {
            ObservationPriority.CRITICAL: "🔴",
            ObservationPriority.CONTEXT: "🟡",
            ObservationPriority.INFO: "🟢",
        }
        lines = [f"Date: {date_str}"]
        for obs in observations:
            icon = icons[obs.priority]
            time_str = obs.timestamp.strftime("%H:%M")
            lines.append(f"- {icon} {time_str} {obs.content}")
        return "\n".join(lines)
