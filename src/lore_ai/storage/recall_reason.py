"""Builds the human-readable 'reason' string on a RecallResult."""
from __future__ import annotations

from datetime import datetime, timezone

from ..types import MemoryLayer


def build_reason(
    relevance: float,
    score: float,
    access_count: int,
    created_at: datetime | str,
    layer: MemoryLayer,
) -> str:
    """
    Returns a readable explanation of why this memory ranked where it did.

    Examples:
      "similarity 0.91 · score +4.0 · used 8× · 3d old"
      "procedural (always included) · similarity 0.67 · first recall · today"
      "similarity 0.54 · score -1.5 · used 2× · 45d old"
    """
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)

    age_days = (datetime.now(tz=timezone.utc) - created_at.replace(tzinfo=timezone.utc)).days

    parts: list[str] = []

    if layer in (MemoryLayer.IDENTITY, MemoryLayer.PROCEDURAL):
        parts.append(f"{layer.value} (always included)")

    parts.append(f"similarity {relevance:.2f}")

    if score > 0.05:
        parts.append(f"score +{score:.1f}")
    elif score < -0.05:
        parts.append(f"score {score:.1f}")

    if access_count == 0:
        parts.append("first recall")
    elif access_count == 1:
        parts.append("used 1×")
    else:
        parts.append(f"used {access_count}×")

    if age_days == 0:
        parts.append("today")
    elif age_days == 1:
        parts.append("yesterday")
    else:
        parts.append(f"{age_days}d old")

    return " · ".join(parts)
