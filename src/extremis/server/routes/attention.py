"""Attention scoring route.

Wraps the in-process AttentionScorer so SDK callers can grade incoming
messages without running the heuristic locally.

Method choice: POST with query parameters (no body). Scoring is logically
idempotent so GET would also fit, but POST avoids surprises with browsers
caching scores and matches the existing TS SDK's `postWithQuery` shape.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from ...scorer.attention import AttentionScorer
from ...types import AttentionResult
from ..deps import Memory

router = APIRouter(tags=["attention"])


def _csv_to_set(value: str) -> Optional[set[str]]:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


@router.post("/score")
def score(
    mem: Memory,
    message: str = Query(...),
    sender: str = Query(""),
    channel: str = Query("dm"),
    owner_ids: str = Query(""),
    allowlist: str = Query(""),
    ongoing: bool = Query(False),
    already_answered: bool = Query(False),
) -> AttentionResult:
    scorer = AttentionScorer(mem._config)
    return scorer.score(
        message=message,
        sender=sender,
        channel=channel,
        owner_ids=_csv_to_set(owner_ids),
        allowlist=_csv_to_set(allowlist),
        context={"ongoing": ongoing, "already_answered": already_answered},
    )
