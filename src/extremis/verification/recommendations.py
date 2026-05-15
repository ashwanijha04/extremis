"""Actionable recommendations for verification + recall issues.

Pure functions over (verification result, memory state, current time) →
structured Recommendation objects. Stamped on memory.metadata at write
time and surfaced on RecallResult.sources at read time so operators can
see *what to do* about each flagged memory, not just *that* it's flagged.

Severity guide:
- high   — needs human attention or programmatic action soon
- medium — pattern worth watching; fix at the prompt/extractor level
- low    — informational, fine to ignore unless it clusters
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from ..types import Memory


@dataclass
class Recommendation:
    issue: str  # stable machine-readable identifier
    severity: str  # "low" | "medium" | "high"
    action: str  # imperative — what to do now
    suggestion: str  # systemic fix — what to do to prevent recurrence
    refs: dict  # pointers (source_message_ids, source_memory_ids, conversation_id)

    def to_dict(self) -> dict:
        return asdict(self)


# Thresholds — kept here so all heuristics live in one place
_STALE_EFFECTIVE_CONFIDENCE = 0.3
_BORDERLINE_FAITHFULNESS = 0.7


def recommend_for_verification(
    verification: dict,
    memory_refs: dict,
) -> list[Recommendation]:
    """Recommendations stamped at write time based on the faithfulness verdict.

    `verification` is the dict produced by `VerificationResult.to_metadata()`.
    `memory_refs` carries `conversation_id` and `source_message_ids` for the
    refs payload — fine to leave any unknown.
    """
    recs: list[Recommendation] = []
    verdict = verification.get("verdict", "")
    score = float(verification.get("score") or 0.0)
    method = verification.get("method", "")
    judge_reason = verification.get("judge_reason") or ""

    refs = {
        "conversation_id": memory_refs.get("conversation_id"),
        "source_message_ids": memory_refs.get("source_message_ids", []),
        "method": method,
    }

    if verdict == "CONTRADICTED":
        recs.append(
            Recommendation(
                issue="claim_contradicts_source",
                severity="high",
                action=(
                    "Open the source conversation and inspect the messages listed in "
                    "refs.source_message_ids. Manually correct the memory or delete it; "
                    "if multiple memories from the same conversation are contradicted, "
                    "purge that conversation's extractions and re-run consolidation."
                ),
                suggestion=(
                    "If contradictions cluster on a specific user or model, the extractor "
                    "is over-generalising. Tighten EXTRACTION_SYSTEM with explicit "
                    "no-guessing language, lower max_tokens, or downgrade the model — "
                    "Haiku is more conservative than Sonnet on this task."
                ),
                refs={**refs, "judge_reason": judge_reason} if judge_reason else refs,
            )
        )
        return recs

    if verdict == "UNVERIFIABLE":
        recs.append(
            Recommendation(
                issue="claim_unverifiable",
                severity="medium",
                action=(
                    "Treat this memory as low-confidence. If a downstream agent uses it, "
                    "hedge the response and ask the user to confirm before acting on it."
                ),
                suggestion=(
                    "The extractor is producing claims that aren't grounded in the "
                    "conversation. Either widen the consolidation batch window so more "
                    "context is available, or add a rule to EXTRACTION_SYSTEM: "
                    "'Skip claims that require inference beyond the literal text.'"
                ),
                refs=refs,
            )
        )
        return recs

    if verdict == "SUPPORTED" and score < _BORDERLINE_FAITHFULNESS:
        recs.append(
            Recommendation(
                issue="borderline_support",
                severity="low",
                action=(
                    "Memory is supported but only weakly. Acceptable to surface; "
                    "monitor whether similar claims start failing."
                ),
                suggestion=(
                    "If borderline scores cluster on one claim type, that template "
                    "is probably ambiguous — split it into more specific patterns."
                ),
                refs=refs,
            )
        )

    return recs


def recommend_for_recall(
    memory: Memory,
    effective_confidence: Optional[float],
    now: Optional[datetime] = None,
) -> list[Recommendation]:
    """Recommendations computed at recall time from memory age + stored verification."""
    recs: list[Recommendation] = []
    now = now or datetime.now(tz=timezone.utc)

    refs = {
        "memory_id": str(memory.id),
        "conversation_id": (memory.metadata or {}).get("conversation_id"),
        "source_memory_ids": [str(mid) for mid in memory.source_memory_ids],
    }

    # Expired — past validity_end but still surfaceable
    validity_end = memory.validity_end
    if validity_end is not None:
        if validity_end.tzinfo is None:
            validity_end = validity_end.replace(tzinfo=timezone.utc)
        if validity_end < now:
            recs.append(
                Recommendation(
                    issue="memory_expired",
                    severity="high",
                    action=(
                        "Memory is past its validity_end. Re-validate against a recent "
                        "conversation or remove it via supersede()."
                    ),
                    suggestion=(
                        "Schedule a periodic compact() pass that deletes memories with validity_end older than N days."
                    ),
                    refs={**refs, "validity_end": validity_end.isoformat()},
                )
            )

    # Stored verification flagged it but it's still in the store
    verification = (memory.metadata or {}).get("verification") or {}
    verdict = verification.get("verdict")
    if verdict == "CONTRADICTED":
        recs.append(
            Recommendation(
                issue="surfacing_contradicted_memory",
                severity="high",
                action=(
                    "This memory failed faithfulness at write time and is still in "
                    "the store. Run compact() on its layer, or manually delete it "
                    "before it influences a response."
                ),
                suggestion=(
                    "Add a recall-time filter: drop memories where "
                    "metadata.verification.verdict == 'CONTRADICTED' unless the caller "
                    "explicitly opts in for audit purposes."
                ),
                refs={
                    **refs,
                    "verification_method": verification.get("method"),
                    "verification_score": verification.get("score"),
                },
            )
        )

    # Stale: confidence × layer_weight × decay dropped below the threshold
    if (
        effective_confidence is not None
        and effective_confidence > 0.0
        and effective_confidence < _STALE_EFFECTIVE_CONFIDENCE
    ):
        created = memory.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = round((now - created).total_seconds() / 86400.0, 1)
        recs.append(
            Recommendation(
                issue="stale_confidence",
                severity="medium",
                action=(
                    f"Hedge the response — memory is {age_days} days old and its "
                    f"effective confidence is {effective_confidence:.2f}. Phrase "
                    f"answers as 'as of {age_days:.0f} days ago, …' rather than present tense."
                ),
                suggestion=(
                    "If this memory is still load-bearing, ask the user to confirm "
                    "and write a fresh remember_now() to reset its age."
                ),
                refs={**refs, "age_days": age_days},
            )
        )

    return recs


def recommendations_to_metadata(recs: list[Recommendation]) -> list[dict]:
    """Serialize for storage on memory.metadata.recommendations."""
    return [r.to_dict() for r in recs]
