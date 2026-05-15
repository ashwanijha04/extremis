"""Tiered faithfulness orchestration: NLI → judge on grey zone.

verify(claim, sources, ...) → VerificationResult with a final score, the
method that produced it, and the index of the source that best grounds the
claim (None if contradicted/unverifiable). Callers stamp the result into
memory.metadata["verification"] and downrank confidence accordingly.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from .judge import JudgeVerdict, LLMJudge
from .nli import NLIChecker, NLIResult

log = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    score: float  # final faithfulness score in [0, 1]
    verdict: str  # "SUPPORTED" | "UNVERIFIABLE" | "CONTRADICTED"
    method: str  # "nli" | "nli+judge" | "judge-only" | "skipped"
    nli_score: Optional[float] = None
    judge_score: Optional[float] = None
    judge_reason: str = ""
    best_source_idx: Optional[int] = None
    verified_at: str = ""

    def to_metadata(self) -> dict:
        d = asdict(self)
        return d


def _verdict_from_score(score: float, pass_threshold: float) -> str:
    if score >= pass_threshold:
        return "SUPPORTED"
    if score <= 0.2:
        return "CONTRADICTED"
    return "UNVERIFIABLE"


def verify(
    claim: str,
    sources: list[str],
    *,
    nli: Optional[NLIChecker],
    judge: Optional[LLMJudge],
    pass_threshold: float = 0.85,
    grey_zone_low: float = 0.5,
) -> VerificationResult:
    """Run the tiered faithfulness check.

    - If `nli` is None (extras not installed), fall back to judge-only.
    - NLI score ≥ pass_threshold → SUPPORTED, skip judge.
    - NLI score < grey_zone_low → CONTRADICTED/UNVERIFIABLE, skip judge (save cost).
    - Grey zone → escalate to judge, use judge score as final.
    """
    now = datetime.now(tz=timezone.utc).isoformat()

    if nli is None and judge is None:
        return VerificationResult(
            score=1.0,
            verdict="SUPPORTED",
            method="skipped",
            verified_at=now,
        )

    nli_result: Optional[NLIResult] = None
    if nli is not None:
        try:
            nli_result = nli.entailment_score(claim, sources)
        except Exception as exc:
            log.warning("NLI check failed; falling back to judge: %s", exc)
            nli_result = None

    if nli_result is None:
        # Judge-only fallback
        if judge is None:
            return VerificationResult(
                score=0.5,
                verdict="UNVERIFIABLE",
                method="skipped",
                verified_at=now,
            )
        context = "\n".join(sources)
        verdict = judge.judge(claim, context)
        return VerificationResult(
            score=verdict.score,
            verdict=verdict.verdict,
            method="judge-only",
            judge_score=verdict.score,
            judge_reason=verdict.reason,
            verified_at=now,
        )

    # NLI succeeded — decide based on score
    if nli_result.score >= pass_threshold:
        return VerificationResult(
            score=nli_result.score,
            verdict="SUPPORTED",
            method="nli",
            nli_score=nli_result.score,
            best_source_idx=nli_result.best_source_idx,
            verified_at=now,
        )

    if nli_result.score < grey_zone_low or judge is None:
        verdict_label = _verdict_from_score(nli_result.score, pass_threshold)
        # NLI flagged outright contradiction → keep that label
        if nli_result.label == "CONTRADICTION":
            verdict_label = "CONTRADICTED"
        return VerificationResult(
            score=nli_result.score,
            verdict=verdict_label,
            method="nli",
            nli_score=nli_result.score,
            best_source_idx=nli_result.best_source_idx,
            verified_at=now,
        )

    # Grey zone — escalate
    context = "\n".join(sources)
    judge_verdict: JudgeVerdict = judge.judge(claim, context)
    return VerificationResult(
        score=judge_verdict.score,
        verdict=judge_verdict.verdict,
        method="nli+judge",
        nli_score=nli_result.score,
        judge_score=judge_verdict.score,
        judge_reason=judge_verdict.reason,
        best_source_idx=nli_result.best_source_idx,
        verified_at=now,
    )
