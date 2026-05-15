"""Runtime hallucination detection for extremis memory writes.

Implements the memory-relevant portion of the standard three-layer stack:
- NLI faithfulness check against the source conversation (local, ~50ms)
- LLM-as-judge escalation on borderline scores (Anthropic, one extra call)
- Self-consistency on high-stakes layers (N parallel extractions, embedding convergence)

Surfaces are intentionally small: a verify() entrypoint for write-time use and
self_consistency_filter() for the extraction-time gate.
"""

from .consistency import ConsistencyResult, self_consistency_filter
from .faithfulness import VerificationResult, verify
from .judge import JudgeVerdict, LLMJudge
from .nli import NLIChecker, NLIResult
from .recommendations import (
    Recommendation,
    recommend_for_recall,
    recommend_for_verification,
    recommendations_to_metadata,
)

__all__ = [
    "ConsistencyResult",
    "JudgeVerdict",
    "LLMJudge",
    "NLIChecker",
    "NLIResult",
    "Recommendation",
    "VerificationResult",
    "recommend_for_recall",
    "recommend_for_verification",
    "recommendations_to_metadata",
    "self_consistency_filter",
    "verify",
]
