"""LLM-as-judge for borderline faithfulness scores.

Single-claim verdict against the full conversation context. Reuses the
existing Anthropic client + consolidation_model so no new credentials are
needed. Only invoked when NLI lands in the grey zone — keeps cost bounded.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

try:
    from peekr.decorators import trace as _trace
except ImportError:

    def _trace(_func=None, *, name=None, capture_io=True):  # type: ignore[misc]
        def decorator(fn):
            return fn

        return decorator(_func) if _func is not None else decorator


log = logging.getLogger(__name__)


FAITHFULNESS_JUDGE_SYSTEM = """\
You are a strict factual auditor checking whether a single CLAIM about a user \
is actually supported by the CONTEXT (a conversation transcript).

For the claim, return one of:
  SUPPORTED     — directly backed by something the user said in the context
  CONTRADICTED  — conflicts with what the user said in the context
  UNVERIFIABLE  — context does not mention this; you would be guessing

No hedging. No preamble. Return ONLY this JSON:
{"verdict": "SUPPORTED|CONTRADICTED|UNVERIFIABLE", "score": 0.0, "reason": "..."}

The score is your confidence in the verdict on a 0.0–1.0 scale:
- SUPPORTED with strong direct evidence → 0.9+
- SUPPORTED with weak/inferential evidence → 0.6–0.8
- UNVERIFIABLE → 0.3–0.5
- CONTRADICTED → 0.0–0.2 (low score = strongly fails the claim)
"""


@dataclass
class JudgeVerdict:
    verdict: str  # "SUPPORTED" | "UNVERIFIABLE" | "CONTRADICTED"
    score: float  # 0.0–1.0
    reason: str = ""


class LLMJudge:
    """Claim-level faithfulness adjudication via Anthropic."""

    def __init__(self, client, model: str) -> None:
        self._client = client
        self._model = model

    @_trace(name="extremis.verification.judge", capture_io=False)
    def judge(self, claim: str, context: str) -> JudgeVerdict:
        """Return a structured verdict for `claim` against `context`."""
        user_msg = f"CONTEXT:\n{context}\n\nCLAIM:\n{claim}\n\nReturn only the JSON verdict."
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=300,
                system=FAITHFULNESS_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception as exc:
            log.warning("Judge call failed: %s", exc)
            return JudgeVerdict(verdict="UNVERIFIABLE", score=0.5, reason=f"judge error: {exc}")

        raw = response.content[0].text.strip()
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> JudgeVerdict:
        text = raw
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
        text = text.strip()
        try:
            data: dict = json.loads(text)
        except json.JSONDecodeError:
            log.warning("Judge returned non-JSON: %r", raw[:200])
            return JudgeVerdict(verdict="UNVERIFIABLE", score=0.5, reason="non-json response")

        verdict = str(data.get("verdict", "UNVERIFIABLE")).upper()
        if verdict not in {"SUPPORTED", "UNVERIFIABLE", "CONTRADICTED"}:
            verdict = "UNVERIFIABLE"
        score_raw = data.get("score", 0.5)
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = 0.5
        score = max(0.0, min(1.0, score))
        reason = str(data.get("reason", ""))[:500]
        return JudgeVerdict(verdict=verdict, score=score, reason=reason)
