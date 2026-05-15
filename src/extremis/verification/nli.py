"""Local NLI faithfulness check.

Wraps a HuggingFace cross-encoder NLI model (default: deberta-v3-small).
Lazy-loaded so the transformers/torch deps stay optional — install with
`pip install extremis[verification]`.

Why NLI over embedding similarity: embeddings score "Payment must be made
within 30 days" ≈ "Payment does not need to be made within 30 days" as
highly similar (same words). NLI correctly labels them CONTRADICTION.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

try:
    from peekr.decorators import trace as _trace
except ImportError:

    def _trace(_func=None, *, name=None, capture_io=True):  # type: ignore[misc]
        def decorator(fn):
            return fn

        return decorator(_func) if _func is not None else decorator


log = logging.getLogger(__name__)


@dataclass
class NLIResult:
    score: float  # max entailment probability across sources, in [0, 1]
    label: str  # "ENTAILMENT" | "NEUTRAL" | "CONTRADICTION"
    best_source_idx: Optional[int]  # index of the source that best supports the claim


class NLIChecker:
    """Cross-encoder NLI entailment check. Loads model on first use."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small") -> None:
        self._model_name = model_name
        self._pipeline = None  # lazy

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "NLI faithfulness checks require the `verification` extra. "
                "Install with: pip install extremis[verification]"
            ) from exc
        self._pipeline = pipeline(
            "text-classification",
            model=self._model_name,
            return_all_scores=True,
        )
        return self._pipeline

    @_trace(name="extremis.verification.nli", capture_io=False)
    def entailment_score(self, claim: str, sources: list[str]) -> NLIResult:
        """Return the max entailment probability of `claim` against any `source`.

        score in [0, 1]: 1.0 = fully entailed, 0.0 = clearly not supported.
        best_source_idx points to the source with the highest entailment.
        """
        if not sources:
            return NLIResult(score=0.0, label="NEUTRAL", best_source_idx=None)
        if not claim.strip():
            return NLIResult(score=0.0, label="NEUTRAL", best_source_idx=None)

        pipe = self._load()

        best_score = 0.0
        best_label = "NEUTRAL"
        best_idx: Optional[int] = None
        worst_contradiction = 0.0

        for idx, source in enumerate(sources):
            try:
                # Cross-encoder NLI: input is "premise [SEP] hypothesis".
                # The pipeline tokenizer accepts a dict {"text": premise, "text_pair": hypothesis}.
                outputs = pipe({"text": source, "text_pair": claim})
            except Exception as exc:
                log.debug("NLI inference failed on source %d: %s", idx, exc)
                continue

            # return_all_scores=True returns list of dicts, one per label
            scores = outputs[0] if outputs and isinstance(outputs[0], list) else outputs
            entail = 0.0
            contradict = 0.0
            for item in scores:
                label = item.get("label", "").upper()
                p = float(item.get("score", 0.0))
                if "ENTAIL" in label:
                    entail = p
                elif "CONTRADICT" in label:
                    contradict = p

            if entail > best_score:
                best_score = entail
                best_label = "ENTAILMENT"
                best_idx = idx
            if contradict > worst_contradiction:
                worst_contradiction = contradict

        # If any source contradicts strongly and nothing entails strongly,
        # report the contradiction so callers can downrank confidently.
        if worst_contradiction > 0.7 and best_score < 0.5:
            return NLIResult(score=0.0, label="CONTRADICTION", best_source_idx=best_idx)

        return NLIResult(score=best_score, label=best_label, best_source_idx=best_idx)
