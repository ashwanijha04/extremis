"""Self-consistency filter for high-stakes extractions.

Re-sample the extractor N times at temperature > 0 and keep only claims whose
embeddings converge across samples. Catches "model is interpolating" failures
that faithfulness can miss (e.g., the source supports several conclusions and
the extractor is guessing among them).

Does not implement the LLM calls — takes a callable `extract_fn` and an
embedder so callers can wire in their existing extraction path.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Optional

try:
    from peekr.decorators import trace as _trace
except ImportError:

    def _trace(_func=None, *, name=None, capture_io=True):  # type: ignore[misc]
        def decorator(fn):
            return fn

        return decorator(_func) if _func is not None else decorator


log = logging.getLogger(__name__)


@dataclass
class ConsistencyResult:
    claim: str
    mean_similarity: float
    samples: int


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _pairwise_mean(embeddings: list[list[float]]) -> float:
    n = len(embeddings)
    if n < 2:
        return 1.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _cosine(embeddings[i], embeddings[j])
            count += 1
    return total / count if count else 0.0


@_trace(name="extremis.verification.consistency", capture_io=False)
def self_consistency_filter(
    extract_fn: Callable[[], list[dict]],
    *,
    embedder,
    n: int,
    threshold: float,
    layers_in_scope: Optional[set[str]] = None,
) -> tuple[list[dict], list[ConsistencyResult]]:
    """Run extract_fn N times and return claims that converge.

    Strategy:
    - Sample N independent extractions.
    - For each claim in sample 0, find the nearest-neighbour claim in each
      other sample (cosine on content embeddings).
    - Compute mean pairwise similarity across the N matched embeddings.
    - Keep claims whose mean ≥ threshold.

    Claims whose layer is not in `layers_in_scope` pass through untouched
    (gated to keep cost off layers we don't care about).

    Returns (kept_claims, per_claim_stats).
    """
    if n <= 1:
        return extract_fn(), []

    samples: list[list[dict]] = []
    for _ in range(n):
        try:
            samples.append(extract_fn())
        except Exception as exc:
            log.warning("Self-consistency sample failed: %s", exc)
            samples.append([])

    base = samples[0]
    if not base:
        return [], []

    # Embed once per unique content to avoid recomputing
    cache: dict[str, list[float]] = {}

    def embed(text: str) -> list[float]:
        if text not in cache:
            cache[text] = embedder.embed(text)
        return cache[text]

    other_samples = [s for s in samples[1:] if s]
    kept: list[dict] = []
    stats: list[ConsistencyResult] = []

    for claim in base:
        layer = claim.get("layer", "")
        content = (claim.get("content") or "").strip()
        if not content:
            continue

        # Layers outside scope skip the check
        if layers_in_scope is not None and layer not in layers_in_scope:
            kept.append(claim)
            continue

        try:
            base_emb = embed(content)
        except Exception:
            kept.append(claim)
            continue

        matched: list[list[float]] = [base_emb]
        for other in other_samples:
            best_sim = -1.0
            best_emb: Optional[list[float]] = None
            for cand in other:
                cand_content = (cand.get("content") or "").strip()
                if not cand_content or cand.get("layer") != layer:
                    continue
                try:
                    cand_emb = embed(cand_content)
                except Exception:
                    continue
                sim = _cosine(base_emb, cand_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_emb = cand_emb
            if best_emb is not None:
                matched.append(best_emb)

        mean_sim = _pairwise_mean(matched)
        stats.append(ConsistencyResult(claim=content, mean_similarity=mean_sim, samples=len(matched)))

        if mean_sim >= threshold:
            # Stamp the consistency stats on the claim so the caller can
            # propagate them into memory.metadata.consistency.
            claim["_consistency"] = {
                "mean_similarity": round(mean_sim, 4),
                "samples": len(matched),
            }
            kept.append(claim)
        else:
            log.debug(
                "Dropping inconsistent claim (mean_sim=%.3f, threshold=%.2f): %r",
                mean_sim,
                threshold,
                content[:80],
            )

    return kept, stats
