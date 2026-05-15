"""Tests for the verification package — NLI, judge, faithfulness, consistency."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from extremis.verification.consistency import _cosine, _pairwise_mean, self_consistency_filter
from extremis.verification.faithfulness import VerificationResult, verify
from extremis.verification.judge import JudgeVerdict, LLMJudge
from extremis.verification.nli import NLIResult

# ------------------------------------------------------------------ #
# Fake NLI for faithfulness orchestration tests — avoids loading transformers
# ------------------------------------------------------------------ #


class FakeNLI:
    def __init__(self, score: float, label: str = "ENTAILMENT", best_idx: int = 0):
        self._score = score
        self._label = label
        self._best_idx = best_idx
        self.calls = 0

    def entailment_score(self, claim: str, sources: list[str]) -> NLIResult:
        self.calls += 1
        return NLIResult(score=self._score, label=self._label, best_source_idx=self._best_idx)


def _make_judge(verdict: str, score: float, reason: str = ""):
    """Build a JudgeVerdict-returning mock without touching the Anthropic SDK."""
    judge = MagicMock(spec=LLMJudge)
    judge.judge.return_value = JudgeVerdict(verdict=verdict, score=score, reason=reason)
    return judge


class TestVerify:
    def test_clean_pass_skips_judge(self):
        nli = FakeNLI(score=0.95, label="ENTAILMENT", best_idx=2)
        judge = _make_judge("SUPPORTED", 1.0)
        result = verify("claim", ["a", "b", "c"], nli=nli, judge=judge)
        assert result.verdict == "SUPPORTED"
        assert result.method == "nli"
        assert result.score == pytest.approx(0.95)
        assert result.best_source_idx == 2
        judge.judge.assert_not_called()

    def test_grey_zone_escalates_to_judge(self):
        nli = FakeNLI(score=0.65, label="NEUTRAL")
        judge = _make_judge("CONTRADICTED", 0.1, reason="conflicts with line 2")
        result = verify("claim", ["src"], nli=nli, judge=judge)
        assert result.method == "nli+judge"
        assert result.verdict == "CONTRADICTED"
        assert result.score == pytest.approx(0.1)
        assert result.judge_reason == "conflicts with line 2"
        judge.judge.assert_called_once()

    def test_low_score_skips_judge(self):
        """Below grey_zone_low — don't waste a judge call on obvious fails."""
        nli = FakeNLI(score=0.2, label="NEUTRAL")
        judge = _make_judge("SUPPORTED", 0.9)
        result = verify("claim", ["src"], nli=nli, judge=judge, grey_zone_low=0.5)
        assert result.method == "nli"
        assert result.verdict in ("UNVERIFIABLE", "CONTRADICTED")
        judge.judge.assert_not_called()

    def test_contradiction_label_preserved(self):
        nli = FakeNLI(score=0.1, label="CONTRADICTION")
        judge = _make_judge("SUPPORTED", 0.9)
        result = verify("claim", ["src"], nli=nli, judge=judge)
        assert result.verdict == "CONTRADICTED"
        judge.judge.assert_not_called()

    def test_nli_unavailable_falls_back_to_judge(self):
        judge = _make_judge("SUPPORTED", 0.88)
        result = verify("claim", ["src"], nli=None, judge=judge)
        assert result.method == "judge-only"
        assert result.score == pytest.approx(0.88)
        judge.judge.assert_called_once()

    def test_both_unavailable_returns_skipped(self):
        result = verify("claim", ["src"], nli=None, judge=None)
        assert result.method == "skipped"

    def test_to_metadata_is_json_serializable(self):
        result = VerificationResult(
            score=0.9,
            verdict="SUPPORTED",
            method="nli",
            nli_score=0.9,
            verified_at="2025-01-01T00:00:00+00:00",
        )
        meta = result.to_metadata()
        json.dumps(meta)  # must not raise
        assert meta["verdict"] == "SUPPORTED"


# ------------------------------------------------------------------ #
# Judge — parser robustness
# ------------------------------------------------------------------ #


class TestLLMJudge:
    def _client_returning(self, text: str):
        block = MagicMock()
        block.text = text
        response = MagicMock()
        response.content = [block]
        client = MagicMock()
        client.messages.create.return_value = response
        return client

    def test_parses_clean_json(self):
        client = self._client_returning('{"verdict": "SUPPORTED", "score": 0.92, "reason": "directly stated"}')
        judge = LLMJudge(client, model="claude-haiku-4-5-20251001")
        verdict = judge.judge("claim", "context")
        assert verdict.verdict == "SUPPORTED"
        assert verdict.score == pytest.approx(0.92)
        assert verdict.reason == "directly stated"

    def test_strips_markdown_fences(self):
        client = self._client_returning('```json\n{"verdict": "CONTRADICTED", "score": 0.1}\n```')
        judge = LLMJudge(client, model="claude-haiku-4-5-20251001")
        verdict = judge.judge("claim", "context")
        assert verdict.verdict == "CONTRADICTED"
        assert verdict.score == pytest.approx(0.1)

    def test_handles_malformed_json(self):
        client = self._client_returning("not json at all")
        judge = LLMJudge(client, model="claude-haiku-4-5-20251001")
        verdict = judge.judge("claim", "context")
        assert verdict.verdict == "UNVERIFIABLE"
        assert verdict.score == pytest.approx(0.5)

    def test_clamps_score_to_unit_interval(self):
        client = self._client_returning('{"verdict": "SUPPORTED", "score": 5.0}')
        judge = LLMJudge(client, model="claude-haiku-4-5-20251001")
        verdict = judge.judge("claim", "context")
        assert verdict.score == 1.0

    def test_unknown_verdict_coerced_to_unverifiable(self):
        client = self._client_returning('{"verdict": "MAYBE", "score": 0.5}')
        judge = LLMJudge(client, model="claude-haiku-4-5-20251001")
        verdict = judge.judge("claim", "context")
        assert verdict.verdict == "UNVERIFIABLE"

    def test_api_exception_returns_unverifiable(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("network down")
        judge = LLMJudge(client, model="claude-haiku-4-5-20251001")
        verdict = judge.judge("claim", "context")
        assert verdict.verdict == "UNVERIFIABLE"


# ------------------------------------------------------------------ #
# Self-consistency filter
# ------------------------------------------------------------------ #


class TestSelfConsistency:
    def _embedder_from_map(self, embedding_map: dict[str, list[float]]):
        """Build a mock embedder whose embed(text) returns a fixed vector."""
        e = MagicMock()
        e.embed.side_effect = lambda text: embedding_map[text]
        return e

    def test_convergent_claims_kept(self):
        # All three samples emit the same claim text — embeddings are identical
        claim = {"layer": "semantic", "content": "User uses Python"}
        samples = [[claim], [claim], [claim]]
        i = iter(samples)

        def extract_fn():
            return next(i)

        embedder = self._embedder_from_map({"User uses Python": [1.0, 0.0]})

        kept, stats = self_consistency_filter(
            extract_fn,
            embedder=embedder,
            n=3,
            threshold=0.85,
            layers_in_scope={"semantic"},
        )
        assert len(kept) == 1
        assert kept[0]["_consistency"]["samples"] == 3
        assert kept[0]["_consistency"]["mean_similarity"] == pytest.approx(1.0)
        assert len(stats) == 1

    def test_divergent_claims_dropped(self):
        # Sample 0 says A; samples 1 and 2 say wildly different things with
        # orthogonal embeddings. Mean pairwise sim < 0.85.
        s0 = [{"layer": "semantic", "content": "A"}]
        s1 = [{"layer": "semantic", "content": "B"}]
        s2 = [{"layer": "semantic", "content": "C"}]
        samples = [s0, s1, s2]
        i = iter(samples)

        def extract_fn():
            return next(i)

        embedder = self._embedder_from_map({"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0], "C": [0.0, 0.0, 1.0]})

        kept, stats = self_consistency_filter(
            extract_fn,
            embedder=embedder,
            n=3,
            threshold=0.85,
            layers_in_scope={"semantic"},
        )
        assert kept == []
        assert stats[0].mean_similarity < 0.85

    def test_out_of_scope_layers_pass_through(self):
        """Episodic claims aren't in self_consistency_layers — keep them as-is."""
        claim = {"layer": "episodic", "content": "msg"}
        i = iter([[claim], [], []])

        def extract_fn():
            return next(i)

        embedder = MagicMock()

        kept, _ = self_consistency_filter(
            extract_fn,
            embedder=embedder,
            n=3,
            threshold=0.85,
            layers_in_scope={"semantic", "identity"},
        )
        assert kept == [{"layer": "episodic", "content": "msg"}]
        embedder.embed.assert_not_called()

    def test_n_le_1_short_circuits(self):
        """n=1 disables sampling — one extract call, no filtering."""
        called = {"count": 0}

        def extract_fn():
            called["count"] += 1
            return [{"layer": "semantic", "content": "x"}]

        kept, stats = self_consistency_filter(
            extract_fn,
            embedder=MagicMock(),
            n=1,
            threshold=0.85,
            layers_in_scope={"semantic"},
        )
        assert called["count"] == 1
        assert kept == [{"layer": "semantic", "content": "x"}]
        assert stats == []


class TestCosineHelpers:
    def test_cosine_identical(self):
        assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_cosine_handles_zero_vector(self):
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_pairwise_mean_single_vector(self):
        assert _pairwise_mean([[1.0, 0.0]]) == 1.0
