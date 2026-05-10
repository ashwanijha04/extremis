"""Verify peekr observability integration — no real API calls."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from extremis import Extremis
from extremis.config import Config


def _fake_response(text: str, input_tokens: int = 50, output_tokens: int = 20) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    block = MagicMock()
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.usage = usage
    return resp


class TestPeekrObservability:
    def test_observe_false_creates_no_traces(self, tmp_path, mock_embedder):
        config = Config(extremis_home=str(tmp_path), observe=False)
        mem = Extremis(config=config, embedder=mock_embedder)
        mem.remember("test fact", conversation_id="c1")
        mem.recall("test", limit=3)
        traces = Path(config.resolved_traces_path())
        assert not traces.exists()

    def test_observe_true_creates_traces_file(self, tmp_path, mock_embedder):
        config = Config(extremis_home=str(tmp_path), observe=True)
        # reset global peekr state so it re-instruments
        import extremis.api as api_mod

        api_mod._peekr_instrumented = False

        mem = Extremis(config=config, embedder=mock_embedder)
        mem.remember("user builds AI agents", conversation_id="c1")
        mem.recall("AI agents", limit=3)

        traces = Path(config.resolved_traces_path())
        assert traces.exists(), "traces.jsonl should be created when observe=True"
        spans = [json.loads(line) for line in traces.read_text().splitlines() if line.strip()]
        names = [s["name"] for s in spans]
        assert "extremis.remember" in names
        assert "extremis.recall" in names

    def test_spans_have_duration(self, tmp_path, mock_embedder):
        config = Config(extremis_home=str(tmp_path), observe=True)
        import extremis.api as api_mod

        api_mod._peekr_instrumented = False

        mem = Extremis(config=config, embedder=mock_embedder)
        mem.remember("some fact", conversation_id="c1")

        traces = Path(config.resolved_traces_path())
        spans = [json.loads(line) for line in traces.read_text().splitlines() if line.strip()]
        remember_spans = [s for s in spans if s["name"] == "extremis.remember"]
        assert remember_spans
        assert remember_spans[0]["duration_ms"] is not None
        assert remember_spans[0]["duration_ms"] >= 0

    def test_consolidation_spans_captured(self, tmp_path, mock_embedder):
        """Consolidation LLM calls should produce spans with token counts."""
        config = Config(extremis_home=str(tmp_path), observe=True)
        import extremis.api as api_mod

        api_mod._peekr_instrumented = False

        mem = Extremis(config=config, embedder=mock_embedder)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = _fake_response(
                '{"memories": [{"layer": "semantic", "content": "User builds agents", "confidence": 0.9}]}',
                input_tokens=150,
                output_tokens=40,
            )
            from extremis.consolidation.consolidator import LLMConsolidator

            for i in range(3):
                mem.remember(f"turn {i}", conversation_id="c1")

            consolidator = LLMConsolidator(config, mock_embedder)
            consolidator.run_pass(mem.get_log(), mem.get_local_store(), mem.get_local_store())

        traces = Path(config.resolved_traces_path())
        spans = [json.loads(line) for line in traces.read_text().splitlines() if line.strip()]
        extract_spans = [s for s in spans if "extract" in s["name"]]
        assert extract_spans, "consolidation extract should produce a span"

    def test_benchmark_traces_cost_summary(self, tmp_path):
        """Simulate reading a benchmark traces file and computing cost."""
        traces_file = tmp_path / "bench_traces.jsonl"
        fake_spans = [
            {
                "name": "anthropic.messages",
                "duration_ms": 320.0,
                "attributes": {"model": "claude-haiku-4-5-20251001", "tokens_input": 600, "tokens_output": 100},
            },
            {
                "name": "anthropic.messages",
                "duration_ms": 180.0,
                "attributes": {"model": "claude-haiku-4-5-20251001", "tokens_input": 200, "tokens_output": 5},
            },
        ]
        with open(traces_file, "w") as f:
            for span in fake_spans:
                f.write(json.dumps(span) + "\n")

        total_input = total_output = 0
        for line in traces_file.read_text().splitlines():
            span = json.loads(line)
            attrs = span.get("attributes", {})
            total_input += attrs.get("tokens_input", 0)
            total_output += attrs.get("tokens_output", 0)

        cost = (total_input / 1_000_000 * 0.80) + (total_output / 1_000_000 * 4.00)
        assert total_input == 800
        assert total_output == 105
        assert cost == pytest.approx((800 / 1_000_000 * 0.80) + (105 / 1_000_000 * 4.00), rel=1e-3)
