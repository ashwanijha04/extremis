"""Consolidator tests — Anthropic client is fully mocked."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from extremis.config import Config
from extremis.consolidation.consolidator import LLMConsolidator
from extremis.storage.log import FileLogStore
from extremis.storage.sqlite import SQLiteMemoryStore
from extremis.types import LogEntry, MemoryLayer


@pytest.fixture
def config(tmp_path):
    return Config(
        extremis_home=str(tmp_path),
        log_dir=str(tmp_path / "log"),
        local_db_path=str(tmp_path / "local.db"),
    )


@pytest.fixture
def log_store(config):
    return FileLogStore(config.resolved_log_dir())


@pytest.fixture
def memory_store(config):
    return SQLiteMemoryStore(config.resolved_local_db_path(), config)


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 384
    embedder.dim = 384
    return embedder


def make_llm_response(memories: list[dict]) -> MagicMock:
    """Build a fake Anthropic response object."""
    content_block = MagicMock()
    content_block.text = json.dumps({"memories": memories})
    response = MagicMock()
    response.content = [content_block]
    return response


class TestLLMConsolidator:
    def _make_consolidator(self, config, embedder):
        with patch("extremis.consolidation.consolidator.anthropic.Anthropic"):
            c = LLMConsolidator(config, embedder)
        return c

    def test_no_entries_returns_empty_result(self, config, log_store, memory_store, mock_embedder):
        consolidator = self._make_consolidator(config, mock_embedder)
        result = consolidator.run_pass(log_store, memory_store, memory_store)

        assert result.memories_created == 0
        assert result.memories_superseded == 0

    def test_extracts_semantic_memories(self, config, log_store, memory_store, mock_embedder):
        # Write a multi-turn conversation to the log
        for role, content in [
            ("user", "I've been using Python for 10 years"),
            ("assistant", "That's great! Any favourite frameworks?"),
            ("user", "FastAPI for APIs, always. Hate Flask."),
        ]:
            log_store.append(
                LogEntry(role=role, content=content, conversation_id="conv1")
            )

        extracted = [
            {"layer": "semantic", "content": "User has 10 years of Python experience", "confidence": 0.9},
            {"layer": "procedural", "content": "Prefer FastAPI over Flask when suggesting Python web frameworks", "confidence": 0.85},
        ]

        consolidator = self._make_consolidator(config, mock_embedder)
        consolidator._client.messages.create.return_value = make_llm_response(extracted)

        result = consolidator.run_pass(log_store, memory_store, memory_store)

        assert result.memories_created == 2
        stored = memory_store.list_recent()
        contents = [m.content for m in stored]
        assert "User has 10 years of Python experience" in contents
        assert "Prefer FastAPI over Flask when suggesting Python web frameworks" in contents

    def test_skips_single_message_conversations(self, config, log_store, memory_store, mock_embedder):
        log_store.append(LogEntry(role="user", content="hi", conversation_id="solo"))

        consolidator = self._make_consolidator(config, mock_embedder)
        result = consolidator.run_pass(log_store, memory_store, memory_store)

        # Never called because conversation has < 2 messages
        consolidator._client.messages.create.assert_not_called()
        assert result.memories_created == 0

    def test_handles_malformed_llm_json_gracefully(self, config, log_store, memory_store, mock_embedder):
        for i in range(3):
            log_store.append(LogEntry(role="user", content=f"msg {i}", conversation_id="conv1"))

        bad_response = MagicMock()
        bad_response.content = [MagicMock(text="not valid json at all")]

        consolidator = self._make_consolidator(config, mock_embedder)
        consolidator._client.messages.create.return_value = bad_response

        result = consolidator.run_pass(log_store, memory_store, memory_store)
        assert result.memories_created == 0  # graceful failure

    def test_advances_checkpoint_after_pass(self, config, log_store, memory_store, mock_embedder):
        assert log_store.get_checkpoint() is None

        for i in range(3):
            log_store.append(LogEntry(role="user", content=f"msg {i}", conversation_id="c1"))

        consolidator = self._make_consolidator(config, mock_embedder)
        consolidator._client.messages.create.return_value = make_llm_response([])

        consolidator.run_pass(log_store, memory_store, memory_store)

        assert log_store.get_checkpoint() is not None

    def test_subsequent_pass_only_processes_new_entries(self, config, log_store, memory_store, mock_embedder):
        for i in range(3):
            log_store.append(LogEntry(role="user", content=f"old msg {i}", conversation_id="old"))

        consolidator = self._make_consolidator(config, mock_embedder)
        consolidator._client.messages.create.return_value = make_llm_response([])
        consolidator.run_pass(log_store, memory_store, memory_store)
        first_call_count = consolidator._client.messages.create.call_count

        # Add new entries AFTER the checkpoint
        for i in range(3):
            log_store.append(LogEntry(role="user", content=f"new msg {i}", conversation_id="new"))

        consolidator._client.messages.create.return_value = make_llm_response([])
        consolidator.run_pass(log_store, memory_store, memory_store)
        second_call_count = consolidator._client.messages.create.call_count

        # Second pass should make exactly one more call (the new conversation)
        assert second_call_count == first_call_count + 1

    def test_memories_stored_with_correct_layer(self, config, log_store, memory_store, mock_embedder):
        for role, content in [("user", "I prefer dark mode"), ("assistant", "Got it.")]:
            log_store.append(LogEntry(role=role, content=content, conversation_id="c"))

        consolidator = self._make_consolidator(config, mock_embedder)
        consolidator._client.messages.create.return_value = make_llm_response([
            {"layer": "semantic", "content": "User prefers dark mode", "confidence": 0.95},
        ])
        consolidator.run_pass(log_store, memory_store, memory_store)

        semantics = memory_store.list_recent(layer=MemoryLayer.SEMANTIC)
        assert len(semantics) == 1
        assert semantics[0].layer == MemoryLayer.SEMANTIC
        assert semantics[0].confidence == 0.95
