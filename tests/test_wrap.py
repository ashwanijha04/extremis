"""
Tests for extremis.wrap — LLM client wrappers.
All LLM calls are mocked; no real API keys needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from extremis import Extremis, MemoryLayer
from extremis.wrap.anthropic import Anthropic, _build_context_prefix, _extract_user_text
from extremis.wrap.openai import OpenAI, _inject_system

# ── helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mem(tmp_path, mock_embedder):
    from extremis.config import Config

    return Extremis(
        config=Config(
            extremis_home=str(tmp_path),
            log_dir=str(tmp_path / "log"),
            local_db_path=str(tmp_path / "local.db"),
        ),
        embedder=mock_embedder,
    )


def fake_anthropic_response(text: str) -> MagicMock:
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def fake_openai_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


# ── unit tests — pure helpers ─────────────────────────────────────────────────


class TestExtractUserText:
    def test_string_content(self):
        msgs = [{"role": "user", "content": "Hello there"}]
        assert _extract_user_text(msgs) == "Hello there"

    def test_list_content(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
        assert _extract_user_text(msgs) == "Hi"

    def test_last_user_message(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert _extract_user_text(msgs) == "second"

    def test_no_user_message(self):
        assert _extract_user_text([{"role": "assistant", "content": "hi"}]) == ""


class TestBuildContextPrefix:
    def test_empty(self):
        assert _build_context_prefix([]) == ""

    def test_with_results(self):
        r = MagicMock()
        r.memory.content = "User likes Python"
        prefix = _build_context_prefix([r])
        assert "[Relevant context from memory]" in prefix
        assert "User likes Python" in prefix


class TestInjectSystem:
    def test_inserts_system_when_absent(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = _inject_system(msgs, "context here")
        assert result[0]["role"] == "system"
        assert "context here" in result[0]["content"]

    def test_prepends_to_existing_system(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        result = _inject_system(msgs, "ctx")
        assert result[0]["role"] == "system"
        assert "ctx" in result[0]["content"]
        assert "You are helpful." in result[0]["content"]


# ── Anthropic wrapper ─────────────────────────────────────────────────────────


class TestAnthropicWrapper:
    def _make_client(self, mem):
        with patch("anthropic.Anthropic") as mock_cls:
            mock_inner = MagicMock()
            mock_cls.return_value = mock_inner
            client = Anthropic(api_key="sk-test", memory=mem, session_id="test-session")
        return client, mock_inner

    def test_passthrough_returns_response(self, mem):
        client, mock_inner = self._make_client(mem)
        expected = fake_anthropic_response("Hello!")
        mock_inner.messages.create.return_value = expected

        result = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result is expected

    def test_remembers_conversation(self, mem):
        client, mock_inner = self._make_client(mem)
        mock_inner.messages.create.return_value = fake_anthropic_response("I am Claude")

        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "who are you?"}],
        )

        log = mem.get_log()
        entries = log.read_since(None)
        contents = [e.content for e in entries]
        assert any("who are you?" in c for c in contents)
        assert any("I am Claude" in c for c in contents)

    def test_injects_context_into_system(self, mem):
        # Pre-store a memory
        mem.remember_now("User is called Ashwani", layer=MemoryLayer.SEMANTIC)

        client, mock_inner = self._make_client(mem)
        mock_inner.messages.create.return_value = fake_anthropic_response("Hi Ashwani")

        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "what is my name?"}],
        )

        call_kwargs = mock_inner.messages.create.call_args[1]
        system_sent = call_kwargs.get("system", "")
        assert "Ashwani" in system_sent

    def test_memory_failure_doesnt_break_call(self, mem):
        client, mock_inner = self._make_client(mem)
        expected = fake_anthropic_response("ok")
        mock_inner.messages.create.return_value = expected

        # Break the memory
        client.messages._memory = None

        result = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result is expected

    def test_passthrough_other_attributes(self, mem):
        client, mock_inner = self._make_client(mem)
        mock_inner.models = "models-obj"
        assert client.models == "models-obj"

    def test_no_memory_works(self):
        with patch("anthropic.Anthropic") as mock_cls:
            mock_inner = MagicMock()
            mock_cls.return_value = mock_inner
            client = Anthropic(api_key="sk-test", memory=None)

        mock_inner.messages.create.return_value = fake_anthropic_response("hi")
        result = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result is not None


openai = pytest.importorskip("openai", reason="openai not installed")


# ── OpenAI wrapper ────────────────────────────────────────────────────────────


class TestOpenAIWrapper:
    def _make_client(self, mem):
        with patch("openai.OpenAI") as mock_cls:
            mock_inner = MagicMock()
            mock_cls.return_value = mock_inner
            client = OpenAI(api_key="sk-test", memory=mem, session_id="test-session")
        return client, mock_inner

    def test_passthrough_returns_response(self, mem):
        client, mock_inner = self._make_client(mem)
        expected = fake_openai_response("Hello!")
        mock_inner.chat.completions.create.return_value = expected

        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result is expected

    def test_remembers_conversation(self, mem):
        client, mock_inner = self._make_client(mem)
        mock_inner.chat.completions.create.return_value = fake_openai_response("I am GPT")

        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "who are you?"}],
        )

        log = mem.get_log()
        entries = log.read_since(None)
        contents = [e.content for e in entries]
        assert any("who are you?" in c for c in contents)
        assert any("I am GPT" in c for c in contents)

    def test_injects_context_into_system(self, mem):
        mem.remember_now("User prefers Python", layer=MemoryLayer.SEMANTIC)

        client, mock_inner = self._make_client(mem)
        mock_inner.chat.completions.create.return_value = fake_openai_response("ok")

        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "what language do I prefer?"}],
        )

        call_kwargs = mock_inner.chat.completions.create.call_args[1]
        msgs_sent = call_kwargs.get("messages", [])
        system_msgs = [m for m in msgs_sent if m.get("role") == "system"]
        assert system_msgs
        assert "Python" in system_msgs[0]["content"]
