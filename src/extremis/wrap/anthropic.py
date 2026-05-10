"""
extremis.wrap.Anthropic — drop-in replacement for anthropic.Anthropic.

Intercepts messages.create() to inject recalled memories into the system
prompt and persist the conversation automatically. Everything else passes
through untouched: streaming, tool use, models.list, etc.

Usage:
    from extremis.wrap import Anthropic
    from extremis import Extremis

    client = Anthropic(api_key="sk-ant-...", memory=Extremis())
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "What's my name?"}]
    )
    # memory injected and saved automatically
"""

from __future__ import annotations

import uuid
from typing import Any, Optional


def _extract_user_text(messages: list[dict]) -> str:
    """Return the text of the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return " ".join(parts)
    return ""


def _extract_assistant_text(response: Any) -> str:
    """Extract text from an Anthropic Message response."""
    try:
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
    except Exception:
        pass
    return ""


class _StreamWrapper:
    """Wraps an Anthropic stream, collecting assistant text and saving to memory on exhaustion."""

    def __init__(self, stream: Any, memory: Any, user_text: str, session_id: str) -> None:
        self._stream = stream
        self._memory = memory
        self._user_text = user_text
        self._session_id = session_id
        self._collected: list[str] = []
        self._saved = False

    def __iter__(self) -> "_StreamWrapper":
        return self

    def __next__(self) -> Any:
        try:
            event = next(self._stream.__iter__() if not hasattr(self._stream, "__next__") else self._stream)
            if getattr(event, "type", None) == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta and hasattr(delta, "text"):
                    self._collected.append(delta.text)
            return event
        except StopIteration:
            self._flush()
            raise

    def _flush(self) -> None:
        if self._saved or not self._memory:
            return
        self._saved = True
        try:
            self._memory.remember(self._user_text, role="user", conversation_id=self._session_id)
            text = "".join(self._collected)
            if text:
                self._memory.remember(text, role="assistant", conversation_id=self._session_id)
        except Exception:
            pass

    def __enter__(self) -> "_StreamWrapper":
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        self._flush()
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(*args)
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _build_context_prefix(results: list) -> str:
    """Format recalled memories as a readable context block."""
    if not results:
        return ""
    lines = ["[Relevant context from memory]"]
    for r in results:
        lines.append(f"- {r.memory.content}")
    return "\n".join(lines)


class _MessagesWrapper:
    """Wraps the anthropic.messages namespace."""

    def __init__(self, messages: Any, memory: Any, session_id: str) -> None:
        self._messages = messages
        self._memory = memory
        self._session_id = session_id

    def create(self, *, messages: list[dict], system: str = "", **kwargs: Any) -> Any:
        user_text = _extract_user_text(messages)

        # Inject recalled context into system prompt
        if user_text and self._memory is not None:
            try:
                results = self._memory.recall(user_text, limit=5)
                prefix = _build_context_prefix(results)
                if prefix:
                    system = f"{prefix}\n\n{system}".strip() if system else prefix
            except Exception:
                pass  # memory failure must never break the LLM call

        stream = kwargs.get("stream", False)
        response = self._messages.create(messages=messages, system=system, **kwargs)

        if stream and user_text and self._memory is not None:
            return _StreamWrapper(response, self._memory, user_text, self._session_id)

        # Persist conversation for non-streaming responses
        if user_text and self._memory is not None:
            try:
                self._memory.remember(user_text, role="user", conversation_id=self._session_id)
                assistant_text = _extract_assistant_text(response)
                if assistant_text:
                    self._memory.remember(
                        assistant_text,
                        role="assistant",
                        conversation_id=self._session_id,
                    )
            except Exception:
                pass  # memory failure must never break the return

        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class Anthropic:
    """
    Drop-in replacement for anthropic.Anthropic with automatic memory.

    Args:
        memory:     Extremis or HostedClient instance. If None, behaves
                    identically to the plain anthropic.Anthropic client.
        session_id: Groups messages for consolidation. Defaults to a
                    fresh UUID per client instance (one session per client).
        **kwargs:   Passed directly to anthropic.Anthropic().
    """

    def __init__(
        self,
        memory: Optional[Any] = None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install 'extremis[wrap-anthropic]'") from None

        self._client = _anthropic.Anthropic(**kwargs)
        self._memory = memory
        self._session_id = session_id or str(uuid.uuid4())
        self.messages = _MessagesWrapper(self._client.messages, self._memory, self._session_id)

    def __getattr__(self, name: str) -> Any:
        """Pass everything else through to the underlying client."""
        return getattr(self._client, name)
