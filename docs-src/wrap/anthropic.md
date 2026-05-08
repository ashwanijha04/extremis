# wrap.Anthropic

Drop-in replacement for `anthropic.Anthropic` with automatic memory.

## Install

```bash
pip3.11 install "extremis[wrap-anthropic]"
```

## Basic usage

```python
from extremis.wrap import Anthropic
from extremis import Extremis

client = Anthropic(api_key="sk-ant-...", memory=Extremis())

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What's my name?"}]
)
print(response.content[0].text)
```

## Constructor

```python
Anthropic(
    memory=None,         # Extremis or HostedClient instance
    session_id=None,     # str — defaults to UUID per instance
    **kwargs             # passed to anthropic.Anthropic()
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `memory` | `Extremis \| HostedClient \| None` | Memory backend. If `None`, behaves identically to the plain Anthropic client |
| `session_id` | `str \| None` | Groups messages for consolidation. Use a stable user ID in production |
| `**kwargs` | any | Forwarded to `anthropic.Anthropic()` — `api_key`, `base_url`, `timeout`, etc. |

## What gets intercepted

Only `client.messages.create()` is intercepted. Everything else passes through to the underlying Anthropic client:

```python
# ✅ Intercepted — memory injected + saved
client.messages.create(...)

# ✅ Pass-through — no memory involvement
client.models.list()
client.beta.messages.count_tokens(...)
```

## Streaming

Streaming is **not yet intercepted** — the response passes through but messages are not saved to memory. Full streaming support is on the roadmap.

```python
# This works (streaming passes through) but memory is not saved
with client.messages.stream(...) as stream:
    for text in stream.text_stream:
        print(text)
```

## Tool use

Tool use works — extremis saves the final text content of the response. Tool calls themselves are not stored separately.

## With hosted memory

```python
from extremis.wrap import Anthropic
from extremis import HostedClient

client = Anthropic(
    api_key="sk-ant-...",
    memory=HostedClient(
        api_key="extremis_sk_...",
        base_url="https://your-server.onrender.com",
    ),
    session_id="user_alice",
)
```

## Context format

The recalled memories are prepended to the `system` parameter as:

```
[Relevant context from memory]
- User is a Python developer with 8 years experience
- User prefers concise answers, no filler words
- User is building a WhatsApp AI

[your existing system prompt follows]
```

If you pass a `system` parameter, it's appended after the memory context. If you don't, only the memory context is used as the system prompt.

## Error handling

Memory failures are silent — they never break the LLM call:

```python
# If recall fails → call proceeds without context
# If remember fails → response still returned
# Network errors → logged, not raised
```
