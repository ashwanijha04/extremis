# wrap.OpenAI

Drop-in replacement for `openai.OpenAI` with automatic memory.

## Install

```bash
pip3.11 install "extremis[wrap-openai]"
```

## Basic usage

```python
from extremis.wrap import OpenAI
from extremis import Extremis

client = OpenAI(api_key="sk-...", memory=Extremis())

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's my name?"}]
)
print(response.choices[0].message.content)
```

## Constructor

```python
OpenAI(
    memory=None,         # Extremis or HostedClient instance
    session_id=None,     # str — defaults to UUID per instance
    **kwargs             # passed to openai.OpenAI()
)
```

## What gets intercepted

Only `client.chat.completions.create()` is intercepted:

```python
# ✅ Intercepted
client.chat.completions.create(...)

# ✅ Pass-through
client.models.list()
client.embeddings.create(...)
client.images.generate(...)
```

## Context injection

Recalled memories are injected as a `system` message prepended to the messages list:

```python
# Your call
messages=[{"role": "user", "content": "What's my name?"}]

# What gets sent to OpenAI
messages=[
    {"role": "system", "content": "[Relevant context from memory]\n- User's name is Ashwani\n- ..."},
    {"role": "user", "content": "What's my name?"}
]
```

If you already have a system message, the memory context is prepended to it.

## With existing system prompt

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Always respond in English."},
        {"role": "user", "content": "What did I tell you last time?"}
    ]
)
# System sent to OpenAI:
# "[Relevant context from memory]\n- ...\n\nYou are a helpful assistant. Always respond in English."
```
