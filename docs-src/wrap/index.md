# Wrap — automatic memory for any LLM client

The wrap module gives you persistent, learning memory without changing any of your application logic.

**Before:**
```python
import anthropic
client = anthropic.Anthropic(api_key="sk-ant-...")
```

**After:**
```python
from extremis.wrap import Anthropic
from extremis import Extremis

client = Anthropic(api_key="sk-ant-...", memory=Extremis())
```

That's the entire change. Every `client.messages.create()` call now:

1. **Before the call** — recalls relevant memories and injects them as system context
2. **After the call** — saves both the user message and the assistant reply

The return type, method signatures, streaming, tool use — everything is unchanged.

---

## How it works

```
User message
    │
    ▼
extremis.recall(message, limit=5)
    │  → retrieves top memories
    │  → formats as "[Relevant context from memory]\n- fact 1\n- fact 2"
    │  → prepends to system prompt
    ▼
LLM API call  ←── your original call, unchanged
    │
    ▼
extremis.remember(user_message, role="user")
extremis.remember(assistant_reply, role="assistant")
    │
    ▼
Response returned to your code  ←── identical to what the SDK would return
```

**Memory failure is silent.** If recall or remember fails for any reason, the LLM call goes through normally. Memory never breaks your application.

---

## Session ID

The `session_id` groups messages for consolidation — all calls under the same session are treated as one conversation.

```python
# Default: new UUID per client instance (one session per client object)
client = Anthropic(api_key="...", memory=mem)

# Per-user: stable ID across restarts
client = Anthropic(api_key="...", memory=mem, session_id="user_alice")

# Per-conversation: new session per chat window
import uuid
client = Anthropic(api_key="...", memory=mem, session_id=str(uuid.uuid4()))
```

---

## Multi-user setup

Use a separate `Extremis` instance per user (different namespace), or one shared instance with different `session_id` values:

```python
# Option A: namespace isolation (recommended for production)
from extremis import Extremis, Config
from extremis.wrap import Anthropic

def get_client(user_id: str):
    return Anthropic(
        api_key="sk-ant-...",
        memory=Extremis(config=Config(namespace=user_id)),
        session_id=user_id,
    )

alice_client = get_client("alice")
bob_client   = get_client("bob")
# Alice and Bob's memories never mix
```

---

## Accessing memory directly

The underlying `Extremis` instance is still fully accessible:

```python
mem = Extremis()
client = Anthropic(api_key="...", memory=mem)

# All extremis features still work
mem.kg_add_entity("Alice", EntityType.PERSON)
mem.compact()  # reconcile contradictions
results = mem.recall("what does Alice like?")
```

---

## Supported clients

| Client | Import | Extra |
|--------|--------|-------|
| Anthropic (Claude) | `from extremis.wrap import Anthropic` | `extremis[wrap-anthropic]` |
| OpenAI | `from extremis.wrap import OpenAI` | `extremis[wrap-openai]` |

See individual pages for details:

- [**wrap.Anthropic**](anthropic.md)
- [**wrap.OpenAI**](openai.md)
