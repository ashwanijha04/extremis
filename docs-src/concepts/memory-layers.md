# Memory layers

extremis organises memory into five distinct layers, each with different retention and retrieval rules.

## The five layers

| Layer | Written by | Always recalled? | Expires? | Use for |
|-------|-----------|-----------------|---------|---------|
| `identity` | Human review only | ✅ Always | Never | Who the user fundamentally is |
| `procedural` | Consolidator / `remember_now` | ✅ Always | Never | Behavioural rules |
| `semantic` | Consolidator / `remember_now` | By relevance | Never | Durable facts |
| `episodic` | `remember()` | By relevance | Never | Timestamped events |
| `working` | `remember_now(expires_at=...)` | By relevance | ✅ Yes | Temporary session state |

### identity

The highest-trust layer. Facts here are **always** returned on every recall — they define who the user is.

```python
mem.remember_now(
    "User is Ashwani Jha, a senior engineer based in Dubai",
    layer=MemoryLayer.IDENTITY,
    confidence=1.0,
)
```

!!! warning "Never auto-committed"
    The consolidator proposes identity updates but never commits them automatically. Identity memories must be written explicitly with `remember_now()`. This prevents an agent from incorrectly overwriting who the user is.

### procedural

Behavioural rules for the agent. **Always included** in recall results regardless of the query.

```python
mem.remember_now(
    "Always ask about the user's deadline before suggesting a solution",
    layer=MemoryLayer.PROCEDURAL,
)
```

Use this for rules that should govern every interaction: tone, format preferences, things to always or never do.

### semantic

Durable facts about the user or the world. Retrieved by relevance.

```python
mem.remember_now(
    "User is a Python developer, prefers FastAPI over Flask",
    layer=MemoryLayer.SEMANTIC,
    confidence=0.9,
)
```

Semantic memories are the primary output of consolidation — the consolidator reads conversation logs and extracts distilled facts into this layer.

### episodic

Timestamped events from conversations. Written by `remember()` automatically.

```python
mem.remember(
    "User mentioned their flight departs Thursday at 06:00",
    conversation_id="session_42",
)
```

Episodic memories decay in ranking over time (controlled by `EXTREMIS_RECENCY_HALF_LIFE_DAYS`). Old episodic facts fade unless they're confirmed useful via `report_outcome()`.

### working

Temporary memories that expire at a set time. Useful for session-scoped state.

```python
from datetime import datetime, timezone, timedelta

mem.remember_now(
    "User is currently in a meeting until 3pm",
    layer=MemoryLayer.WORKING,
    expires_at=datetime.now(tz=timezone.utc) + timedelta(hours=2),
)
```

## Retrieval behaviour

When you call `mem.recall(query)`:

1. **Pinned search** — always fetches identity + procedural, regardless of query
2. **Ranked search** — fetches semantic + episodic, ranked by `cosine × RL score × recency`
3. **Merge** — pinned results first (deduped), then ranked, up to `limit`

To filter by specific layers:

```python
# Only semantic memories
results = mem.recall("user preferences", layers=[MemoryLayer.SEMANTIC])

# Identity and procedural only
results = mem.recall("who is this user?", layers=[MemoryLayer.IDENTITY, MemoryLayer.PROCEDURAL])
```

!!! note "Filtering identity/procedural"
    When you pass `layers=[MemoryLayer.IDENTITY]`, you get identity memories (from the pinned search) plus zero ranked results. Procedural is always included in pinned results regardless of the `layers` filter — this matches the design intent that procedural rules are always active.
