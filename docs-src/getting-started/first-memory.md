# Your first memory

This tutorial walks through every core feature in ~10 minutes.

## Setup

```bash
pip3.11 install extremis
python3.11
```

```python
from extremis import Extremis, MemoryLayer
from extremis.types import EntityType

mem = Extremis()  # stores in ~/.extremis/ — no config needed
```

## Store your first memory

```python
# Episodic: a specific event in a conversation
mem.remember(
    "User said they prefer dark mode in all tools",
    role="user",
    conversation_id="session_001",
)

# Semantic: a durable fact (write directly, skip the log)
mem.remember_now(
    "User is a Python developer with 8 years experience",
    layer=MemoryLayer.SEMANTIC,
    confidence=0.9,
)

# Procedural: a behavioural rule
mem.remember_now(
    "Always ask about the user's deadline before suggesting a solution",
    layer=MemoryLayer.PROCEDURAL,
)
```

## Recall

```python
results = mem.recall("what does the user prefer?", limit=5)

for r in results:
    print(f"[{r.memory.layer.value}] {r.memory.content}")
    print(f"  score={r.memory.score:+.1f}  reason: {r.reason}")
    print()
```

Output:
```
[procedural] Always ask about the user's deadline before suggesting a solution
  score=+0.0  reason: procedural (always included) · similarity 0.42 · first recall · today

[semantic] User is a Python developer with 8 years experience
  score=+0.0  reason: similarity 0.68 · first recall · today

[episodic] User said they prefer dark mode in all tools
  score=+0.0  reason: similarity 0.54 · first recall · today
```

Notice:
- **Procedural memories are always included** — they define how you should behave with this user
- Every result has a `reason` — you can see exactly why each memory ranked where it did

## Give feedback

After using recalled memories in a response, tell extremis whether they were helpful:

```python
# The response was good — mark the top 2 as useful
mem.report_outcome(
    [results[0].memory.id, results[1].memory.id],
    success=True,
    weight=1.0,
)

# Recall again — useful memories now rank higher
results2 = mem.recall("what does the user prefer?", limit=5)
for r in results2:
    print(f"score={r.memory.score:+.1f}  {r.memory.content[:50]}")
```

Output:
```
score=+1.0  Always ask about the user's deadline...
score=+1.0  User is a Python developer with 8 years...
score=+0.0  User said they prefer dark mode in all tools
```

!!! info "Asymmetric weighting"
    Negative feedback (`success=False`) applies **1.5× weight** — the same asymmetry human threat-memory uses. Mistakes fade faster than successes accumulate.

## Knowledge graph

Alongside vectors, extremis maintains a structured graph:

```python
# Add entities
mem.kg_add_entity("Alice", EntityType.PERSON)
mem.kg_add_entity("Acme Corp", EntityType.ORG)

# Connect them
mem.kg_add_relationship("Alice", "Acme Corp", "works_at", weight=0.95)

# Add attributes
mem.kg_add_attribute("Alice", "timezone", "Asia/Dubai")
mem.kg_add_attribute("Alice", "language", "Python")

# Query
result = mem.kg_query("Alice")
for rel in result.relationships:
    print(f"{rel.from_entity} → [{rel.rel_type}] → {rel.to_entity}")

# Graph traversal — 2 hops from Alice
graph = mem.kg_traverse("Alice", depth=2)
for entity in graph:
    print(f"[{entity.entity.type.value}] {entity.entity.name}")
```

## What's stored where

By default, extremis writes to `~/.extremis/`:

```
~/.extremis/
├── local.db          # SQLite: memories, KG, scores
└── log/
    └── default/
        └── 2026-05-08.jsonl   # Conversation log (fsync'd)
```

The JSONL log is the source of truth — plain text, readable with any editor, never deleted automatically.

## Next steps

- [**RL scoring in depth**](../concepts/rl-scoring.md) — understand the ranking formula
- [**Wrap your LLM client**](../wrap/index.md) — skip `remember`/`recall` entirely
- [**Run the demo**](../getting-started/quickstart.md) — `extremis-demo` shows everything live
