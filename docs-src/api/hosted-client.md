# HostedClient

HTTP client for the extremis hosted server. Same API as `Extremis`, no local database or model download.

## Install

```bash
pip3.11 install "extremis[client]"
```

## Usage

```python
from extremis import HostedClient

mem = HostedClient(
    api_key="extremis_sk_...",
    base_url="https://your-server.onrender.com",
    timeout=30.0,   # seconds
)

mem.remember("User is building extremis", conversation_id="c1")
results = mem.recall("what is the user building?")
mem.report_outcome([results[0].memory.id], success=True)
```

## Constructor

```python
HostedClient(
    api_key: str,
    base_url: str = "https://api.extremis.com",  # cloud (coming soon)
    timeout: float = 30.0,
)
```

## Methods

`HostedClient` exposes the same methods as `Extremis`:

- `remember(content, role, conversation_id, metadata)`
- `recall(query, limit, layers, min_score)`
- `report_outcome(memory_ids, success, weight)`
- `remember_now(content, layer, expires_at, confidence, metadata)`
- `observe(conversation_id)`
- `consolidate()` → calls the server's consolidation endpoint
- `compact(layer)` → calls the server's compaction endpoint
- `kg_add_entity(name, type, metadata)`
- `kg_add_relationship(from_entity, to_entity, rel_type, weight, metadata)`
- `kg_add_attribute(entity, key, value)`
- `kg_query(name)`
- `kg_traverse(name, depth)`
- `score_attention(message, sender, channel, owner_ids, allowlist, context)`

## With wrap

```python
from extremis.wrap import Anthropic
from extremis import HostedClient

client = Anthropic(
    api_key="sk-ant-...",
    memory=HostedClient(api_key="extremis_sk_...", base_url="https://..."),
    session_id="user_alice",
)
```

## Context manager

```python
with HostedClient(api_key="...", base_url="...") as mem:
    mem.remember("test")
# HTTP connection closed automatically
```
