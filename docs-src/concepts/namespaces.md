# Namespaces

Namespaces isolate one user or agent's memories from another. Everything — memories, logs, knowledge graph — is scoped to the namespace.

## Default namespace

Without configuration, all memories go into the `default` namespace:

```python
mem = Extremis()  # namespace="default"
```

## Per-user isolation

Set a different namespace per user to keep their memories completely separate:

```python
from extremis import Extremis, Config

def get_memory(user_id: str) -> Extremis:
    return Extremis(config=Config(namespace=user_id))

alice_mem = get_memory("alice")
bob_mem   = get_memory("bob")

alice_mem.remember("Alice prefers Python", conversation_id="a1")
bob_mem.remember("Bob prefers Go", conversation_id="b1")

# Alice's memory cannot see Bob's, and vice versa
assert not any("Go" in r.memory.content for r in alice_mem.recall("language preference"))
```

## Via environment variable

```bash
EXTREMIS_NAMESPACE=alice python3.11 your_app.py
```

## With the wrap

```python
from extremis.wrap import Anthropic
from extremis import Extremis, Config

def get_client(user_id: str):
    return Anthropic(
        api_key="sk-ant-...",
        memory=Extremis(config=Config(namespace=user_id)),
        session_id=user_id,
    )
```

## With the hosted server

The hosted server derives the namespace from the API key. Each key has an associated namespace set at creation:

```bash
extremis-server create-key --namespace alice --label "alice prod"
# → extremis_sk_...  (this key only sees alice's memories)
```

```python
from extremis import HostedClient

alice_client = HostedClient(api_key="extremis_sk_alice...", base_url="...")
bob_client   = HostedClient(api_key="extremis_sk_bob...", base_url="...")
```

## Cross-agent shared memory

Multiple agents can share a namespace — this is how agent teams work:

```python
research_agent = Extremis(config=Config(namespace="team_alpha"))
writing_agent  = Extremis(config=Config(namespace="team_alpha"))

# Research agent stores findings
research_agent.remember("GPT-4 outperforms on math by 12%")

# Writing agent recalls them automatically
results = writing_agent.recall("GPT-4 performance")
print(results[0].memory.content)  # → "GPT-4 outperforms on math by 12%"
```

## Storage

For SQLite (default), each namespace gets its own log directory:

```
~/.extremis/
├── local.db        # all memories, namespaced by column
└── log/
    ├── alice/
    │   └── 2026-05-08.jsonl
    └── bob/
        └── 2026-05-08.jsonl
```

For Postgres, memories are namespaced by a `namespace` column in the `memories` table.
