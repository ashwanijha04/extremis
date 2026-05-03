<div align="center">

# 🧠 friday-memory

**Layered, learning memory for AI agents**

*Log-first · RL-scored · Consolidation-backed · MCP-ready*

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-coming%20soon-orange?logo=pypi&logoColor=white)](https://pypi.org)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple?logo=anthropic&logoColor=white)](https://modelcontextprotocol.io)
[![Status](https://img.shields.io/badge/status-alpha-yellow)](https://github.com/ashwanijha04/friday-memory)

<br/>

> Add persistent, learning memory to any AI agent in two lines of config.  
> No vector database to run. No RAG pipeline to maintain.

<br/>

[**How it works**](#how-it-works) · [**Install**](#install) · [**Quick start**](#quick-start) · [**MCP setup**](#mcp-setup) · [**Configuration**](#configuration)

</div>

---

## What is friday-memory?

Most AI agents forget everything the moment the conversation ends. friday-memory gives agents a durable, layered memory that **learns from feedback** — the more you use it, the better it gets at surfacing what matters.

It handles the hard parts so you don't have to:

- ✅ **Embedding + semantic search** — no vector database to provision
- ✅ **Layered memory types** — episodic, semantic, procedural, identity
- ✅ **Reinforcement learning** — recalled memories get scored; useful ones surface first
- ✅ **Consolidation** — Claude Haiku distils conversation logs into structured facts overnight
- ✅ **Knowledge graph** — entities, relationships, and attributes for rich context
- ✅ **Namespace isolation** — one deployment, many users, no data leakage
- ✅ **MCP server** — plug into Claude Desktop or Claude Code in 30 seconds

---

## How it works

### The flow

```
Every conversation
─────────────────
  remember("user said X")    ──▶  JSONL log (fsync, durable)
                                   + episodic memory (embedded + stored)

  recall("topic")            ──▶  embed query
                                   → identity + procedural  (always included)
                                   → semantic + episodic    (ranked by score)
                                   ← ranked results

  report_outcome(ids, +1/-1) ──▶  adjust utility scores
                                   negative signals get 1.5× weight
                                   (mirrors human memory asymmetry)

Periodically
────────────
  consolidate()              ──▶  read log since last checkpoint
                                   → Claude Haiku extracts facts
                                   → written as semantic/procedural memories
                                   → checkpoint advanced (safe to re-run)
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR AGENT / APP                          │
│           FridayMemory.remember() · recall() · report_outcome()  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
   ┌───────────┐     ┌────────────┐    ┌─────────────┐
   │ FileLog   │     │MemoryStore │    │ KGStore     │
   │ (JSONL,   │     │            │    │             │
   │  fsync)   │     │ SQLite     │    │ Entities    │
   │           │     │   ──or──   │    │ Relations   │
   │ source    │     │ Postgres   │    │ Attributes  │
   │ of truth  │     │ +pgvector  │    │             │
   └─────┬─────┘     └────────────┘    └─────────────┘
         │
         ▼
   ┌───────────────────────────────┐
   │       LLM Consolidator        │
   │  (Claude Haiku, nightly/manual)│
   │  log entries → semantic facts  │
   │  → procedural rules            │
   └───────────────────────────────┘
```

### Memory layers

| Layer | What it holds | Written by | Always recalled? |
|-------|--------------|-----------|-----------------|
| `identity` | Who the user is — never auto-committed | Human review only | ✅ Yes |
| `procedural` | Behavioural rules: *"always ask about deadline first"* | Consolidator | ✅ Yes |
| `semantic` | Durable facts: *"user is a Python developer, works solo"* | Consolidator | By relevance |
| `episodic` | Timestamped conversation events | `remember()` | By relevance |
| `working` | Session-scoped, expires on a set datetime | `remember_now()` | By relevance |

### Ranking formula

Every recalled memory gets a `final_rank` score that balances three signals:

```
final_rank = cosine_similarity
           × (1 + α · tanh(utility_score))   ← RL signal
           × exp(−ln2 · age_days / half_life) ← recency decay
```

Memories that have been marked useful (`+1`) surface above equally relevant but untested memories. Negative signals (`−1`) apply **1.5× weight** — the same asymmetry human memory uses for threat learning.

### Knowledge graph

Beyond embeddings, friday-memory maintains a structured graph of entities and relationships:

```python
mem.kg_add_entity("Alice", EntityType.PERSON)
mem.kg_add_entity("Acme Corp", EntityType.ORG)
mem.kg_add_relationship("Alice", "Acme Corp", "works_at", weight=0.95)
mem.kg_add_attribute("Alice", "timezone", "Asia/Dubai")
mem.kg_add_attribute("Alice", "tone", "formal")

result = mem.kg_query("Alice")
# → Entity + all relationships + all attributes
```

### Attention scoring

Before generating a full response, score the incoming message to decide how much effort to apply:

```
score = sender_score + channel_score + content_score + context_score
        (0–100, pure heuristic, zero LLM cost)

full      ≥ 75  → engage fully
standard  ≥ 50  → balanced response
minimal   ≥ 25  → brief acknowledgement
ignore    < 25  → skip
```

### Observer (log compression)

Compresses raw log entries into priority-tagged observations — free, no LLM:

```
🔴 CRITICAL  decisions, errors, deadlines, shipped/launched, +1/-1 signals
🟡 CONTEXT   reasons, insights, learnings, "because", "discovered"
🟢 INFO      everything else
```

---

## Install

```bash
# Core: SQLite + sentence-transformers (local embeddings, no API key needed)
pip install friday-memory

# + MCP server for Claude Desktop / Claude Code
pip install "friday-memory[mcp]"

# + Postgres consolidated store (requires pgvector)
pip install "friday-memory[postgres]"

# Everything
pip install "friday-memory[all]"
```

**Requires Python 3.11+**

> **Note** — on first `recall()`, `sentence-transformers` downloads `all-MiniLM-L6-v2` (~90 MB). This is a one-time download cached to `~/.cache/huggingface/`.

---

## Quick start

```python
from friday_memory import FridayMemory, MemoryLayer

# Defaults to ~/.friday/ — zero config needed
mem = FridayMemory()

# ── Remember ─────────────────────────────────────────────────
mem.remember("User is building a WhatsApp AI called Friday", conversation_id="conv_001")
mem.remember("User prefers concise answers, hates filler words", conversation_id="conv_001")

# Direct write for time-sensitive or high-confidence facts
mem.remember_now(
    "User's flight departs Thursday at 06:00",
    layer=MemoryLayer.EPISODIC,
    confidence=0.99,
)

# ── Recall ───────────────────────────────────────────────────
results = mem.recall("what product is the user building?", limit=5)
for r in results:
    print(f"[{r.memory.layer.value}] {r.memory.content}  (rank={r.final_rank:.3f})")

# ── Feedback ─────────────────────────────────────────────────
useful_ids = [r.memory.id for r in results[:2]]
mem.report_outcome(useful_ids, success=True)

# ── Knowledge graph ──────────────────────────────────────────
from friday_memory.types import EntityType

mem.kg_add_entity("User", EntityType.PERSON)
mem.kg_add_entity("Friday", EntityType.PROJECT)
mem.kg_add_relationship("User", "Friday", "building", weight=1.0)
mem.kg_add_attribute("User", "timezone", "Asia/Dubai")

result = mem.kg_query("User")
print(result)

# ── Attention scorer ─────────────────────────────────────────
attention = mem.score_attention("URGENT: the API is down!", channel="dm")
print(attention.level)   # → "full"
print(attention.score)   # → 85

# ── Consolidation ────────────────────────────────────────────
from friday_memory.consolidation import LLMConsolidator

consolidator = LLMConsolidator(mem._config, mem._embedder)
result = consolidator.run_pass(mem.get_log(), mem.get_local_store(), mem.get_local_store())
print(f"{result.memories_created} memories extracted from logs")
```

---

## MCP setup

### Claude Desktop

```bash
pip install "friday-memory[mcp]"
```

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "friday-memory": {
      "command": "friday-memory-mcp",
      "env": {
        "FRIDAY_HOME": "~/.friday",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

Restart Claude Desktop. **Nine tools** appear automatically.

### Claude Code

```bash
claude mcp add friday-memory friday-memory-mcp \
  --env FRIDAY_HOME=~/.friday \
  --env ANTHROPIC_API_KEY=sk-ant-...
```

### SSE / HTTP mode (for non-Claude integrations)

```bash
friday-memory-mcp --transport sse --port 8765
```

### Available MCP tools

| Tool | What it does | LLM cost |
|------|-------------|---------|
| `memory_remember` | Append to log + episodic store | None |
| `memory_recall` | Semantic search, identity+procedural always included | None |
| `memory_report_outcome` | +1/−1 RL signal on recalled memories | None |
| `memory_remember_now` | Direct write to structured memory (bypass log) | None |
| `memory_consolidate` | Distil logs into semantic/procedural memories | Haiku |
| `memory_kg_write` | Add entity / relationship / attribute to KG | None |
| `memory_kg_query` | Query entity and its connections (+ BFS traverse) | None |
| `memory_observe` | Compress log into 🔴🟡🟢 priority observations | None |
| `memory_score_attention` | Score a message 0–100 (full/standard/minimal/ignore) | None |

---

## Multi-user / namespace isolation

friday-memory supports two isolation models:

**Instance-level** — each user runs their own process with their own `FRIDAY_HOME`. This is what Claude Desktop does naturally.

**Namespace-level** — one deployment, many users. All memories, logs, and graph data are scoped to the namespace. Nothing bleeds across.

```bash
# User A
FRIDAY_NAMESPACE=user_alice friday-memory-mcp

# User B — completely separate memory, same database
FRIDAY_NAMESPACE=user_bob friday-memory-mcp
```

```python
from friday_memory import FridayMemory, Config

mem = FridayMemory(config=Config(namespace="user_alice"))
```

---

## Storage backends

### SQLite (default — local, zero infrastructure)

```bash
FRIDAY_HOME=~/.friday  # DB at ~/.friday/local.db
```

### Postgres + pgvector (for scale / multi-device)

```bash
pip install "friday-memory[postgres]"
FRIDAY_STORE=postgres
FRIDAY_POSTGRES_URL=postgresql://user:pass@host/friday
```

Requires `CREATE EXTENSION vector;` in your Postgres database. The schema migration runs automatically on first start.

---

## Configuration

All settings via environment variables (prefix `FRIDAY_`) or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `FRIDAY_HOME` | `~/.friday` | Base directory for logs and SQLite DB |
| `FRIDAY_NAMESPACE` | `default` | User/agent isolation scope |
| `FRIDAY_STORE` | `sqlite` | Storage backend: `sqlite` or `postgres` |
| `FRIDAY_POSTGRES_URL` | *(empty)* | Postgres connection string (required if store=postgres) |
| `FRIDAY_EMBEDDER` | `all-MiniLM-L6-v2` | Sentence-transformers model name |
| `FRIDAY_EMBEDDING_DIM` | `384` | Vector dimension (must match model) |
| `FRIDAY_CONSOLIDATION_MODEL` | `claude-haiku-4-5-20251001` | Model for nightly consolidation |
| `FRIDAY_RL_ALPHA` | `0.5` | Weight of utility score in retrieval ranking |
| `FRIDAY_RECENCY_HALF_LIFE_DAYS` | `90` | Recency decay half-life in days |
| `FRIDAY_ATTENTION_FULL_THRESHOLD` | `75` | Score ≥ this → full attention |
| `FRIDAY_ATTENTION_STANDARD_THRESHOLD` | `50` | Score ≥ this → standard attention |
| `FRIDAY_ATTENTION_MINIMAL_THRESHOLD` | `25` | Score ≥ this → minimal attention |

---

## How it compares

| | friday-memory | Raw vector DB | LangChain Memory | Mem0 |
|--|--------------|--------------|-----------------|------|
| Zero infra (local) | ✅ | ❌ needs server | ✅ | ❌ hosted |
| Layered memory types | ✅ 5 layers | ❌ | ⚠️ basic | ⚠️ basic |
| RL scoring | ✅ | ❌ | ❌ | ⚠️ implicit |
| Knowledge graph | ✅ | ❌ | ❌ | ❌ |
| Consolidation (log → facts) | ✅ | ❌ | ❌ | ✅ |
| Attention scoring | ✅ | ❌ | ❌ | ❌ |
| Namespace isolation | ✅ | ⚠️ manual | ❌ | ✅ |
| MCP server | ✅ | ❌ | ❌ | ❌ |
| Postgres backend | ✅ | ✅ | ❌ | ✅ |
| Open source | ✅ | ✅ | ✅ | ⚠️ partial |

---

## Project structure

```
friday-memory/
├── src/friday_memory/
│   ├── api.py                  ← FridayMemory (the public API)
│   ├── config.py               ← Config (all settings, env var driven)
│   ├── types.py                ← Memory, LogEntry, Entity, Observation, ...
│   ├── interfaces.py           ← LogStore, MemoryStore, Embedder protocols
│   ├── storage/
│   │   ├── log.py              ← FileLogStore (JSONL, fsync, checkpoints)
│   │   ├── sqlite.py           ← SQLiteMemoryStore (cosine search in numpy)
│   │   ├── postgres.py         ← PostgresMemoryStore (pgvector, ranking in SQL)
│   │   └── kg.py               ← SQLiteKGStore (entities, relationships, attributes)
│   ├── embeddings/
│   │   └── sentence_transformers.py  ← lazy-loaded local model
│   ├── consolidation/
│   │   ├── consolidator.py     ← LLMConsolidator (log → Claude → memories)
│   │   └── prompts.py          ← extraction prompt templates
│   ├── observer/
│   │   └── observer.py         ← HeuristicObserver (🔴🟡🟢 classification)
│   ├── scorer/
│   │   └── attention.py        ← AttentionScorer (0–100, four levels)
│   └── mcp/
│       └── server.py           ← FastMCP server (9 tools)
├── migrations/
│   ├── 001_initial.sql         ← Postgres + pgvector schema
│   └── 001_initial_sqlite.sql  ← SQLite schema (mirrors Postgres)
└── tests/
```

---

## License

[MIT](LICENSE) · Built by [Ashwani Jha](https://github.com/ashwanijha04)

---

<div align="center">
  <sub>If friday-memory saves you from building yet another RAG pipeline, consider giving it a ⭐</sub>
</div>
