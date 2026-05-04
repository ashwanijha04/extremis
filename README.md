<div align="center">

# 🧠 lore-ai

**Layered, learning memory for AI agents**

*Log-first · RL-scored · Consolidation-backed · Backend-agnostic · MCP-ready*

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/lore-ai?logo=pypi&logoColor=white&color=orange)](https://pypi.org/project/lore-ai)
[![CI](https://img.shields.io/github/actions/workflow/status/ashwanijha04/lore-ai/ci.yml?label=CI&logo=github)](https://github.com/ashwanijha04/lore-ai/actions)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple?logo=anthropic&logoColor=white)](https://modelcontextprotocol.io)

<br/>

> Add persistent, learning memory to any AI agent — local, self-hosted, or cloud.  
> Same API regardless of where your vectors live.

<br/>

[**How it works**](#how-it-works) · [**Install**](#install) · [**Quick start**](#quick-start) · [**Backends**](#storage-backends) · [**Hosted API**](#hosted-api) · [**MCP**](#mcp-setup) · [**Migrate**](#migrating-backends) · [**Config**](#configuration)

</div>

---

## What is lore-ai?

Most AI agents forget everything the moment the conversation ends. lore-ai gives agents a durable, layered memory that **learns from feedback** and works with any vector store you already have.

**Three ways to use it — same API for all three:**

```python
# 1. Local (zero infra, great for development)
from lore_ai import FridayMemory
mem = FridayMemory()

# 2. Bring your own backend (Pinecone, Chroma, Postgres, ...)
from lore_ai import FridayMemory, Config
mem = FridayMemory(config=Config(store="pinecone", pinecone_api_key="..."))

# 3. Hosted cloud (no local setup, no model download)
from lore_ai import HostedClient
mem = HostedClient(api_key="lore_sk_...")
```

---

## How it works

### The intelligence layer

lore-ai sits **above** your vector store. RL scoring, the knowledge graph, consolidation, and attention scoring are all backend-independent — they work the same whether your vectors are in SQLite, Pinecone, or Chroma.

```
┌────────────────────────────────────────────────────────────────┐
│                     YOUR APP / AGENT                            │
│      remember() · recall() · report_outcome() · kg_*()         │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────┐
│                  LORE-AI INTELLIGENCE LAYER                      │
│   RL scoring · Knowledge graph · Consolidation · Observer       │
│   Attention scorer · Namespace isolation · Log durability       │
└────┬──────────────┬──────────────┬──────────────┬──────────────┘
     │              │              │              │
┌────▼───┐   ┌──────▼──┐   ┌──────▼──┐   ┌──────▼──┐
│SQLite  │   │Postgres │   │  Chroma  │   │Pinecone │
│(local) │   │+pgvector│   │ (local)  │   │(hosted) │
└────────┘   └─────────┘   └──────────┘   └─────────┘
```

### The memory flow

```
Every conversation
─────────────────
  remember("user said X")     ──▶  fsync to JSONL log (durable)
                                    + episodic memory (embedded + stored)

  recall("topic")             ──▶  embed query
                                    → identity + procedural  (always included)
                                    → semantic + episodic    (ranked by score)
                                    ← ranked results

  report_outcome(ids, +1/-1)  ──▶  adjust utility scores
                                    negative gets 1.5× weight (human memory bias)

Periodically
────────────
  consolidate()               ──▶  read log since last checkpoint
                                    → Claude Haiku extracts facts
                                    → semantic/procedural memories written
                                    → checkpoint advanced (safe to re-run)
```

### Retrieval ranking

Every recalled memory gets a `final_rank` that balances three signals:

```
final_rank = cosine_similarity
           × (1 + α · tanh(utility_score))   ← learned from feedback
           × exp(−ln2 · age_days / half_life) ← recency decay
```

A memory that has proven useful (`+1` feedback) ranks above an equally similar but unvalidated memory. Negative signals apply **1.5× weight** — the same asymmetry human threat-learning uses.

### Memory layers

| Layer | What it holds | Written by | Always recalled? |
|-------|--------------|-----------|-----------------|
| `identity` | Who the user fundamentally is | Human review only | ✅ Always |
| `procedural` | Behavioural rules: *"ask about deadline first"* | Consolidator | ✅ Always |
| `semantic` | Durable facts: *"user is a solo Python developer"* | Consolidator | By relevance |
| `episodic` | Timestamped conversation events | `remember()` | By relevance |
| `working` | Session-scoped, expires on a set datetime | `remember_now()` | By relevance |

### Knowledge graph

Beyond vectors, lore-ai maintains a structured graph — answers structural questions that semantic search can't:

```python
mem.kg_add_entity("Alice", EntityType.PERSON)
mem.kg_add_entity("Acme Corp", EntityType.ORG)
mem.kg_add_relationship("Alice", "Acme Corp", "works_at", weight=0.95)
mem.kg_add_attribute("Alice", "timezone", "Asia/Dubai")
mem.kg_add_attribute("Alice", "tone", "formal")

# "Who does Alice work for?" — can't answer with cosine similarity alone
result = mem.kg_query("Alice")
# → Entity + all relationships + all attributes + BFS traverse

# Two-hop traverse
graph = mem.kg_traverse("Alice", depth=2)
```

### Attention scoring

Before deciding how much to engage with an incoming message, score it — free, zero LLM cost:

```
score = sender_score + channel_score + content_score + context_score  (0–100)

full      ≥ 75  → engage fully
standard  ≥ 50  → balanced response  
minimal   ≥ 25  → brief acknowledgement
ignore    < 25  → skip
```

### Observer (log compression)

Compresses raw log entries into priority-tagged observations — no LLM, runs instantly:

```
🔴 CRITICAL  decisions, errors, deadlines, shipped/launched, reward signals
🟡 CONTEXT   reasons, insights, learnings, "because", "discovered"
🟢 INFO      everything else
```

---

## Install

```bash
# Core — SQLite + local sentence-transformers (no API key needed)
pip install lore-ai

# + MCP server (Claude Desktop / Code)
pip install "lore-ai[mcp]"

# + Postgres backend
pip install "lore-ai[postgres]"

# + Chroma backend
pip install "lore-ai[chroma]"

# + Pinecone backend
pip install "lore-ai[pinecone]"

# + OpenAI embeddings (swap out the 90 MB model download)
pip install "lore-ai[openai]"

# + Hosted API server
pip install "lore-ai[server]"

# + Python SDK for hosted cloud
pip install "lore-ai[client]"

# Everything
pip install "lore-ai[all]"
```

**Requires Python 3.11+**

> **First run note** — `sentence-transformers` downloads `all-MiniLM-L6-v2` (~90 MB) on first use. One-time, cached to `~/.cache/huggingface/`. To skip it, use OpenAI embeddings: `LORE_EMBEDDER=text-embedding-3-small`.

---

## Quick start

```python
from lore_ai import FridayMemory, MemoryLayer
from lore_ai.types import EntityType

mem = FridayMemory()  # ~/.lore/ by default

# ── Remember ──────────────────────────────────────────────────
mem.remember("User is building a WhatsApp AI", conversation_id="conv_001")
mem.remember("User prefers concise answers", conversation_id="conv_001")

# Skip the log for time-sensitive or high-confidence facts
mem.remember_now(
    "Flight departs Thursday at 06:00",
    layer=MemoryLayer.EPISODIC,
    confidence=0.99,
)

# ── Recall ────────────────────────────────────────────────────
results = mem.recall("what product is the user building?", limit=5)
for r in results:
    print(f"[{r.memory.layer.value}] {r.memory.content}  rank={r.final_rank:.3f}")

# ── Feedback → memories get smarter over time ─────────────────
mem.report_outcome([r.memory.id for r in results[:2]], success=True)

# ── Knowledge graph ───────────────────────────────────────────
mem.kg_add_entity("User", EntityType.PERSON)
mem.kg_add_entity("Friday", EntityType.PROJECT)
mem.kg_add_relationship("User", "Friday", "building")
mem.kg_add_attribute("User", "timezone", "Asia/Dubai")

print(mem.kg_query("User"))

# ── Attention scoring ─────────────────────────────────────────
result = mem.score_attention("URGENT: the API is down!", channel="dm")
print(result.level)   # → "full"
print(result.score)   # → 85

# ── Consolidation (nightly / on-demand) ───────────────────────
from lore_ai.consolidation import LLMConsolidator
consolidator = LLMConsolidator(mem._config, mem._embedder)
r = consolidator.run_pass(mem.get_log(), mem.get_local_store(), mem.get_local_store())
print(f"{r.memories_created} facts extracted from logs")
```

---

## Storage backends

All backends share the same API. Swap with one env var.

### SQLite — default, zero infrastructure

```bash
LORE_STORE=sqlite
LORE_FRIDAY_HOME=~/.lore   # DB at ~/.lore/local.db
```

### Postgres + pgvector — production scale, ranking in SQL

```bash
pip install "lore-ai[postgres]"
LORE_STORE=postgres
LORE_POSTGRES_URL=postgresql://user:pass@host/lore
```

Requires `CREATE EXTENSION vector;` in your database. Schema migrates automatically on first start.

### Chroma — local vector DB, great for teams

```bash
pip install "lore-ai[chroma]"
LORE_STORE=chroma
LORE_CHROMA_PATH=~/.lore/chroma
```

### Pinecone — serverless hosted vectors

```bash
pip install "lore-ai[pinecone]"
LORE_STORE=pinecone
LORE_PINECONE_API_KEY=pk_...
LORE_PINECONE_INDEX=lore-ai
```

Create the index first (dimension must match your embedder):
```python
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key="pk_...")
pc.create_index("lore-ai", dimension=384, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"))
```

### OpenAI embeddings — no model download

```bash
pip install "lore-ai[openai]"
LORE_EMBEDDER=text-embedding-3-small
OPENAI_API_KEY=sk-...
LORE_EMBEDDING_DIM=1536
```

Works with any storage backend. Removes the 90 MB local model download.

---

## Migrating backends

Move all memories between backends in one command. lore-ai re-embeds automatically if the source and destination use different embedding models.

```bash
pip install "lore-ai[chroma,pinecone]"

# Escape Pinecone lock-in → local SQLite
lore-migrate --from pinecone --to sqlite \
  --source-pinecone-api-key pk_... \
  --source-pinecone-index my-index

# Local SQLite → Postgres (upgrade to production)
lore-migrate --from sqlite --to postgres \
  --dest-postgres-url postgresql://...

# Switch to OpenAI embeddings while migrating
lore-migrate --from sqlite --to chroma \
  --dest-embedder text-embedding-3-small

# Dry run — count what would be migrated
lore-migrate --from sqlite --to chroma --dry-run
```

---

## Hosted API

Run lore-ai as a service — your users call it with an API key, all compute happens server-side.

### Start the server

```bash
pip install "lore-ai[server]"

# Create an API key
lore-server create-key --namespace alice --label "alice dev"
# → lore_sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  (shown once)

# Start
lore-server serve --host 0.0.0.0 --port 8000

# Or with Docker (includes Postgres + pgvector)
docker compose up
```

### Connect from Python

```python
from lore_ai import HostedClient

mem = HostedClient(api_key="lore_sk_...")

# Exact same API as FridayMemory — nothing else changes
mem.remember("User is building a WhatsApp AI", conversation_id="c1")
results = mem.recall("WhatsApp")
mem.report_outcome([r.memory.id for r in results], success=True)
mem.kg_add_entity("Alice", EntityType.PERSON)
```

### API endpoints

```
POST /v1/memories/remember     append to log + episodic store
POST /v1/memories/recall       semantic search, layered retrieval
POST /v1/memories/report       RL signal (+1/−1)
POST /v1/memories/store        direct write to any layer
POST /v1/memories/consolidate  LLM consolidation pass
GET  /v1/memories/observe      priority-tagged log compression
POST /v1/kg/write              add entity / relationship / attribute
POST /v1/kg/query              query + BFS graph traverse
POST /v1/attention/score       0–100 message priority score
GET  /v1/health
```

All requests require `Authorization: Bearer lore_sk_...`. Namespace is derived from the key.

### Key management

```bash
lore-server create-key --namespace prod_user_123 --label "production"
lore-server list-keys
lore-server list-keys --namespace prod_user_123
lore-server revoke-key --key-hash abc123...
```

### Deploy to production

**Railway / Render** (fastest — 10 minutes):
1. Point at the `Dockerfile`
2. Set `LORE_STORE=postgres` and `LORE_POSTGRES_URL`
3. Deploy

**Fly.io:**
```bash
fly launch
fly secrets set LORE_STORE=postgres LORE_POSTGRES_URL=postgresql://...
fly deploy
```

**Self-hosted Docker:**
```bash
docker build -t lore-ai-server .
docker run -p 8000:8000 \
  -e LORE_STORE=postgres \
  -e LORE_POSTGRES_URL=postgresql://... \
  -v lore_data:/data \
  lore-ai-server
```

---

## MCP setup

### Claude Desktop

```bash
pip install "lore-ai[mcp]"
```

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "lore-ai": {
      "command": "lore-mcp",
      "env": {
        "LORE_FRIDAY_HOME": "~/.lore",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

Restart Claude Desktop. Nine tools appear automatically.

### Claude Code

```bash
claude mcp add lore-ai lore-mcp \
  --env LORE_FRIDAY_HOME=~/.lore \
  --env ANTHROPIC_API_KEY=sk-ant-...
```

### SSE / HTTP mode

```bash
lore-mcp --transport sse --port 8765
```

### MCP tools

| Tool | What it does | LLM cost |
|------|-------------|---------|
| `memory_remember` | Append to log + episodic store | None |
| `memory_recall` | Semantic search, identity+procedural always included | None |
| `memory_report_outcome` | +1/−1 RL signal on recalled memories | None |
| `memory_remember_now` | Direct write to any layer (bypass log) | None |
| `memory_consolidate` | Distil logs into semantic/procedural memories | Haiku |
| `memory_kg_write` | Add entity / relationship / attribute | None |
| `memory_kg_query` | Query entity + BFS graph traverse | None |
| `memory_observe` | Compress log into 🔴🟡🟢 observations | None |
| `memory_score_attention` | Score a message 0–100 | None |

---

## Multi-user / namespace isolation

Two isolation models:

**Instance-level** — each user gets their own process and `LORE_FRIDAY_HOME`. What Claude Desktop does naturally.

**Namespace-level** — one deployment, many users. All memories, logs, and graph data scoped per namespace. Zero leakage.

```bash
LORE_NAMESPACE=alice lore-mcp   # Alice's memory
LORE_NAMESPACE=bob   lore-mcp   # Bob's — completely separate, same DB
```

```python
mem_alice = FridayMemory(config=Config(namespace="alice"))
mem_bob   = FridayMemory(config=Config(namespace="bob"))
# same DB file, zero crossover
```

---

## Configuration

All settings via `LORE_` environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `LORE_STORE` | `sqlite` | Backend: `sqlite` · `postgres` · `chroma` · `pinecone` |
| `LORE_NAMESPACE` | `default` | User/agent isolation scope |
| `LORE_FRIDAY_HOME` | `~/.lore` | Base dir for logs and SQLite DB |
| `LORE_POSTGRES_URL` | *(empty)* | Postgres DSN (required when store=postgres) |
| `LORE_CHROMA_PATH` | `~/.lore/chroma` | ChromaDB persistence directory |
| `LORE_PINECONE_API_KEY` | *(empty)* | Pinecone API key |
| `LORE_PINECONE_INDEX` | `lore-ai` | Pinecone index name |
| `LORE_EMBEDDER` | `all-MiniLM-L6-v2` | Model name — sentence-transformers or OpenAI |
| `LORE_EMBEDDING_DIM` | `384` | Vector dimension (must match model) |
| `LORE_OPENAI_API_KEY` | *(empty)* | OpenAI key (required for OpenAI embedders) |
| `LORE_CONSOLIDATION_MODEL` | `claude-haiku-4-5-20251001` | LLM for consolidation |
| `LORE_RL_ALPHA` | `0.5` | Utility score weight in retrieval ranking |
| `LORE_RECENCY_HALF_LIFE_DAYS` | `90` | Recency decay half-life |
| `LORE_ATTENTION_FULL_THRESHOLD` | `75` | Score ≥ this → full attention |
| `LORE_ATTENTION_STANDARD_THRESHOLD` | `50` | Score ≥ this → standard |
| `LORE_ATTENTION_MINIMAL_THRESHOLD` | `25` | Score ≥ this → minimal |

---

## How it compares

| | lore-ai | Mem0 | LangChain | Zep | Raw Pinecone |
|--|---------|------|-----------|-----|-------------|
| Self-hostable | ✅ | ❌ cloud only | ✅ | ✅ | ✅ |
| Backend-agnostic | ✅ 4 backends | ❌ | ⚠️ manual | ❌ | — |
| RL-scored retrieval | ✅ | ❌ | ❌ | ❌ | ❌ |
| Asymmetric feedback (1.5×) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Knowledge graph | ✅ | ❌ | ❌ | ✅ | ❌ |
| 5-layer memory | ✅ | ⚠️ basic | ⚠️ basic | ⚠️ basic | ❌ |
| Log-first durability | ✅ | ❌ | ❌ | ❌ | ❌ |
| Migration CLI | ✅ | ❌ | ❌ | ❌ | — |
| Attention scoring | ✅ | ❌ | ❌ | ❌ | ❌ |
| MCP server (Claude) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Hosted API | ✅ self-host | ✅ | ❌ | ✅ | — |
| Open source | ✅ MIT | ⚠️ partial | ✅ | ✅ | — |

---

## Project structure

```
lore-ai/
├── src/lore_ai/
│   ├── api.py              ← FridayMemory — the local API
│   ├── client.py           ← HostedClient — the cloud API (same interface)
│   ├── config.py           ← Config (LORE_ env vars)
│   ├── types.py            ← Memory, Entity, Observation, AttentionResult, ...
│   ├── interfaces.py       ← LogStore, MemoryStore, Embedder protocols
│   ├── migrate.py          ← Migrator + lore-migrate CLI
│   ├── storage/
│   │   ├── sqlite.py       ← SQLiteMemoryStore
│   │   ├── postgres.py     ← PostgresMemoryStore (pgvector, ranking in SQL)
│   │   ├── chroma.py       ← ChromaMemoryStore
│   │   ├── pinecone_store.py ← PineconeMemoryStore
│   │   ├── kg.py           ← SQLiteKGStore
│   │   ├── log.py          ← FileLogStore (JSONL, fsync, checkpoints)
│   │   └── score_index.py  ← SQLiteScoreIndex (RL scores for external backends)
│   ├── embeddings/
│   │   ├── sentence_transformers.py
│   │   └── openai.py
│   ├── consolidation/
│   │   ├── consolidator.py ← LLMConsolidator (log → Claude Haiku → memories)
│   │   └── prompts.py
│   ├── observer/
│   │   └── observer.py     ← HeuristicObserver (🔴🟡🟢)
│   ├── scorer/
│   │   └── attention.py    ← AttentionScorer (0–100)
│   ├── mcp/
│   │   └── server.py       ← FastMCP server (9 tools)
│   └── server/
│       ├── app.py          ← FastAPI hosted API
│       ├── auth.py         ← API key management
│       ├── deps.py         ← FastAPI dependencies
│       └── routes/         ← memories, kg, health
├── Dockerfile
├── docker-compose.yml
└── tests/                  ← 50 test files, no LLM calls
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The quickest contribution is a new storage backend — implement the `MemoryStore` protocol in `storage/` and add tests. We'll merge it.

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## License

[MIT](LICENSE) · Built by [Ashwani Jha](https://github.com/ashwanijha04)

---

<div align="center">
  <sub>If lore-ai saves you from building another RAG pipeline, a ⭐ goes a long way.</sub>
</div>
