<div align="center">

# 🧠 extremis

**Memory that gets smarter the more your agent uses it**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/extremis?logo=pypi&logoColor=white&color=orange)](https://pypi.org/project/extremis)
[![CI](https://img.shields.io/github/actions/workflow/status/ashwanijha04/extremis/ci.yml?label=CI&logo=github)](https://github.com/ashwanijha04/extremis/actions)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple?logo=anthropic&logoColor=white)](https://modelcontextprotocol.io)
[![Cloud](https://img.shields.io/badge/hosted%20cloud-waitlist-blue)](https://github.com/ashwanijha04/extremis/issues/1)

<br/>

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/ashwanijha04/extremis)

<sub>One click · auto-provisions Postgres · memory persists across restarts</sub>

</div>

---

## The problem

Every team building an AI agent hits the same wall.

Your agent forgets everything the moment a conversation ends. So you add memory. You set up a vector database, write chunking logic, figure out retrieval ranking, handle stale entries, add multi-user isolation. Three weeks later you've built a half-working RAG pipeline and still haven't shipped the actual feature.

And even when you ship it — **it doesn't learn**. Every memory is treated identically. The fact your agent recalled a hundred times and the user loved sits next to one it got wrong once. Nothing improves. There's no feedback loop. You're running the same dumb cosine search forever.

The other problem is lock-in. Your vectors are in Pinecone. Moving them means re-embedding everything, rewriting your retrieval logic, and hoping nothing breaks.

extremis solves all three.

---

## What makes extremis different

### 1. Memory that forgets intelligently

Every competitor focuses on storing memory. Nobody talks about forgetting.

Human memory doesn't keep everything forever — unimportant things fade, important things strengthen. Agents with infinite, flat memory become slow and noisy over time. Intelligent forgetting is the hard problem nobody is solving.

extremis does two things here: **recency decay** (old memories rank lower automatically) and **asymmetric RL weighting** (negative feedback hurts 1.5× more than positive feedback helps, because mistakes should leave a stronger mark). The result is a memory that naturally surfaces what matters and buries what doesn't.

```python
mem = Extremis(config=Config(
    recency_half_life_days=30,  # episodic memories halve in rank every 30 days
    rl_alpha=0.8,               # strong RL signal — useful things stick, useless things fade
))

# This memory will rank lower in every future search
mem.report_outcome([bad_memory_id], success=False, weight=1.0)
# → score decreases by 1.5 (not 1.0 — the asymmetry is intentional)
```

---

### 2. Memory that explains itself

Agents make decisions based on memory. But *why* did it recall that specific memory? Without explainability you're guessing, debugging is painful, and auditing is impossible.

Every `recall()` result includes a plain-English reason:

```python
results = mem.recall("what does the user prefer?")

for r in results:
    print(r.memory.content)
    print(r.reason)

# "User prefers concise answers, no filler words"
# → "similarity 0.91 · score +4.0 · used 8× · 3d old"

# "User prefers dark mode in all UIs"
# → "semantic (always included) · similarity 0.73 · score +1.0 · used 3× · 12d old"

# "User once mentioned preferring email over Slack"
# → "similarity 0.54 · score -1.5 · first recall · 45d old"
```

The reason tells you: how semantically relevant it was, how much feedback has validated it, how many times it's been used, and how old it is. Auditable. Debuggable.

---

### 3. Cross-agent shared memory

Right now memory is per-agent. But the next wave of AI is **agent teams** — a research agent, a writing agent, a review agent, all working together. They need a shared brain.

extremis's namespace model already supports this. Multiple agents can read from and write to the same memory pool:

```python
# All three agents share the same memory namespace
research = Extremis(config=Config(namespace="team_alpha"))
writer   = Extremis(config=Config(namespace="team_alpha"))
reviewer = Extremis(config=Config(namespace="team_alpha"))

# Research agent stores what it found
research.remember("GPT-4 outperforms Claude on math benchmarks by 12%")
research.remember("Source: Stanford HAI report, April 2026")

# Writing agent recalls it without any extra wiring
results = writer.recall("GPT-4 performance data")
# → [GPT-4 outperforms Claude on math benchmarks by 12%]
# → [Source: Stanford HAI report, April 2026]

# Knowledge graph is shared too
research.kg_add_entity("Stanford HAI", EntityType.ORG)
research.kg_add_relationship("Stanford HAI", "HAI Report", "published")
print(writer.kg_query("Stanford HAI"))  # same graph
```

---

### 4. No RAG pipeline to build

One `pip install`. Two lines of config. extremis handles embedding, storage, retrieval ranking, consolidation, and the knowledge graph. You call `remember()` and `recall()`.

```python
# Local — zero infra
from extremis import Extremis
mem = Extremis()

# Your existing vector store
mem = Extremis(config=Config(store="pinecone", pinecone_api_key="..."))

# Self-hosted server — no model download on the client
from extremis import HostedClient
mem = HostedClient(api_key="extremis_sk_...", base_url="http://your-server:8000")

# Same three lines work for all three
mem.remember("User is building a WhatsApp AI", conversation_id="c1")
results = mem.recall("what is the user building?")
mem.report_outcome([r.memory.id for r in results], success=True)
```

---

### 5. Backend portability — no lock-in

Your vectors in Pinecone. Your team moves to Chroma. Your product needs Postgres. One command, everything migrates — and re-embeds automatically if you're switching models:

```bash
extremis-migrate --from pinecone --to postgres \
  --source-pinecone-api-key pk_... \
  --dest-postgres-url postgresql://...

# Switching to OpenAI embeddings at the same time
extremis-migrate --from sqlite --to chroma \
  --dest-embedder text-embedding-3-small
```

---

### Coming soon

**Memory health dashboard** — freshness score, contradiction count, retrieval hit rate, coverage gaps. Memory observability nobody is building yet.

**Domain profiles** — pre-built memory configurations for common agent types:
```python
# Coming in v0.2
from extremis.profiles import SalesAgent, CodingAgent, SupportAgent

mem = Extremis(profile=SalesAgent())
# Knows to remember: customer names, deal stage, objections, preferences
# Knows to forget: small talk after 7 days, meeting logistics after 24h
# Attention: high for "budget", "decision maker", "timeline"
```

---

---

## How it works

### The intelligence layer

extremis sits **above** your vector store. RL scoring, the knowledge graph, consolidation, and attention scoring are all backend-independent — they work the same whether your vectors are in SQLite, Pinecone, or Chroma.

```
┌────────────────────────────────────────────────────────────────┐
│                     YOUR APP / AGENT                            │
│      remember() · recall() · report_outcome() · kg_*()         │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────┐
│                  EXTREMIS INTELLIGENCE LAYER                      │
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

Beyond vectors, extremis maintains a structured graph — answers structural questions that semantic search can't:

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

> **Requires Python 3.11+**
>
> If `pip install extremis` says **"no matching distribution found"** — your default `pip` points to Python 3.9 or older. This is common on macOS.
>
> **Check your version:** `python3 --version`
>
> | Platform | Fix |
> |----------|-----|
> | macOS | `brew install python@3.11` then use `pip3.11` |
> | Linux | `sudo apt install python3.11 python3.11-pip` |
> | Windows | [python.org/downloads](https://python.org/downloads) |

```bash
# Confirm you have Python 3.11+
python3.11 --version

# Core — SQLite + local sentence-transformers (no API key needed)
pip3.11 install extremis

# + MCP server (Claude Desktop / Code)
pip3.11 install "extremis[mcp]"

# + Postgres backend
pip3.11 install "extremis[postgres]"

# + Chroma backend
pip3.11 install "extremis[chroma]"

# + Pinecone backend
pip3.11 install "extremis[pinecone]"

# + OpenAI embeddings (swap out the 90 MB model download)
pip3.11 install "extremis[openai]"

# + Hosted API server
pip3.11 install "extremis[server]"

# + Python SDK for hosted cloud
pip3.11 install "extremis[client]"

# Everything
pip3.11 install "extremis[all]"
```

**Requires Python 3.11+**

> **First run note** — `sentence-transformers` downloads `all-MiniLM-L6-v2` (~90 MB) on first use. One-time, cached to `~/.cache/huggingface/`. To skip it, use OpenAI embeddings: `EXTREMIS_EMBEDDER=text-embedding-3-small`.

---

## Quick start

```python
from extremis import Extremis, MemoryLayer
from extremis.types import EntityType

mem = Extremis()  # ~/.extremis/ by default

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
from extremis.consolidation import LLMConsolidator
consolidator = LLMConsolidator(mem._config, mem._embedder)
r = consolidator.run_pass(mem.get_log(), mem.get_local_store(), mem.get_local_store())
print(f"{r.memories_created} facts extracted from logs")
```

---

## Storage backends

All backends share the same API. Swap with one env var.

### Don't want anything stored locally?

Three options — all work out of the box:

| Option | Local footprint | Cost |
|--------|----------------|------|
| **Postgres on Supabase / Neon** | None | Free tier available |
| **Pinecone** | RL score sidecar only (~KB) | Free tier available |
| **HostedClient** (your own server) | None at all | Your hosting cost |

**Quickest: free Postgres on Supabase**
```bash
# 1. Create project at supabase.com, grab the connection string
# 2. Enable pgvector: run "CREATE EXTENSION vector;" in the SQL editor
pip3.11 install "extremis[postgres]"
EXTREMIS_STORE=postgres EXTREMIS_POSTGRES_URL=postgresql://... python3.11 your_app.py
```

**Zero footprint: HostedClient**
```python
from extremis import HostedClient
# deploy extremis-server on Railway/Fly/Render, point at it
mem = HostedClient(api_key="extremis_sk_...", base_url="https://your-server.railway.app")
# nothing written locally — not even the embedding model
```

---

### SQLite — default, zero infrastructure

```bash
EXTREMIS_STORE=sqlite
EXTREMIS_FRIDAY_HOME=~/.extremis   # DB at ~/.extremis/local.db
```

### Postgres + pgvector — production scale, ranking in SQL

```bash
pip3.11 install "extremis[postgres]"
EXTREMIS_STORE=postgres
EXTREMIS_POSTGRES_URL=postgresql://user:pass@host/extremis
```

Requires `CREATE EXTENSION vector;` in your database. Schema migrates automatically on first start.

### Chroma — local vector DB, great for teams

```bash
pip3.11 install "extremis[chroma]"
EXTREMIS_STORE=chroma
EXTREMIS_CHROMA_PATH=~/.extremis/chroma
```

### Pinecone — serverless hosted vectors

```bash
pip3.11 install "extremis[pinecone]"
EXTREMIS_STORE=pinecone
EXTREMIS_PINECONE_API_KEY=pk_...
EXTREMIS_PINECONE_INDEX=extremis
```

Create the index first (dimension must match your embedder):
```python
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key="pk_...")
pc.create_index("extremis", dimension=384, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"))
```

### OpenAI embeddings — no model download

```bash
pip3.11 install "extremis[openai]"
EXTREMIS_EMBEDDER=text-embedding-3-small
OPENAI_API_KEY=sk-...
EXTREMIS_EMBEDDING_DIM=1536
```

Works with any storage backend. Removes the 90 MB local model download.

---

## Migrating backends

Move all memories between backends in one command. extremis re-embeds automatically if the source and destination use different embedding models.

```bash
pip3.11 install "extremis[chroma,pinecone]"

# Escape Pinecone lock-in → local SQLite
extremis-migrate --from pinecone --to sqlite \
  --source-pinecone-api-key pk_... \
  --source-pinecone-index my-index

# Local SQLite → Postgres (upgrade to production)
extremis-migrate --from sqlite --to postgres \
  --dest-postgres-url postgresql://...

# Switch to OpenAI embeddings while migrating
extremis-migrate --from sqlite --to chroma \
  --dest-embedder text-embedding-3-small

# Dry run — count what would be migrated
extremis-migrate --from sqlite --to chroma --dry-run
```

---

## Hosted API

Run extremis as a service — your users call it with an API key, all compute happens server-side. No model download on the client. No local database.

> **Status:** The server is fully built and self-hostable today. A managed cloud at `api.extremis.com` is in progress — [join the waitlist](https://github.com/ashwanijha04/extremis/issues/1).

### One-click deploy to Render (memory lives in Render Postgres)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/ashwanijha04/extremis)

Clicking this button deploys `extremis-server` **and** provisions a free Postgres database automatically via `render.yaml`. Memory lives in Render's managed Postgres — persistent across restarts and redeploys.

After deploy, generate an API key:
```bash
# in Render's shell tab for the extremis service
extremis-server create-key --namespace myapp --label prod
```

Connect from anywhere with zero local footprint:
```python
from extremis import HostedClient
mem = HostedClient(api_key="extremis_sk_...", base_url="https://your-app.onrender.com")
```

### Deploy to Railway (manual — 3 steps)

> **⚠️ Don't use SQLite on Railway.** Container filesystems are ephemeral — memories are lost on every restart. Always use Railway Postgres.

1. Create a new project on [railway.app](https://railway.app) → **Deploy from GitHub repo** → select `extremis`
2. Add a Postgres plugin: **+ New** → **Database** → **PostgreSQL**
3. Set these environment variables on the extremis service:
   ```
   EXTREMIS_STORE=postgres
   EXTREMIS_POSTGRES_URL=${{Postgres.DATABASE_URL}}
   ```

Railway injects the URL automatically. Memory now lives in Railway's managed Postgres.

### Self-host locally in 2 minutes

```bash
pip3.11 install "extremis[server]"

# Generate an API key
extremis-server create-key --namespace alice --label "prod"
# → extremis_sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  (shown once, store it)

# Start the server
extremis-server serve --host 0.0.0.0 --port 8000

# Or with Docker (bundles Postgres + pgvector)
docker compose up
```

### Connect from Python

```python
from extremis import HostedClient

# Point at your self-hosted server
mem = HostedClient(api_key="extremis_sk_...", base_url="http://your-server:8000")

# Exact same API as Memory — nothing else changes
mem.remember("User is building a WhatsApp AI", conversation_id="c1")
results = mem.recall("WhatsApp")
mem.report_outcome([r.memory.id for r in results], success=True)
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

All requests require `Authorization: Bearer extremis_sk_...`. Namespace is derived from the key.

### Key management

```bash
extremis-server create-key --namespace prod_user_123 --label "production"
extremis-server list-keys
extremis-server list-keys --namespace prod_user_123
extremis-server revoke-key --key-hash abc123...
```

### Deploy to production

**Railway / Render** (fastest — 10 minutes):
1. Point at the `Dockerfile`
2. Set `EXTREMIS_STORE=postgres` and `EXTREMIS_POSTGRES_URL`
3. Deploy

**Fly.io:**
```bash
fly launch
fly secrets set EXTREMIS_STORE=postgres EXTREMIS_POSTGRES_URL=postgresql://...
fly deploy
```

**Self-hosted Docker:**
```bash
docker build -t extremis-server .
docker run -p 8000:8000 \
  -e EXTREMIS_STORE=postgres \
  -e EXTREMIS_POSTGRES_URL=postgresql://... \
  -v lore_data:/data \
  extremis-server
```

---

## MCP setup

### Claude Desktop

```bash
pip3.11 install "extremis[mcp]"
```

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "extremis": {
      "command": "extremis-mcp",
      "env": {
        "EXTREMIS_FRIDAY_HOME": "~/.extremis",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

Restart Claude Desktop. Nine tools appear automatically.

### Claude Code

```bash
claude mcp add extremis extremis-mcp \
  --env EXTREMIS_FRIDAY_HOME=~/.extremis \
  --env ANTHROPIC_API_KEY=sk-ant-...
```

### SSE / HTTP mode

```bash
extremis-mcp --transport sse --port 8765
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

**Instance-level** — each user gets their own process and `EXTREMIS_FRIDAY_HOME`. What Claude Desktop does naturally.

**Namespace-level** — one deployment, many users. All memories, logs, and graph data scoped per namespace. Zero leakage.

```bash
EXTREMIS_NAMESPACE=alice extremis-mcp   # Alice's memory
EXTREMIS_NAMESPACE=bob   extremis-mcp   # Bob's — completely separate, same DB
```

```python
mem_alice = Extremis(config=Config(namespace="alice"))
mem_bob   = Extremis(config=Config(namespace="bob"))
# same DB file, zero crossover
```

---

## Configuration

All settings via `EXTREMIS_` environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `EXTREMIS_STORE` | `sqlite` | Backend: `sqlite` · `postgres` · `chroma` · `pinecone` |
| `EXTREMIS_NAMESPACE` | `default` | User/agent isolation scope |
| `EXTREMIS_FRIDAY_HOME` | `~/.extremis` | Base dir for logs and SQLite DB |
| `EXTREMIS_POSTGRES_URL` | *(empty)* | Postgres DSN (required when store=postgres) |
| `EXTREMIS_CHROMA_PATH` | `~/.extremis/chroma` | ChromaDB persistence directory |
| `EXTREMIS_PINECONE_API_KEY` | *(empty)* | Pinecone API key |
| `EXTREMIS_PINECONE_INDEX` | `extremis` | Pinecone index name |
| `EXTREMIS_EMBEDDER` | `all-MiniLM-L6-v2` | Model name — sentence-transformers or OpenAI |
| `EXTREMIS_EMBEDDING_DIM` | `384` | Vector dimension (must match model) |
| `EXTREMIS_OPENAI_API_KEY` | *(empty)* | OpenAI key (required for OpenAI embedders) |
| `EXTREMIS_CONSOLIDATION_MODEL` | `claude-haiku-4-5-20251001` | LLM for consolidation |
| `EXTREMIS_RL_ALPHA` | `0.5` | Utility score weight in retrieval ranking |
| `EXTREMIS_RECENCY_HALF_LIFE_DAYS` | `90` | Recency decay half-life |
| `EXTREMIS_ATTENTION_FULL_THRESHOLD` | `75` | Score ≥ this → full attention |
| `EXTREMIS_ATTENTION_STANDARD_THRESHOLD` | `50` | Score ≥ this → standard |
| `EXTREMIS_ATTENTION_MINIMAL_THRESHOLD` | `25` | Score ≥ this → minimal |

---

## How it compares

| | extremis | Mem0 | LangChain | Zep | Raw Pinecone |
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
extremis/
├── src/extremis/
│   ├── api.py              ← Memory — the local API
│   ├── client.py           ← HostedClient — the cloud API (same interface)
│   ├── config.py           ← Config (EXTREMIS_ env vars)
│   ├── types.py            ← Memory, Entity, Observation, AttentionResult, ...
│   ├── interfaces.py       ← LogStore, MemoryStore, Embedder protocols
│   ├── migrate.py          ← Migrator + extremis-migrate CLI
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
  <sub>If extremis saves you from building another RAG pipeline, a ⭐ goes a long way.</sub>
</div>
