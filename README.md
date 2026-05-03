# friday-memory

Layered memory for AI agents. Log-first, RL-scored, consolidation-backed.

Plug in two lines of config and your agent gets persistent memory across conversations — no vector database to run, no RAG pipeline to maintain.

## How it works

```
every conversation
    remember("user said X")   →  appended to JSONL log + episodic store
    recall("topic")           →  semantic search, identity + procedural always included
    report_outcome(ids, +1)   →  adjusts utility scores on recalled memories

periodically (daily / every 20 convos)
    consolidate()             →  Claude Haiku reads new log entries,
                                 extracts semantic + procedural memories,
                                 advances checkpoint
```

### Memory layers

| Layer | What it holds | Written by |
|-------|--------------|-----------|
| `identity` | Who the user is (never auto-committed) | Human review only |
| `procedural` | Behavioral rules ("always ask about deadline first") | Consolidator |
| `semantic` | Durable facts ("user is a Python developer, works solo") | Consolidator |
| `episodic` | Timestamped events from conversations | `remember()` |

## Install

```bash
pip install friday-memory          # core: SQLite + sentence-transformers
pip install "friday-memory[mcp]"   # + MCP server for Claude Desktop/Code
pip install "friday-memory[all]"   # + Postgres consolidated store
```

Requires Python 3.11+.

## Quickstart (Python)

```python
from friday_memory import FridayMemory, MemoryLayer

mem = FridayMemory()  # stores in ~/.friday/ by default

# Remember things
mem.remember("User said they prefer concise answers", conversation_id="conv_001")
mem.remember("User is building a WhatsApp AI product", conversation_id="conv_001")

# Direct write for time-sensitive or high-confidence facts
mem.remember_now("User's flight departs Thursday 06:00", layer=MemoryLayer.EPISODIC)

# Recall — identity + procedural always returned; semantic + episodic by relevance
results = mem.recall("what kind of product is the user building?")
for r in results:
    print(f"[{r.memory.layer.value}] {r.memory.content}  (relevance={r.relevance:.2f})")

# Reinforcement signal
mem.report_outcome([results[0].memory.id], success=True)
```

## MCP server (Claude Desktop / Claude Code)

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

Restart Claude Desktop. Five tools appear automatically:

| Tool | When to call it |
|------|----------------|
| `memory_remember` | After every message worth keeping |
| `memory_recall` | At conversation start with user's first message |
| `memory_report_outcome` | When user rates a response |
| `memory_remember_now` | For time-sensitive / high-confidence direct writes |
| `memory_consolidate` | Every ~20 conversations or once a day |

## Configuration

All config via environment variables with `FRIDAY_` prefix, or a `.env` file:

```env
FRIDAY_HOME=~/.friday
FRIDAY_EMBEDDER=sentence-transformers/all-MiniLM-L6-v2
FRIDAY_CONSOLIDATION_MODEL=claude-haiku-4-5-20251001
FRIDAY_RL_ALPHA=0.5
FRIDAY_RECENCY_HALF_LIFE_DAYS=90
```

## Architecture

```
FridayMemory (api.py)
├── FileLogStore      — daily JSONL, fsync on every write, checkpoint-based reads
├── SQLiteMemoryStore — cosine search in numpy, score × recency ranking
├── SentenceTransformerEmbedder — all-MiniLM-L6-v2, lazy-loaded
└── LLMConsolidator   — Claude Haiku extracts memories from log batches
```

Storage lives in `~/.friday/` by default:
```
~/.friday/
├── log/
│   ├── 2026-05-04.jsonl
│   └── .checkpoint
└── local.db
```

## License

MIT
