# Quickstart

Get memory working in 5 minutes.

## Option 1 — Wrap your existing LLM client (easiest)

If you're already using Claude or OpenAI, change one import:

=== "Claude"

    ```bash
    pip3.11 install "extremis[wrap-anthropic]"
    ```

    ```python
    from extremis.wrap import Anthropic  # (1)
    from extremis import Extremis

    client = Anthropic(
        api_key="sk-ant-...",
        memory=Extremis(),          # (2)
        session_id="user_123",      # (3)
    )

    # Turn 1 — no memory yet
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": "My name is Ashwani and I prefer Python."}]
    )

    # Turn 2 — memory kicks in automatically
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": "What language do I prefer?"}]
    )
    # Claude knows the answer — extremis injected context before the call
    ```

    1. Drop-in for `import anthropic; client = anthropic.Anthropic(...)`. All other methods pass through unchanged.
    2. Local SQLite by default — no infra needed. Swap to Postgres or Pinecone later.
    3. Groups messages for consolidation. Use a stable user ID in production.

=== "OpenAI"

    ```bash
    pip3.11 install "extremis[wrap-openai]"
    ```

    ```python
    from extremis.wrap import OpenAI
    from extremis import Extremis

    client = OpenAI(api_key="sk-...", memory=Extremis())

    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "My name is Ashwani."}]
    )

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What's my name?"}]
    )
    print(r.choices[0].message.content)  # → "Your name is Ashwani."
    ```

---

## Option 2 — Manual API

Full control over what gets stored and recalled.

```bash
pip3.11 install extremis
```

```python
from extremis import Extremis, MemoryLayer

mem = Extremis()

# ── 1. Store memories ─────────────────────────────────
mem.remember(
    "User is building a WhatsApp AI assistant",
    conversation_id="conv_001",
)
mem.remember(
    "User confirmed they prefer concise answers",
    conversation_id="conv_001",
)

# Write directly to a layer (skip the log)
mem.remember_now(
    "Always ask about deadlines before suggesting solutions",
    layer=MemoryLayer.PROCEDURAL,
)

# ── 2. Recall ─────────────────────────────────────────
results = mem.recall("what product is the user building?", limit=5)
for r in results:
    print(f"[{r.memory.layer.value}] {r.memory.content}")
    print(f"  reason: {r.reason}")
    # reason: "similarity 0.91 · score +2.0 · used 3× · today"

# ── 3. Feedback ───────────────────────────────────────
# Mark memories that were helpful — they'll rank higher next time
mem.report_outcome(
    [r.memory.id for r in results[:2]],
    success=True,
    weight=1.0,
)

# ── 4. Consolidation (optional, nightly) ─────────────
# Distil conversation logs into structured semantic facts
from extremis.consolidation import LLMConsolidator
import os

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
consolidator = LLMConsolidator(mem._config, mem._embedder)
result = consolidator.run_pass(
    mem.get_log(), mem.get_local_store(), mem.get_local_store()
)
print(f"{result.memories_created} facts extracted from logs")
```

---

## Option 3 — Claude Desktop (MCP)

No Python code at all. Add to your Claude Desktop config and get 10 memory tools automatically.

```bash
pip3.11 install "extremis[mcp]"
```

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "extremis": {
      "command": "/opt/homebrew/bin/extremis-mcp",
      "env": {
        "EXTREMIS_HOME": "~/.extremis"
      }
    }
  }
}
```

Restart Claude Desktop. See [MCP setup](../mcp/claude-desktop.md) for full details.

---

## What's next

- [**Memory layers**](../concepts/memory-layers.md) — understand episodic, semantic, procedural, identity
- [**RL scoring**](../concepts/rl-scoring.md) — how feedback improves retrieval over time
- [**Backends**](../backends/index.md) — Postgres, Chroma, Pinecone for production
- [**Namespaces**](../concepts/namespaces.md) — multi-user isolation
