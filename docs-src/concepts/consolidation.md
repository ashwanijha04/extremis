# Consolidation

Consolidation is the "dream pass" — it reads raw conversation logs and distils them into structured semantic and procedural memories using an LLM.

## What it does

Without consolidation, `remember()` only creates episodic memories — timestamped events with no structure. Consolidation reads those logs and extracts:

- **Semantic facts**: "User is a Python developer" (from "I've been writing Python for 10 years")
- **Procedural rules**: "Suggest FastAPI over Flask for Python APIs" (from a pattern of user preferences)

## When to run it

- After every 20–50 conversations (cron job or background task)
- Once a day, nightly
- Manually when you want to refresh structured memory

Consolidation is **idempotent** — safe to re-run. It uses a checkpoint to track which log entries have been processed and skips already-consolidated entries.

## Running consolidation

```python
import os
from extremis import Extremis
from extremis.consolidation import LLMConsolidator

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

mem = Extremis()
consolidator = LLMConsolidator(mem._config, mem._embedder)
result = consolidator.run_pass(
    mem.get_log(),
    mem.get_local_store(),
    mem.get_local_store(),
)

print(f"Created: {result.memories_created}")
print(f"Duration: {result.duration_seconds:.1f}s")
```

## Via MCP tool

In Claude Desktop, call `memory_consolidate`:

```
Run memory_consolidate to distil the conversation logs.
```

## How it works

1. Reads all log entries since the last checkpoint
2. Groups entries by `conversation_id`
3. For each conversation with ≥ 2 messages, calls Claude Haiku
4. Extracts semantic facts and procedural rules as JSON
5. Embeds and stores them in the memory store
6. Advances the checkpoint

The extraction prompt instructs Claude to:
- Extract only facts that generalise beyond this specific conversation
- Skip transient task details and moods
- Write one fact per memory, no padding
- Rate confidence 0.0–1.0

## Compaction — resolving contradictions

Consolidation distils new logs. **Compaction** resolves conflicts in *existing* structured memories.

If a user told you they prefer concise answers last month but verbose answers this week, you'll have contradictory semantic memories. `compact()` sends all semantic memories to Claude and asks it to reconcile them.

```python
result = mem.compact(layer=MemoryLayer.SEMANTIC)
print(f"Reconciled: {result.memories_reconciled}")
print(f"Deduped: {result.memories_deduped}")
```

Via MCP: `memory_compact(layer="semantic")`

!!! warning "Consolidation vs compaction"
    - `memory_consolidate` — processes **new log entries**, writes to structured memory
    - `memory_compact` — reconciles **existing structured memories**, resolves contradictions

## Cost

Consolidation uses Claude Haiku (cheapest Anthropic model). A typical session of 20 messages costs ~$0.001.

Configure the model:
```bash
EXTREMIS_CONSOLIDATION_MODEL=claude-haiku-4-5-20251001  # default
EXTREMIS_CONSOLIDATION_HARD_MODEL=claude-sonnet-4-6     # for complex reconciliation
```
