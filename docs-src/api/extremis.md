# Extremis

The main API class. Manages memory storage, retrieval, feedback, and the knowledge graph.

## Constructor

```python
from extremis import Extremis, Config

mem = Extremis(
    config=None,    # Config instance — defaults to Config()
    log=None,       # LogStore override
    local=None,     # MemoryStore override
    embedder=None,  # Embedder override
)
```

All parameters are optional. `Extremis()` with no arguments works out of the box.

---

## remember()

Append to the conversation log and write an episodic memory.

```python
mem.remember(
    content: str,
    role: str = "user",           # "user" | "assistant" | "system"
    conversation_id: str = "default",
    metadata: dict = None,
)
```

**What it does:**
1. Appends to `~/.extremis/log/{namespace}/YYYY-MM-DD.jsonl` (fsync'd)
2. Embeds the content
3. Stores as an `episodic` memory

---

## recall()

Semantic search over all memories.

```python
results: list[RecallResult] = mem.recall(
    query: str,
    limit: int = 10,
    layers: list[MemoryLayer] = None,   # None = all layers
    min_score: float = None,            # None = use Config.recall_min_relevance
)
```

**Retrieval order:**
1. Identity + procedural always fetched first (pinned, min_score bypassed)
2. Semantic + episodic (or `layers` filter) ranked by `cosine × RL × recency`
3. Results merged, deduped, capped at `limit`

**Returns:** `list[RecallResult]` — each with `.memory`, `.relevance`, `.final_rank`, `.reason`

---

## report_outcome()

Reinforcement signal — adjust utility scores on recalled memories.

```python
mem.report_outcome(
    memory_ids: list[UUID],
    success: bool,
    weight: float = 1.0,
)
```

- `success=True` → `score += weight`
- `success=False` → `score -= weight * 1.5` (asymmetric)

---

## remember_now()

Write directly to a structured layer, bypassing the log.

```python
memory: Memory = mem.remember_now(
    content: str,
    layer: MemoryLayer,
    expires_at: datetime = None,
    confidence: float = 0.9,
    metadata: dict = None,
)
```

**Write-time dedup:** For `semantic` and `procedural` layers, if an existing memory has cosine similarity ≥ `Config.dedup_similarity_threshold` (default 0.92), the old memory is superseded rather than creating a duplicate.

---

## compact()

Reconcile contradictions in existing structured memories via LLM.

```python
result: CompactionResult = mem.compact(
    layer: MemoryLayer = MemoryLayer.SEMANTIC,
)
```

See [Consolidation](../concepts/consolidation.md) for details.

---

## Knowledge graph methods

```python
# Add / update entities
mem.kg_add_entity(name: str, type: EntityType, metadata: dict = None)

# Add / update relationships
mem.kg_add_relationship(from_entity, to_entity, rel_type, weight=1.0, metadata=None)

# Add / update attributes
mem.kg_add_attribute(entity: str, key: str, value: str)

# Query
result: EntityResult = mem.kg_query(name: str)
graph: list[EntityResult] = mem.kg_traverse(name: str, depth: int = 2)
```

---

## score_attention()

```python
result: AttentionResult = mem.score_attention(
    message: str,
    sender: str = "",
    channel: str = "dm",
    owner_ids: set[str] = None,
    allowlist: set[str] = None,
    context: dict = None,
)
```

Returns `AttentionResult` with `.score` (0–100), `.level` (full/standard/minimal/ignore), `.reason`.

---

## observe()

Compress conversation log entries into priority-tagged observations.

```python
observations: list[Observation] = mem.observe(conversation_id: str = "default")
```

---

## Internal accessors

```python
mem.get_local_store()   # MemoryStore
mem.get_log()           # LogStore
mem.get_kg()            # SQLiteKGStore
mem._embedder           # Embedder (for advanced use)
mem._config             # Config
```
