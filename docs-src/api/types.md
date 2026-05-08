# Types

## MemoryLayer

```python
from extremis import MemoryLayer

MemoryLayer.EPISODIC    # "episodic"
MemoryLayer.SEMANTIC    # "semantic"
MemoryLayer.PROCEDURAL  # "procedural"
MemoryLayer.IDENTITY    # "identity"
MemoryLayer.WORKING     # "working"
```

## Memory

A single stored memory.

```python
class Memory:
    id: UUID
    namespace: str
    layer: MemoryLayer
    content: str
    embedding: list[float] | None   # not returned on recall
    score: float                    # RL utility score (starts at 0)
    confidence: float               # 0.0–1.0
    metadata: dict
    source_memory_ids: list[UUID]   # for promoted/reconciled memories
    validity_start: datetime
    validity_end: datetime | None   # None = currently valid
    created_at: datetime
    last_accessed_at: datetime | None
    access_count: int
    do_not_consolidate: bool
```

## RecallResult

Returned by `mem.recall()`.

```python
class RecallResult:
    memory: Memory
    relevance: float    # raw cosine similarity
    final_rank: float   # relevance × utility × recency
    reason: str         # "similarity 0.91 · score +4.0 · used 8× · 3d old"
```

## EntityType

```python
from extremis.types import EntityType

EntityType.PERSON   # "person"
EntityType.ORG      # "org"
EntityType.PROJECT  # "project"
EntityType.GROUP    # "group"
EntityType.CONCEPT  # "concept"
EntityType.OTHER    # "other"
```

## Entity, Relationship, KGAttribute

```python
class Entity:
    id: int
    namespace: str
    name: str
    type: EntityType
    metadata: dict

class Relationship:
    id: int
    namespace: str
    from_entity: str
    to_entity: str
    rel_type: str
    weight: float
    metadata: dict

class KGAttribute:
    id: int
    namespace: str
    entity: str
    key: str
    value: str
```

## EntityResult

Returned by `mem.kg_query()`.

```python
class EntityResult:
    entity: Entity
    relationships: list[Relationship]
    attributes: list[KGAttribute]
```

## AttentionResult

Returned by `mem.score_attention()`.

```python
class AttentionResult:
    score: int      # 0–100
    level: str      # "full" | "standard" | "minimal" | "ignore"
    reason: str
    breakdown: dict
```

## ObservationPriority

```python
from extremis.types import ObservationPriority

ObservationPriority.CRITICAL  # 🔴 decisions, errors, deadlines
ObservationPriority.CONTEXT   # 🟡 reasons, insights
ObservationPriority.INFO      # 🟢 everything else
```

## ConsolidationResult

```python
class ConsolidationResult:
    memories_created: int
    memories_superseded: int
    log_checkpoint: str
    duration_seconds: float
```

## CompactionResult

```python
class CompactionResult:
    memories_reconciled: int
    memories_deduped: int
    memories_unchanged: int
    duration_seconds: float
```
