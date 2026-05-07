from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# ------------------------------------------------------------------ #
# Core memory types
# ------------------------------------------------------------------ #


class MemoryLayer(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    IDENTITY = "identity"
    WORKING = "working"


class Memory(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    namespace: str = "default"
    layer: MemoryLayer
    content: str
    embedding: Optional[list[float]] = None
    score: float = 0.0
    confidence: float = 0.5
    metadata: dict = Field(default_factory=dict)
    source_memory_ids: list[UUID] = Field(default_factory=list)
    validity_start: datetime
    validity_end: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0
    do_not_consolidate: bool = False


class LogEntry(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    namespace: str = "default"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    role: str  # user | assistant | system
    content: str
    conversation_id: str
    metadata: dict = Field(default_factory=dict)


class RecallResult(BaseModel):
    memory: Memory
    relevance: float  # raw cosine similarity
    final_rank: float  # relevance × utility × recency
    reason: str = ""  # human-readable explanation of why this memory ranked here


class ConsolidationResult(BaseModel):
    memories_created: int = 0
    memories_superseded: int = 0
    log_checkpoint: str = ""
    duration_seconds: float = 0.0
    notes: str = ""


class CompactionResult(BaseModel):
    """Result of a compact() pass over existing structured memories."""

    memories_reconciled: int = 0  # contradictions resolved by LLM
    memories_deduped: int = 0  # near-duplicates merged at write time
    memories_unchanged: int = 0
    duration_seconds: float = 0.0


class FeedbackSignal(BaseModel):
    memory_ids: list[UUID]
    success: bool
    weight: float = 1.0
    context: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ------------------------------------------------------------------ #
# Knowledge Graph types
# ------------------------------------------------------------------ #


class EntityType(str, Enum):
    PERSON = "person"
    ORG = "org"
    PROJECT = "project"
    GROUP = "group"
    CONCEPT = "concept"
    OTHER = "other"


class Entity(BaseModel):
    id: int = 0
    namespace: str = "default"
    name: str
    type: EntityType
    metadata: dict = Field(default_factory=dict)


class Relationship(BaseModel):
    id: int = 0
    namespace: str = "default"
    from_entity: str
    to_entity: str
    rel_type: str
    weight: float = 1.0
    metadata: dict = Field(default_factory=dict)


class KGAttribute(BaseModel):
    id: int = 0
    namespace: str = "default"
    entity: str
    key: str
    value: str


class EntityResult(BaseModel):
    entity: Entity
    relationships: list[Relationship] = Field(default_factory=list)
    attributes: list[KGAttribute] = Field(default_factory=list)


# ------------------------------------------------------------------ #
# Observer / attention types
# ------------------------------------------------------------------ #


class ObservationPriority(str, Enum):
    CRITICAL = "critical"  # 🔴 decisions, errors, deadlines
    CONTEXT = "context"  # 🟡 reasons, insights, learnings
    INFO = "info"  # 🟢 everything else


class Observation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    namespace: str = "default"
    content: str
    priority: ObservationPriority
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    conversation_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class AttentionResult(BaseModel):
    score: int  # 0–100
    level: str  # full | standard | minimal | ignore
    reason: str
    breakdown: dict = Field(default_factory=dict)
