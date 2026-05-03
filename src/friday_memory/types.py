from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MemoryLayer(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    IDENTITY = "identity"


class Memory(BaseModel):
    id: UUID = Field(default_factory=uuid4)
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
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    role: str  # user | assistant | system
    content: str
    conversation_id: str
    metadata: dict = Field(default_factory=dict)


class RecallResult(BaseModel):
    memory: Memory
    relevance: float   # raw cosine similarity
    final_rank: float  # relevance × utility × recency


class ConsolidationResult(BaseModel):
    memories_created: int = 0
    memories_superseded: int = 0
    log_checkpoint: str = ""
    duration_seconds: float = 0.0
    notes: str = ""


class FeedbackSignal(BaseModel):
    memory_ids: list[UUID]
    success: bool
    weight: float = 1.0
    context: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
