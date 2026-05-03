from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable
from uuid import UUID

from .types import (
    ConsolidationResult,
    FeedbackSignal,
    LogEntry,
    Memory,
    MemoryLayer,
    RecallResult,
)


@runtime_checkable
class LogStore(Protocol):
    """Append-only conversation log. Source of truth for the consolidator."""

    def append(self, entry: LogEntry) -> None: ...

    def read_since(self, checkpoint: Optional[str]) -> list[LogEntry]: ...

    def get_checkpoint(self) -> Optional[str]: ...

    def set_checkpoint(self, checkpoint: str) -> None: ...


@runtime_checkable
class MemoryStore(Protocol):
    """Structured memory. Same interface for SQLite (local) and Postgres (consolidated)."""

    def store(self, memory: Memory) -> Memory: ...

    def get(self, memory_id: UUID) -> Optional[Memory]: ...

    def search(
        self,
        query_embedding: list[float],
        layers: Optional[list[MemoryLayer]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[RecallResult]: ...

    def update_score(self, memory_id: UUID, delta: float) -> None: ...

    def supersede(self, old_id: UUID, new_memory: Memory) -> None: ...

    def list_recent(
        self,
        layer: Optional[MemoryLayer] = None,
        limit: int = 50,
    ) -> list[Memory]: ...


@runtime_checkable
class Embedder(Protocol):
    """Text → float vector. Local model or remote API."""

    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Consolidator(Protocol):
    """Reads new log entries past the last checkpoint and writes structured memories."""

    def run_pass(
        self,
        log: LogStore,
        local: MemoryStore,
        consolidated: MemoryStore,
    ) -> ConsolidationResult: ...
