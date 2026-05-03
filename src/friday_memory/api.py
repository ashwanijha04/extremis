from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from .config import Config
from .embeddings.sentence_transformers import SentenceTransformerEmbedder
from .interfaces import Embedder, LogStore, MemoryStore
from .storage.log import FileLogStore
from .storage.sqlite import SQLiteMemoryStore
from .types import (
    FeedbackSignal,
    LogEntry,
    Memory,
    MemoryLayer,
    RecallResult,
)


def _build_store(config: Config) -> MemoryStore:
    """Select and initialise the memory store from config."""
    if config.store == "postgres":
        if not config.postgres_url:
            raise ValueError(
                "FRIDAY_STORE=postgres requires FRIDAY_POSTGRES_URL to be set."
            )
        from .storage.postgres import PostgresMemoryStore
        return PostgresMemoryStore(config.postgres_url, config)
    return SQLiteMemoryStore(config.resolved_local_db_path(), config)


class FridayMemory:
    """
    The three methods agents actually call: remember, recall, report_outcome.
    Plus remember_now for time-sensitive direct writes.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        log: Optional[LogStore] = None,
        local: Optional[MemoryStore] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self._config = config or Config()
        self._log = log or FileLogStore(self._config.resolved_log_dir())
        self._local = local or _build_store(self._config)
        self._embedder = embedder or SentenceTransformerEmbedder(self._config.embedder)

    def remember(
        self,
        content: str,
        role: str = "user",
        conversation_id: str = "default",
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Append to the log. Cheap path — no LLM, no embedding.
        Also writes an episodic memory to the local store for immediate recall.
        """
        entry = LogEntry(
            role=role,
            content=content,
            conversation_id=conversation_id,
            metadata=metadata or {},
        )
        self._log.append(entry)

        embedding = self._embedder.embed(content)
        memory = Memory(
            layer=MemoryLayer.EPISODIC,
            content=content,
            embedding=embedding,
            metadata={"conversation_id": conversation_id, "role": role, **(metadata or {})},
            validity_start=datetime.now(tz=timezone.utc),
        )
        self._local.store(memory)

    def recall(
        self,
        query: str,
        limit: int = 10,
        layers: Optional[list[MemoryLayer]] = None,
        min_score: float = 0.0,
    ) -> list[RecallResult]:
        """
        Layered retrieval:
        - Identity layer always included (who the user is, invariants)
        - Procedural layer always included (behavioral rules)
        - Semantic + Episodic ranked by relevance × utility × recency
        """
        query_embedding = self._embedder.embed(query)

        # Always pull identity + procedural regardless of layer filter
        pinned_layers = [MemoryLayer.IDENTITY, MemoryLayer.PROCEDURAL]
        pinned = self._local.search(
            query_embedding,
            layers=pinned_layers,
            limit=5,
            min_score=min_score,
        )

        # Ranked recall for semantic + episodic (or custom filter)
        search_layers = layers or [MemoryLayer.SEMANTIC, MemoryLayer.EPISODIC]
        search_layers = [l for l in search_layers if l not in pinned_layers]

        ranked = self._local.search(
            query_embedding,
            layers=search_layers,
            limit=limit,
            min_score=min_score,
        )

        # Merge: pinned first (deduped), then ranked
        seen: set[UUID] = set()
        results: list[RecallResult] = []
        for r in pinned + ranked:
            if r.memory.id not in seen:
                seen.add(r.memory.id)
                results.append(r)

        return results[:limit]

    def report_outcome(
        self,
        memory_ids: list[UUID],
        success: bool,
        weight: float = 1.0,
    ) -> None:
        """RL signal. Adjusts utility scores on the referenced memories."""
        signal = FeedbackSignal(
            memory_ids=memory_ids,
            success=success,
            weight=weight,
        )
        delta = weight if signal.success else -weight
        for mid in memory_ids:
            self._local.update_score(mid, delta)

    def remember_now(
        self,
        content: str,
        layer: MemoryLayer,
        expires_at: Optional[datetime] = None,
        confidence: float = 0.9,
        metadata: Optional[dict] = None,
    ) -> Memory:
        """
        Skip the log; write directly to structured memory.
        Use for time-sensitive facts ('flight Thursday 6am') or high-confidence
        identity/procedural rules that don't need to be derived from logs.
        """
        embedding = self._embedder.embed(content)
        memory = Memory(
            layer=layer,
            content=content,
            embedding=embedding,
            confidence=confidence,
            metadata=metadata or {},
            validity_start=datetime.now(tz=timezone.utc),
            validity_end=expires_at,
        )
        return self._local.store(memory)

    def get_local_store(self) -> MemoryStore:
        return self._local

    def get_log(self) -> LogStore:
        return self._log
