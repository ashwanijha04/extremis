from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from .config import Config
from .embeddings.sentence_transformers import SentenceTransformerEmbedder
from .interfaces import Embedder, LogStore, MemoryStore
from .observer.observer import HeuristicObserver
from .scorer.attention import AttentionScorer
from .storage.kg import SQLiteKGStore
from .storage.log import FileLogStore
from .storage.sqlite import SQLiteMemoryStore
from .types import (
    AttentionResult,
    EntityResult,
    EntityType,
    FeedbackSignal,
    LogEntry,
    Memory,
    MemoryLayer,
    Observation,
    RecallResult,
)

_NEGATIVE_WEIGHT_MULTIPLIER = 1.5  # match friday-saas asymmetric RL weighting


def _build_store(config: Config) -> MemoryStore:
    """Select and initialise the memory store from config."""
    if config.store == "postgres":
        if not config.postgres_url:
            raise ValueError("EXTREMIS_STORE=postgres requires EXTREMIS_POSTGRES_URL to be set.")
        from .storage.postgres import PostgresMemoryStore
        return PostgresMemoryStore(config.postgres_url, config)
    if config.store == "chroma":
        from .storage.chroma import ChromaMemoryStore
        return ChromaMemoryStore(config.resolved_chroma_path(), config)
    if config.store == "pinecone":
        if not config.pinecone_api_key:
            raise ValueError("EXTREMIS_STORE=pinecone requires EXTREMIS_PINECONE_API_KEY to be set.")
        from .storage.pinecone_store import PineconeMemoryStore
        return PineconeMemoryStore(
            config.pinecone_api_key,
            config.pinecone_index,
            config,
            score_db_path=config.resolved_pinecone_score_db(),
        )
    return SQLiteMemoryStore(config.resolved_local_db_path(), config)


def _build_embedder(config: Config) -> Embedder:
    """Select embedder based on model name."""
    if config.embedder.startswith("text-embedding"):
        from .embeddings.openai import OpenAIEmbedder
        return OpenAIEmbedder(config.embedder, config.openai_api_key or None)
    return SentenceTransformerEmbedder(config.embedder)


class Extremis:
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
        self._log = log or FileLogStore(
            self._config.resolved_log_dir(),
            namespace=self._config.namespace,
        )
        self._local = local or _build_store(self._config)
        self._embedder = embedder or _build_embedder(self._config)
        self._kg = SQLiteKGStore(self._config.resolved_local_db_path(), self._config)
        self._observer = HeuristicObserver(namespace=self._config.namespace)
        self._attention = AttentionScorer(self._config)

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
        search_layers = [layer for layer in search_layers if layer not in pinned_layers]

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
        """
        RL signal. Adjusts utility scores on the referenced memories.
        Negative signals are amplified by 1.5× (mirrors human memory asymmetry).
        """
        FeedbackSignal(memory_ids=memory_ids, success=success, weight=weight)
        delta = weight if success else -(weight * _NEGATIVE_WEIGHT_MULTIPLIER)
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

    # ------------------------------------------------------------------ #
    # Knowledge graph
    # ------------------------------------------------------------------ #

    def kg_add_entity(
        self,
        name: str,
        type: EntityType,
        metadata: Optional[dict] = None,
    ):
        return self._kg.add_entity(name, type, metadata)

    def kg_add_relationship(
        self,
        from_entity: str,
        to_entity: str,
        rel_type: str,
        weight: float = 1.0,
        metadata: Optional[dict] = None,
    ):
        return self._kg.add_relationship(from_entity, to_entity, rel_type, weight, metadata)

    def kg_add_attribute(self, entity: str, key: str, value: str):
        return self._kg.add_attribute(entity, key, value)

    def kg_query(self, name: str) -> Optional[EntityResult]:
        return self._kg.query_entity(name)

    def kg_traverse(self, name: str, depth: int = 2) -> list[EntityResult]:
        return self._kg.traverse(name, depth)

    # ------------------------------------------------------------------ #
    # Observer
    # ------------------------------------------------------------------ #

    def observe(self, conversation_id: str = "default") -> list[Observation]:
        """Compress recent log entries for conversation_id into priority observations."""
        all_entries = self._log.read_since(None)
        entries = [e for e in all_entries if e.conversation_id == conversation_id]
        return self._observer.compress(entries)

    # ------------------------------------------------------------------ #
    # Attention scoring
    # ------------------------------------------------------------------ #

    def score_attention(
        self,
        message: str,
        sender: str = "",
        channel: str = "dm",
        owner_ids: Optional[set[str]] = None,
        allowlist: Optional[set[str]] = None,
        context: Optional[dict] = None,
    ) -> AttentionResult:
        return self._attention.score(
            message, sender=sender, channel=channel,
            owner_ids=owner_ids, allowlist=allowlist, context=context,
        )

    # ------------------------------------------------------------------ #
    # Internal accessors (used by consolidator, MCP server, tests)
    # ------------------------------------------------------------------ #

    def get_local_store(self) -> MemoryStore:
        return self._local

    def get_log(self) -> LogStore:
        return self._log

    def get_kg(self) -> SQLiteKGStore:
        return self._kg
