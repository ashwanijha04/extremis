from __future__ import annotations

import os
import re
import threading
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

try:
    from peekr.decorators import trace as _trace
except ImportError:

    def _trace(_func=None, *, name=None, capture_io=True):  # type: ignore[misc]
        def decorator(fn):
            return fn

        return decorator(_func) if _func is not None else decorator


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
    CompactionResult,
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
_peekr_instrumented = False

# Layer trust weights from the production hallucination-detection stack.
# Identity > semantic > procedural > episodic > working. Folded into
# effective_confidence at recall time so callers can hedge ("as of …").
_LAYER_WEIGHTS: dict[MemoryLayer, float] = {
    MemoryLayer.IDENTITY: 0.95,
    MemoryLayer.SEMANTIC: 0.80,
    MemoryLayer.PROCEDURAL: 0.70,
    MemoryLayer.EPISODIC: 0.60,
    MemoryLayer.WORKING: 0.40,
}


def _setup_observability(traces_path: str) -> None:
    global _peekr_instrumented
    if _peekr_instrumented:
        return
    try:
        import pathlib

        import peekr
        from peekr.exporters import JSONLExporter, add_exporter

        pathlib.Path(traces_path).parent.mkdir(parents=True, exist_ok=True)
        add_exporter(JSONLExporter(path=traces_path))
        peekr.instrument(console=False, jsonl_path=None)  # exporters already added above
        _peekr_instrumented = True
    except ImportError:
        pass  # peekr not installed — observability silently disabled


_APPROX_CHARS_PER_TOKEN = 4


def _chunk_content(content: str, chunk_size: int) -> list[str]:
    """Split content at sentence boundaries into chunks of ~chunk_size tokens."""
    if chunk_size <= 0 or len(content) <= chunk_size * _APPROX_CHARS_PER_TOKEN:
        return [content]
    sentences = re.split(r"(?<=[.!?])\s+", content.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    max_chars = chunk_size * _APPROX_CHARS_PER_TOKEN
    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks or [content]


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
    if config.store == "supabase":
        from .storage.supabase_store import SupabaseMemoryStore

        return SupabaseMemoryStore(config)
    if config.store == "s3_vectors":
        if not config.s3_vectors_bucket:
            raise ValueError("EXTREMIS_STORE=s3_vectors requires EXTREMIS_S3_VECTORS_BUCKET to be set.")
        from .storage.s3_vectors import S3VectorsMemoryStore

        return S3VectorsMemoryStore(
            config.s3_vectors_bucket,
            config.s3_vectors_index,
            config,
            region=config.s3_vectors_region,
            score_db_path=config.resolved_s3_vectors_score_db(),
        )
    return SQLiteMemoryStore(config.resolved_local_db_path(), config)


def _compute_effective_confidence(memory: Memory, now: datetime, half_life_days: int) -> float:
    """confidence × layer_weight × 2^(-age_days / half_life).

    Memories past validity_end decay to 0 immediately.
    """
    if memory.validity_end is not None and memory.validity_end < now:
        return 0.0
    layer_weight = _LAYER_WEIGHTS.get(memory.layer, 0.5)
    created = memory.created_at
    # created_at may be naive (utcnow). Treat as UTC.
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age_days = max((now - created).total_seconds() / 86400.0, 0.0)
    decay = 2.0 ** (-age_days / half_life_days)
    return round(memory.confidence * layer_weight * decay, 4)


def _build_sources(
    memory: Memory,
    effective_confidence: float,
    now: datetime,
) -> dict:
    """Project memory provenance into a flat dict for RecallResult.sources.

    Includes recall-time recommendations (stale/expired/contradicted-but-surfacing)
    in addition to any write-time recommendations stamped in metadata.
    """
    from .verification import recommend_for_recall, recommendations_to_metadata

    meta = memory.metadata or {}
    write_recs = meta.get("recommendations") or []
    runtime_recs = recommendations_to_metadata(recommend_for_recall(memory, effective_confidence, now=now))
    return {
        "conversation_id": meta.get("conversation_id"),
        "source_message_ids": meta.get("source_message_ids", []),
        "source_memory_ids": [str(mid) for mid in memory.source_memory_ids],
        "layer": memory.layer.value,
        "created_at": memory.created_at.isoformat() if memory.created_at else None,
        "verification": meta.get("verification"),
        "consistency": meta.get("consistency"),
        "recommendations": write_recs + runtime_recs,
    }


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
        self._remember_count = 0
        self._last_conversation_id: Optional[str] = None
        self._consolidation_lock = threading.Lock()
        if self._config.observe:
            _setup_observability(self._config.resolved_traces_path())

    @_trace(name="extremis.remember", capture_io=False)
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
        Long content is chunked at sentence boundaries for better retrieval granularity.
        """
        entry = LogEntry(
            role=role,
            content=content,
            conversation_id=conversation_id,
            metadata=metadata or {},
        )
        self._log.append(entry)

        chunks = _chunk_content(content, self._config.chunk_size)
        for chunk in chunks:
            embedding = self._embedder.embed(chunk)
            memory = Memory(
                layer=MemoryLayer.EPISODIC,
                content=chunk,
                embedding=embedding,
                metadata={"conversation_id": conversation_id, "role": role, **(metadata or {})},
                validity_start=datetime.now(tz=timezone.utc),
            )
            self._local.store(memory)

        self._remember_count += 1

        # Counter-based trigger (off by default — see Config comment on cost)
        if (
            self._config.auto_consolidate
            and self._remember_count % self._config.auto_consolidate_every == 0
            and os.environ.get("ANTHROPIC_API_KEY")
        ):
            threading.Thread(target=self._background_consolidate, daemon=True).start()

        # Session-end trigger: fires once when conversation_id changes
        if (
            self._config.consolidate_on_session_end
            and self._last_conversation_id is not None
            and conversation_id != self._last_conversation_id
            and os.environ.get("ANTHROPIC_API_KEY")
        ):
            threading.Thread(target=self._background_consolidate, daemon=True).start()

        self._last_conversation_id = conversation_id

    def _background_consolidate(self) -> None:
        if not self._consolidation_lock.acquire(blocking=False):
            return  # another consolidation is already running
        try:
            from .consolidation.consolidator import LLMConsolidator

            consolidator = LLMConsolidator(self._config, self._embedder)
            consolidator.run_pass(self._log, self._local, self._local)
        except Exception:
            pass  # never crash the caller
        finally:
            self._consolidation_lock.release()

    @_trace(name="extremis.recall", capture_io=False)
    def recall(
        self,
        query: str,
        limit: int = 10,
        layers: Optional[list[MemoryLayer]] = None,
        min_score: Optional[float] = None,
    ) -> list[RecallResult]:
        """
        Layered retrieval:
        - Identity layer always included (who the user is, invariants)
        - Procedural layer always included (behavioral rules)
        - Semantic + Episodic ranked by relevance × utility × recency
        """
        # Use configured floor if caller didn't specify — prevents near-zero noise
        effective_min = min_score if min_score is not None else self._config.recall_min_relevance

        query_embedding = self._embedder.embed(query)

        # Always pull identity + procedural regardless of layer filter.
        # Use min_score=0 for pinned layers so identity/procedural always surface
        # even when they're not a strong semantic match.
        pinned_layers = [MemoryLayer.IDENTITY, MemoryLayer.PROCEDURAL]
        pinned = self._local.search(
            query_embedding,
            layers=pinned_layers,
            limit=5,
            min_score=0.0,
        )

        # Ranked recall for semantic + episodic (or the caller's custom filter).
        # Remove pinned layers — they're already covered above.
        # If the caller asked only for pinned layers (e.g. layers=["identity"]),
        # search_layers becomes [] and we skip the ranked search entirely
        # rather than falling back to all layers.
        search_layers = layers or [MemoryLayer.SEMANTIC, MemoryLayer.EPISODIC]
        search_layers = [layer for layer in search_layers if layer not in pinned_layers]

        ranked = (
            self._local.search(
                query_embedding,
                layers=search_layers,
                limit=limit,
                min_score=effective_min,
            )
            if search_layers
            else []
        )

        # Merge: pinned first (deduped), then ranked
        seen: set[UUID] = set()
        results: list[RecallResult] = []
        for r in pinned + ranked:
            if r.memory.id not in seen:
                seen.add(r.memory.id)
                results.append(r)

        # Annotate with effective_confidence (hedging signal) and a
        # structured sources trail (provenance) so callers don't have to
        # rummage through Memory.metadata.
        now = datetime.now(tz=timezone.utc)
        half_life = max(self._config.confidence_half_life_days, 1)
        for r in results:
            r.effective_confidence = _compute_effective_confidence(r.memory, now, half_life)
            r.sources = _build_sources(r.memory, r.effective_confidence, now)

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

        # Write-time dedup: if a nearly-identical memory already exists for this layer,
        # supersede it rather than accumulating duplicates or contradictions.
        # Only applies to semantic and procedural — episodic/identity/working are intentionally appendable.
        if layer in (MemoryLayer.SEMANTIC, MemoryLayer.PROCEDURAL):
            if hasattr(self._local, "find_similar"):
                similar = self._local.find_similar(  # type: ignore[union-attr]
                    embedding,
                    layer,
                    threshold=self._config.dedup_similarity_threshold,
                    limit=1,
                )
                if similar:
                    old_memory, similarity = similar[0]
                    new_memory = Memory(
                        layer=layer,
                        content=content,
                        embedding=embedding,
                        confidence=confidence,
                        metadata={**(metadata or {}), "supersedes_similarity": round(similarity, 3)},
                        validity_start=datetime.now(tz=timezone.utc),
                        validity_end=expires_at,
                        source_memory_ids=[old_memory.id],
                    )
                    self._local.supersede(old_memory.id, new_memory)
                    return new_memory

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

    def compact(
        self,
        layer: MemoryLayer = MemoryLayer.SEMANTIC,
    ) -> CompactionResult:
        """
        Reconcile contradictions in existing structured memories via LLM.

        Different from consolidate():
        - consolidate() reads NEW log entries and distils them into structured memories.
        - compact() works on EXISTING structured memories and resolves conflicts.

        Use when you've accumulated contradictory semantic/procedural memories
        (e.g. 'prefers concise answers' AND 'prefers verbose answers' both in store).
        """
        from .consolidation.compactor import LLMCompactor

        compactor = LLMCompactor(self._config, self._embedder)
        return compactor.run(self._local, layer=layer)

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
            message,
            sender=sender,
            channel=channel,
            owner_ids=owner_ids,
            allowlist=allowlist,
            context=context,
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
