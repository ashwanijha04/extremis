"""
Pinecone memory store adapter.

Vectors live in a Pinecone index. RL scores live in a sidecar SQLiteScoreIndex.
Pinecone namespaces map 1:1 to extremis namespaces.

Install: pip install "extremis[pinecone]"
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID

from ..config import Config
from ..types import Memory, MemoryLayer, RecallResult
from .score_index import SQLiteScoreIndex

_NULL = "__null__"
_LIST_SEP = ","


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _meta_to_memory(vec_id: str, metadata: dict, score: float) -> Memory:
    sid_raw = metadata.get("source_memory_ids", "")
    source_ids = [UUID(x) for x in sid_raw.split(_LIST_SEP) if x]
    ve = metadata.get("validity_end", _NULL)
    la = metadata.get("last_accessed_at", _NULL)
    return Memory(
        id=UUID(vec_id),
        namespace=metadata.get("namespace", "default"),
        layer=MemoryLayer(metadata["layer"]),
        content=metadata["content"],
        embedding=None,
        score=score,
        confidence=float(metadata.get("confidence", 0.5)),
        metadata=json.loads(metadata.get("extra_metadata", "{}")),
        source_memory_ids=source_ids,
        validity_start=datetime.fromisoformat(metadata["validity_start"]),
        validity_end=datetime.fromisoformat(ve) if ve != _NULL else None,
        created_at=datetime.fromisoformat(metadata["created_at"]),
        last_accessed_at=datetime.fromisoformat(la) if la != _NULL else None,
        access_count=int(metadata.get("access_count", 0)),
        do_not_consolidate=bool(int(metadata.get("do_not_consolidate", 0))),
    )


class PineconeMemoryStore:
    """
    MemoryStore backed by Pinecone (serverless or pod-based).

    Requires an existing Pinecone index with the correct dimension.
    Create it beforehand:
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key="...")
        pc.create_index("extremis", dimension=384, metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))

    RL scores are stored in a sidecar SQLite file at {score_db_path}.
    """

    def __init__(self, api_key: str, index_name: str, config: Config, score_db_path: str = "") -> None:
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError("Pinecone store requires: pip install 'extremis[pinecone]'") from None

        self._config = config
        self._ns = config.namespace
        self._index = Pinecone(api_key=api_key).Index(index_name)

        score_path = score_db_path or str(
            Path(config.extremis_home).expanduser() / "pinecone_scores.db"
        )
        self._scores = SQLiteScoreIndex(score_path, self._ns)

    def store(self, memory: Memory) -> Memory:
        memory = memory.model_copy(update={"namespace": self._ns})
        metadata: dict = {
            "namespace": self._ns,
            "layer": memory.layer.value,
            "content": memory.content,           # Pinecone metadata stores the text
            "confidence": memory.confidence,
            "source_memory_ids": _LIST_SEP.join(str(s) for s in memory.source_memory_ids),
            "validity_start": memory.validity_start.isoformat(),
            "validity_end": memory.validity_end.isoformat() if memory.validity_end else _NULL,
            "created_at": memory.created_at.isoformat(),
            "last_accessed_at": memory.last_accessed_at.isoformat() if memory.last_accessed_at else _NULL,
            "access_count": memory.access_count,
            "do_not_consolidate": int(memory.do_not_consolidate),
            "extra_metadata": json.dumps(memory.metadata),
        }
        embedding = memory.embedding or ([0.0] * self._config.embedding_dim)
        self._index.upsert(
            vectors=[{"id": str(memory.id), "values": embedding, "metadata": metadata}],
            namespace=self._ns,
        )
        return memory

    def get(self, memory_id: UUID) -> Optional[Memory]:
        result = self._index.fetch(ids=[str(memory_id)], namespace=self._ns)
        vectors = result.get("vectors", {})
        if not vectors:
            return None
        vec = vectors[str(memory_id)]
        score = self._scores.get(memory_id)
        return _meta_to_memory(vec["id"], vec["metadata"], score)

    def search(
        self,
        query_embedding: list[float],
        layers: Optional[list[MemoryLayer]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[RecallResult]:
        pinecone_filter: dict = {"validity_end": {"$eq": _NULL}}
        if layers:
            pinecone_filter["layer"] = {"$in": [lyr.value for lyr in layers]}

        result = self._index.query(
            vector=query_embedding,
            top_k=limit * 3,
            namespace=self._ns,
            filter=pinecone_filter,
            include_metadata=True,
        )

        scores_map = self._scores.get_all()
        results: list[RecallResult] = []

        for match in result.get("matches", []):
            relevance = float(match["score"])  # Pinecone returns cosine similarity directly
            utility_score = scores_map.get(match["id"], 0.0)
            final_rank = self._rank(relevance, utility_score, match["metadata"]["created_at"])

            if final_rank >= min_score:
                mem = _meta_to_memory(match["id"], match["metadata"], utility_score)
                results.append(RecallResult(memory=mem, relevance=relevance, final_rank=final_rank))

        results.sort(key=lambda r: r.final_rank, reverse=True)
        return results[:limit]

    def update_score(self, memory_id: UUID, delta: float) -> None:
        self._scores.update(memory_id, delta)

    def supersede(self, old_id: UUID, new_memory: Memory) -> None:
        existing = self.get(old_id)
        if existing:
            updated = existing.model_copy(update={"validity_end": datetime.now(tz=timezone.utc)})
            self.store(updated)
        self.store(new_memory)

    def list_recent(
        self,
        layer: Optional[MemoryLayer] = None,
        limit: int = 50,
    ) -> list[Memory]:
        # Pinecone doesn't support list-all natively; use a zero vector query
        pinecone_filter: dict = {"validity_end": {"$eq": _NULL}}
        if layer:
            pinecone_filter["layer"] = {"$eq": layer.value}

        result = self._index.query(
            vector=[0.0] * self._config.embedding_dim,
            top_k=limit,
            namespace=self._ns,
            filter=pinecone_filter,
            include_metadata=True,
        )
        scores_map = self._scores.get_all()
        memories = [
            _meta_to_memory(m["id"], m["metadata"], scores_map.get(m["id"], 0.0))
            for m in result.get("matches", [])
        ]
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories

    def _rank(self, relevance: float, score: float, created_at_iso: str) -> float:
        alpha = self._config.rl_alpha
        half_life = self._config.recency_half_life_days
        utility = 1.0 + alpha * math.tanh(score)
        created = datetime.fromisoformat(created_at_iso)
        age_days = (datetime.now(tz=timezone.utc) - created.replace(tzinfo=timezone.utc)).days
        recency = math.exp(-math.log(2) * age_days / half_life)
        return relevance * utility * recency

    def close(self) -> None:
        self._scores.close()
