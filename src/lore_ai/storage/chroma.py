"""
ChromaDB memory store adapter.

Vectors live in Chroma. RL scores live in a sidecar SQLiteScoreIndex.
The knowledge graph is always local SQLite (unchanged).

Install: pip install "lore-ai[chroma]"
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np

from ..config import Config
from ..types import Memory, MemoryLayer, RecallResult
from .score_index import SQLiteScoreIndex

_NULL_DATETIME = ""          # Chroma metadata can't store None; use empty string
_LIST_SEP = ","              # separator for source_memory_ids stored as string


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _meta_to_memory(doc_id: str, document: str, metadata: dict, score: float) -> Memory:
    sid_raw = metadata.get("source_memory_ids", "")
    source_ids = [UUID(x) for x in sid_raw.split(_LIST_SEP) if x]

    validity_end_raw = metadata.get("validity_end", _NULL_DATETIME)
    last_accessed_raw = metadata.get("last_accessed_at", _NULL_DATETIME)

    return Memory(
        id=UUID(doc_id),
        namespace=metadata.get("namespace", "default"),
        layer=MemoryLayer(metadata["layer"]),
        content=document,
        embedding=None,
        score=score,
        confidence=float(metadata.get("confidence", 0.5)),
        metadata=json.loads(metadata.get("extra_metadata", "{}")),
        source_memory_ids=source_ids,
        validity_start=datetime.fromisoformat(metadata["validity_start"]),
        validity_end=datetime.fromisoformat(validity_end_raw) if validity_end_raw else None,
        created_at=datetime.fromisoformat(metadata["created_at"]),
        last_accessed_at=datetime.fromisoformat(last_accessed_raw) if last_accessed_raw else None,
        access_count=int(metadata.get("access_count", 0)),
        do_not_consolidate=bool(metadata.get("do_not_consolidate", False)),
    )


class ChromaMemoryStore:
    """
    MemoryStore backed by ChromaDB (local persistent client).

    Each namespace gets its own Chroma collection.
    RL scores are stored in a sidecar SQLite file.
    """

    def __init__(self, chroma_path: str, config: Config) -> None:
        try:
            import chromadb
        except ImportError:
            raise ImportError("ChromaDB store requires: pip install 'lore-ai[chroma]'") from None

        base = Path(chroma_path).expanduser()
        base.mkdir(parents=True, exist_ok=True)

        self._config = config
        self._ns = config.namespace
        self._client = chromadb.PersistentClient(path=str(base))
        self._col = self._client.get_or_create_collection(
            name=f"lore_ai_{self._ns}",
            metadata={"hnsw:space": "cosine"},
        )
        self._scores = SQLiteScoreIndex(str(base / "scores.db"), self._ns)

    def store(self, memory: Memory) -> Memory:
        memory = memory.model_copy(update={"namespace": self._ns})
        meta: dict = {
            "namespace": self._ns,
            "layer": memory.layer.value,
            "confidence": memory.confidence,
            "source_memory_ids": _LIST_SEP.join(str(s) for s in memory.source_memory_ids),
            "validity_start": memory.validity_start.isoformat(),
            "validity_end": memory.validity_end.isoformat() if memory.validity_end else _NULL_DATETIME,
            "created_at": memory.created_at.isoformat(),
            "last_accessed_at": memory.last_accessed_at.isoformat() if memory.last_accessed_at else _NULL_DATETIME,
            "access_count": memory.access_count,
            "do_not_consolidate": int(memory.do_not_consolidate),
            "extra_metadata": json.dumps(memory.metadata),
        }
        embedding = memory.embedding or ([0.0] * self._config.embedding_dim)
        self._col.upsert(
            ids=[str(memory.id)],
            embeddings=[embedding],
            documents=[memory.content],
            metadatas=[meta],
        )
        return memory

    def get(self, memory_id: UUID) -> Optional[Memory]:
        result = self._col.get(ids=[str(memory_id)], include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        score = self._scores.get(memory_id)
        return _meta_to_memory(result["ids"][0], result["documents"][0], result["metadatas"][0], score)

    def search(
        self,
        query_embedding: list[float],
        layers: Optional[list[MemoryLayer]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[RecallResult]:
        where: Optional[dict] = None
        validity_filter = {"validity_end": {"$eq": _NULL_DATETIME}}

        if layers:
            layer_filter = {"layer": {"$in": [l.value for l in layers]}}
            where = {"$and": [validity_filter, layer_filter]}
        else:
            where = validity_filter

        n_results = min(max(limit * 3, 20), self._col.count() or 1)

        result = self._col.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        if not result["ids"] or not result["ids"][0]:
            return []

        scores_map = self._scores.get_all()
        results: list[RecallResult] = []

        for doc_id, document, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            relevance = max(0.0, 1.0 - float(distance))  # cosine distance → similarity
            utility_score = scores_map.get(doc_id, 0.0)
            final_rank = self._rank(relevance, utility_score, metadata["created_at"])

            if final_rank >= min_score:
                mem = _meta_to_memory(doc_id, document, metadata, utility_score)
                results.append(RecallResult(memory=mem, relevance=relevance, final_rank=final_rank))

        results.sort(key=lambda r: r.final_rank, reverse=True)
        top = results[:limit]

        if top:
            now = _now_iso()
            for r in top:
                meta_update = {**self._col.get(ids=[str(r.memory.id)], include=["metadatas"])["metadatas"][0]}
                meta_update["last_accessed_at"] = now
                meta_update["access_count"] = str(int(meta_update.get("access_count", 0)) + 1)
                self._col.update(ids=[str(r.memory.id)], metadatas=[meta_update])

        return top

    def update_score(self, memory_id: UUID, delta: float) -> None:
        self._scores.update(memory_id, delta)

    def supersede(self, old_id: UUID, new_memory: Memory) -> None:
        result = self._col.get(ids=[str(old_id)], include=["metadatas"])
        if result["ids"]:
            meta = result["metadatas"][0]
            meta["validity_end"] = _now_iso()
            self._col.update(ids=[str(old_id)], metadatas=[meta])
        self.store(new_memory)

    def list_recent(
        self,
        layer: Optional[MemoryLayer] = None,
        limit: int = 50,
    ) -> list[Memory]:
        where: dict
        validity_filter = {"validity_end": {"$eq": _NULL_DATETIME}}
        if layer:
            where = {"$and": [validity_filter, {"layer": {"$eq": layer.value}}]}
        else:
            where = validity_filter

        count = self._col.count()
        if count == 0:
            return []

        result = self._col.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )
        scores_map = self._scores.get_all()
        memories = [
            _meta_to_memory(doc_id, doc, meta, scores_map.get(doc_id, 0.0))
            for doc_id, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
        ]
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[:limit]

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
