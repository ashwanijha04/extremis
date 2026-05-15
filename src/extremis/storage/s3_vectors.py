"""
Amazon S3 Vectors memory store adapter.

Vectors live in an S3 vector bucket + index. RL scores live in a sidecar
SQLiteScoreIndex (S3 Vectors charges per metadata update and limits
filterable-key cardinality, so we don't write scores back to AWS).

S3 Vectors is optimized for cheap, durable, large-scale storage rather than
low-latency hot recall — expect query latency in the 100s of ms. Treat it as
the archival / scale tier in a tiered setup; for chat-rate workloads, Pinecone
or pgvector remain the right primary.

Install:
    pip install "extremis[s3-vectors]"

Setup (AWS-side, one-time):
    aws s3vectors create-vector-bucket --vector-bucket-name extremis-vectors
    aws s3vectors create-index \\
        --vector-bucket-name extremis-vectors \\
        --index-name extremis \\
        --data-type float32 \\
        --dimension 384 \\
        --distance-metric cosine \\
        --metadata-configuration 'nonFilterableMetadataKeys=[
            "content","extra_metadata","source_memory_ids","confidence",
            "created_at","validity_start","last_accessed_at",
            "access_count","do_not_consolidate"
        ]'

Note: S3 Vectors allows at most 10 filterable metadata keys per index.
We keep only `namespace`, `layer`, and `validity_end` filterable — everything
else is stored as non-filterable metadata.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from ..config import Config
from ..types import Memory, MemoryLayer, RecallResult
from .score_index import SQLiteScoreIndex

_NULL = "__null__"
_LIST_SEP = ","


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _meta_to_memory(vec_key: str, metadata: dict, score: float) -> Memory:
    sid_raw = metadata.get("source_memory_ids", "")
    source_ids = [UUID(x) for x in sid_raw.split(_LIST_SEP) if x]
    ve = metadata.get("validity_end", _NULL)
    la = metadata.get("last_accessed_at", _NULL)
    return Memory(
        id=UUID(vec_key),
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


class S3VectorsMemoryStore:
    """
    MemoryStore backed by Amazon S3 Vectors.

    Requires an existing vector bucket + index with the correct dimension
    and `namespace`, `layer`, `validity_end` declared as filterable metadata
    keys (everything else non-filterable). See module docstring for the
    one-time AWS setup command.

    RL scores are kept in a sidecar SQLite file at `score_db_path`.
    """

    def __init__(
        self,
        bucket: str,
        index_name: str,
        config: Config,
        region: str = "",
        score_db_path: str = "",
    ) -> None:
        try:
            import boto3  # type: ignore
        except ImportError:
            raise ImportError("S3 Vectors store requires: pip install 'extremis[s3-vectors]'") from None

        if not bucket:
            raise ValueError("S3 Vectors store requires a vector bucket name")
        if not index_name:
            raise ValueError("S3 Vectors store requires an index name")

        self._config = config
        self._ns = config.namespace
        self._bucket = bucket
        self._index_name = index_name
        client_kwargs: dict = {}
        if region:
            client_kwargs["region_name"] = region
        self._client = boto3.client("s3vectors", **client_kwargs)

        score_path = score_db_path or str(Path(config.extremis_home).expanduser() / "s3_vectors_scores.db")
        self._scores = SQLiteScoreIndex(score_path, self._ns)

    # ------------------------------------------------------------------ #
    # Metadata layout
    # ------------------------------------------------------------------ #

    def _build_metadata(self, memory: Memory) -> dict[str, Any]:
        return {
            # Filterable (must match the index's filterable-key configuration)
            "namespace": self._ns,
            "layer": memory.layer.value,
            "validity_end": memory.validity_end.isoformat() if memory.validity_end else _NULL,
            # Non-filterable
            "content": memory.content,
            "confidence": memory.confidence,
            "source_memory_ids": _LIST_SEP.join(str(s) for s in memory.source_memory_ids),
            "validity_start": memory.validity_start.isoformat(),
            "created_at": memory.created_at.isoformat(),
            "last_accessed_at": memory.last_accessed_at.isoformat() if memory.last_accessed_at else _NULL,
            "access_count": memory.access_count,
            "do_not_consolidate": int(memory.do_not_consolidate),
            "extra_metadata": json.dumps(memory.metadata),
        }

    def _base_filter(self, layers: Optional[list[MemoryLayer]] = None) -> dict[str, Any]:
        flt: dict[str, Any] = {
            "namespace": {"$eq": self._ns},
            "validity_end": {"$eq": _NULL},
        }
        if layers:
            flt["layer"] = {"$in": [lyr.value for lyr in layers]}
        return flt

    # ------------------------------------------------------------------ #
    # MemoryStore protocol
    # ------------------------------------------------------------------ #

    def store(self, memory: Memory) -> Memory:
        memory = memory.model_copy(update={"namespace": self._ns})
        embedding = memory.embedding or ([0.0] * self._config.embedding_dim)
        self._client.put_vectors(
            vectorBucketName=self._bucket,
            indexName=self._index_name,
            vectors=[
                {
                    "key": str(memory.id),
                    "data": {"float32": embedding},
                    "metadata": self._build_metadata(memory),
                }
            ],
        )
        return memory

    def get(self, memory_id: UUID) -> Optional[Memory]:
        result = self._client.get_vectors(
            vectorBucketName=self._bucket,
            indexName=self._index_name,
            keys=[str(memory_id)],
            returnMetadata=True,
        )
        vectors = result.get("vectors", [])
        if not vectors:
            return None
        vec = vectors[0]
        if vec.get("metadata", {}).get("namespace") != self._ns:
            return None  # cross-tenant isolation guard
        return _meta_to_memory(vec["key"], vec["metadata"], self._scores.get(memory_id))

    def search(
        self,
        query_embedding: list[float],
        layers: Optional[list[MemoryLayer]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[RecallResult]:
        result = self._client.query_vectors(
            vectorBucketName=self._bucket,
            indexName=self._index_name,
            queryVector={"float32": query_embedding},
            topK=limit * 3,
            filter=self._base_filter(layers),
            returnMetadata=True,
            returnDistance=True,
        )

        scores_map = self._scores.get_all()
        results: list[RecallResult] = []
        for match in result.get("vectors", []):
            # S3 Vectors returns *distance*. For cosine, similarity = 1 - distance.
            distance = float(match.get("distance", 0.0))
            relevance = max(0.0, min(1.0, 1.0 - distance))
            utility_score = scores_map.get(match["key"], 0.0)
            metadata = match["metadata"]
            final_rank = self._rank(relevance, utility_score, metadata["created_at"])
            if final_rank >= min_score:
                mem = _meta_to_memory(match["key"], metadata, utility_score)
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
        # S3 Vectors supports list_vectors but doesn't sort by recency or filter
        # at list time; we use a zero-vector query with the recency filter and
        # sort client-side, mirroring the Pinecone adapter.
        layers = [layer] if layer else None
        result = self._client.query_vectors(
            vectorBucketName=self._bucket,
            indexName=self._index_name,
            queryVector={"float32": [0.0] * self._config.embedding_dim},
            topK=limit,
            filter=self._base_filter(layers),
            returnMetadata=True,
            returnDistance=False,
        )
        scores_map = self._scores.get_all()
        memories = [
            _meta_to_memory(m["key"], m["metadata"], scores_map.get(m["key"], 0.0)) for m in result.get("vectors", [])
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
