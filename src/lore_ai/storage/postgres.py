from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID

from ..config import Config
from ..types import Memory, MemoryLayer, RecallResult


def _row_to_memory(row: dict) -> Memory:
    return Memory(
        id=UUID(str(row["id"])),
        layer=MemoryLayer(row["layer"]),
        content=row["content"],
        embedding=None,
        score=float(row["score"]),
        confidence=float(row["confidence"]),
        metadata=row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"]),
        source_memory_ids=[UUID(str(x)) for x in (row["source_memory_ids"] or [])],
        validity_start=row["validity_start"],
        validity_end=row["validity_end"],
        created_at=row["created_at"],
        last_accessed_at=row["last_accessed_at"],
        access_count=row["access_count"],
        do_not_consolidate=row["do_not_consolidate"],
    )


class PostgresMemoryStore:
    """
    Memory store backed by Postgres + pgvector.
    Ranking (cosine × utility × recency) runs in SQL — no Python loop over rows.

    Requires:
        pip install "lore-ai[postgres]"
    and a running Postgres with pgvector:
        CREATE EXTENSION vector;
    """

    def __init__(self, url: str, config: Config) -> None:
        try:
            import psycopg2
            import psycopg2.extras
            from pgvector.psycopg2 import register_vector
        except ImportError:
            raise ImportError(
                "Postgres store requires: pip install 'lore-ai[postgres]'"
            ) from None

        self._config = config
        self._conn = psycopg2.connect(url)
        self._conn.autocommit = False
        register_vector(self._conn)
        self._init_schema()

    def _init_schema(self) -> None:
        schema_path = Path(__file__).parent.parent / "migrations" / "001_initial.sql"
        with self._conn.cursor() as cur:
            cur.execute(schema_path.read_text())
        self._conn.commit()

    def store(self, memory: Memory) -> Memory:
        import numpy as np
        embedding = np.array(memory.embedding, dtype=np.float32) if memory.embedding else None

        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memories (
                    id, layer, content, embedding, score, confidence,
                    metadata, source_memory_ids,
                    validity_start, validity_end,
                    created_at, last_accessed_at, access_count,
                    do_not_consolidate
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s
                )
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    score = EXCLUDED.score,
                    confidence = EXCLUDED.confidence,
                    metadata = EXCLUDED.metadata,
                    validity_end = EXCLUDED.validity_end,
                    last_accessed_at = EXCLUDED.last_accessed_at,
                    access_count = EXCLUDED.access_count
                """,
                (
                    str(memory.id),
                    memory.layer.value,
                    memory.content,
                    embedding,
                    memory.score,
                    memory.confidence,
                    json.dumps(memory.metadata),
                    [str(sid) for sid in memory.source_memory_ids],
                    memory.validity_start,
                    memory.validity_end,
                    memory.created_at,
                    memory.last_accessed_at,
                    memory.access_count,
                    memory.do_not_consolidate,
                ),
            )
        self._conn.commit()
        return memory

    def get(self, memory_id: UUID) -> Optional[Memory]:
        with self._conn.cursor(cursor_factory=self._dict_cursor()) as cur:
            cur.execute(
                "SELECT * FROM memories WHERE id = %s",
                (str(memory_id),),
            )
            row = cur.fetchone()
        return _row_to_memory(row) if row else None

    def search(
        self,
        query_embedding: list[float],
        layers: Optional[list[MemoryLayer]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[RecallResult]:
        import numpy as np

        alpha = self._config.rl_alpha
        half_life = self._config.recency_half_life_days
        vec = np.array(query_embedding, dtype=np.float32)

        layer_values = [l.value for l in layers] if layers else None

        with self._conn.cursor(cursor_factory=self._dict_cursor()) as cur:
            cur.execute(
                """
                SELECT
                    *,
                    1 - (embedding <=> %(vec)s) AS relevance,
                    (1 - (embedding <=> %(vec)s))
                      * (1 + %(alpha)s * tanh(score))
                      * exp(-0.693147
                            * EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0
                            / %(half_life)s)
                    AS final_rank
                FROM memories
                WHERE embedding IS NOT NULL
                  AND validity_end IS NULL
                  AND (%(layers)s::text[] IS NULL OR layer = ANY(%(layers)s::text[]))
                ORDER BY final_rank DESC
                LIMIT %(limit)s
                """,
                {
                    "vec": vec,
                    "alpha": alpha,
                    "half_life": float(half_life),
                    "layers": layer_values,
                    "limit": limit,
                },
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            if float(row["final_rank"]) < min_score:
                continue
            results.append(
                RecallResult(
                    memory=_row_to_memory(row),
                    relevance=float(row["relevance"]),
                    final_rank=float(row["final_rank"]),
                )
            )

        # Touch access stats
        if results:
            ids = [str(r.memory.id) for r in results]
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1, last_accessed_at = NOW()
                    WHERE id = ANY(%s)
                    """,
                    (ids,),
                )
            self._conn.commit()

        return results

    def update_score(self, memory_id: UUID, delta: float) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE memories SET score = score + %s WHERE id = %s",
                (delta, str(memory_id)),
            )
        self._conn.commit()

    def supersede(self, old_id: UUID, new_memory: Memory) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE memories SET validity_end = NOW() WHERE id = %s",
                (str(old_id),),
            )
        self.store(new_memory)

    def list_recent(
        self,
        layer: Optional[MemoryLayer] = None,
        limit: int = 50,
    ) -> list[Memory]:
        with self._conn.cursor(cursor_factory=self._dict_cursor()) as cur:
            if layer:
                cur.execute(
                    """
                    SELECT * FROM memories
                    WHERE layer = %s AND validity_end IS NULL
                    ORDER BY created_at DESC LIMIT %s
                    """,
                    (layer.value, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM memories
                    WHERE validity_end IS NULL
                    ORDER BY created_at DESC LIMIT %s
                    """,
                    (limit,),
                )
            rows = cur.fetchall()
        return [_row_to_memory(r) for r in rows]

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _dict_cursor():
        import psycopg2.extras
        return psycopg2.extras.RealDictCursor
