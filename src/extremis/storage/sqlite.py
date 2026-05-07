from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np

from ..config import Config
from ..types import Memory, MemoryLayer, RecallResult


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=UUID(row["id"]),
        layer=MemoryLayer(row["layer"]),
        content=row["content"],
        embedding=None,  # not returned by default; load separately if needed
        score=row["score"],
        confidence=row["confidence"],
        metadata=json.loads(row["metadata"]),
        source_memory_ids=[UUID(x) for x in json.loads(row["source_memory_ids"])],
        validity_start=datetime.fromisoformat(row["validity_start"]),
        validity_end=datetime.fromisoformat(row["validity_end"]) if row["validity_end"] else None,
        created_at=datetime.fromisoformat(row["created_at"]),
        last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]) if row["last_accessed_at"] else None,
        access_count=row["access_count"],
        do_not_consolidate=bool(row["do_not_consolidate"]),
    )


class SQLiteMemoryStore:
    """
    Local memory store backed by SQLite.
    Embeddings stored as float32 BLOB; cosine similarity computed in numpy.
    """

    def __init__(self, db_path: str, config: Config) -> None:
        self._path = Path(db_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._config = config
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        schema_path = Path(__file__).parent.parent / "migrations" / "001_initial_sqlite.sql"
        self._conn.executescript(schema_path.read_text())
        # Add namespace column to existing DBs that predate this field
        try:
            self._conn.execute("ALTER TABLE memories ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'")
        except sqlite3.OperationalError:
            pass  # column already exists
        self._conn.commit()

    @property
    def _ns(self) -> str:
        return self._config.namespace

    def store(self, memory: Memory) -> Memory:
        memory = memory.model_copy(update={"namespace": self._ns})
        embedding_blob = np.array(memory.embedding, dtype=np.float32).tobytes() if memory.embedding else None
        self._conn.execute(
            """
            INSERT OR REPLACE INTO memories (
                id, namespace, layer, content, embedding, score, confidence,
                metadata, source_memory_ids,
                validity_start, validity_end,
                created_at, last_accessed_at, access_count,
                do_not_consolidate
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?,
                ?
            )
            """,
            (
                str(memory.id),
                memory.namespace,
                memory.layer.value,
                memory.content,
                embedding_blob,
                memory.score,
                memory.confidence,
                json.dumps(memory.metadata),
                json.dumps([str(sid) for sid in memory.source_memory_ids]),
                memory.validity_start.isoformat(),
                memory.validity_end.isoformat() if memory.validity_end else None,
                memory.created_at.isoformat(),
                memory.last_accessed_at.isoformat() if memory.last_accessed_at else None,
                memory.access_count,
                int(memory.do_not_consolidate),
            ),
        )
        self._conn.commit()
        return memory

    def get(self, memory_id: UUID) -> Optional[Memory]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ? AND namespace = ?",
            (str(memory_id), self._ns),
        ).fetchone()
        return _row_to_memory(row) if row else None

    def search(
        self,
        query_embedding: list[float],
        layers: Optional[list[MemoryLayer]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[RecallResult]:
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        layer_filter = ""
        params: list = [self._ns]
        if layers:
            placeholders = ",".join("?" * len(layers))
            layer_filter = f"AND layer IN ({placeholders})"
            params += [lyr.value for lyr in layers]

        rows = self._conn.execute(
            f"""
            SELECT * FROM memories
            WHERE namespace = ?
              AND embedding IS NOT NULL
              AND validity_end IS NULL
              {layer_filter}
            """,
            params,
        ).fetchall()

        results = []
        for row in rows:
            stored_vec = np.frombuffer(row["embedding"], dtype=np.float32)
            if len(stored_vec) != len(query_vec):
                continue

            relevance = self._cosine(query_vec, float(query_norm), stored_vec)
            final_rank = self._rank(relevance, row["score"], row["created_at"])

            if final_rank >= min_score:
                from .recall_reason import build_reason

                mem = _row_to_memory(row)
                results.append(
                    RecallResult(
                        memory=mem,
                        relevance=float(relevance),
                        final_rank=float(final_rank),
                        reason=build_reason(
                            float(relevance),
                            row["score"],
                            row["access_count"],
                            row["created_at"],
                            mem.layer,
                        ),
                    )
                )

        results.sort(key=lambda r: r.final_rank, reverse=True)
        top = results[:limit]

        # update access stats for returned memories
        if top:
            now = _now_iso()
            ids = [str(r.memory.id) for r in top]
            self._conn.execute(
                f"""
                UPDATE memories
                SET access_count = access_count + 1, last_accessed_at = ?
                WHERE id IN ({",".join("?" * len(ids))})
                """,
                [now] + ids,
            )
            self._conn.commit()

        return top

    def update_score(self, memory_id: UUID, delta: float) -> None:
        self._conn.execute(
            "UPDATE memories SET score = score + ? WHERE id = ? AND namespace = ?",
            (delta, str(memory_id), self._ns),
        )
        self._conn.commit()

    def supersede(self, old_id: UUID, new_memory: Memory) -> None:
        now = _now_iso()
        self._conn.execute(
            "UPDATE memories SET validity_end = ? WHERE id = ? AND namespace = ?",
            (now, str(old_id), self._ns),
        )
        self.store(new_memory)

    def list_recent(
        self,
        layer: Optional[MemoryLayer] = None,
        limit: int = 50,
    ) -> list[Memory]:
        if layer:
            rows = self._conn.execute(
                (
                    "SELECT * FROM memories"
                    " WHERE namespace = ? AND layer = ? AND validity_end IS NULL"
                    " ORDER BY created_at DESC LIMIT ?"
                ),
                (self._ns, layer.value, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE namespace = ? AND validity_end IS NULL ORDER BY created_at DESC LIMIT ?",
                (self._ns, limit),
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    @staticmethod
    def _cosine(query: np.ndarray, query_norm: float, stored: np.ndarray) -> float:
        if query_norm == 0:
            return 0.0
        stored_norm = np.linalg.norm(stored)
        if stored_norm == 0:
            return 0.0
        return float(np.dot(query, stored) / (query_norm * stored_norm))

    def _rank(self, relevance: float, score: float, created_at_iso: str) -> float:
        """relevance × utility × recency decay."""
        alpha = self._config.rl_alpha
        half_life = self._config.recency_half_life_days

        utility = 1.0 + alpha * math.tanh(score)  # tanh keeps it bounded

        created = datetime.fromisoformat(created_at_iso)
        age_days = (datetime.now(tz=timezone.utc) - created.replace(tzinfo=timezone.utc)).days
        recency = math.exp(-math.log(2) * age_days / half_life)

        return relevance * utility * recency

    def close(self) -> None:
        self._conn.close()
