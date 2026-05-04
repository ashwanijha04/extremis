"""
Lightweight SQLite score store for external vector backend adapters.

When memories live in Pinecone or Chroma, RL scores can't be stored there
efficiently (Pinecone charges per metadata write; Chroma doesn't support
partial updates). This index lives next to the external store as a tiny
SQLite file and owns the score column.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID


class SQLiteScoreIndex:
    def __init__(self, db_path: str, namespace: str = "default") -> None:
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ns = namespace
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                memory_id TEXT NOT NULL,
                namespace  TEXT NOT NULL DEFAULT 'default',
                score      REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (memory_id, namespace)
            )
        """)
        self._conn.commit()

    def get(self, memory_id: UUID) -> float:
        row = self._conn.execute(
            "SELECT score FROM scores WHERE memory_id = ? AND namespace = ?",
            (str(memory_id), self._ns),
        ).fetchone()
        return row[0] if row else 0.0

    def update(self, memory_id: UUID, delta: float) -> None:
        self._conn.execute(
            """
            INSERT INTO scores (memory_id, namespace, score) VALUES (?, ?, ?)
            ON CONFLICT(memory_id, namespace) DO UPDATE SET score = score + excluded.score
            """,
            (str(memory_id), self._ns, delta),
        )
        self._conn.commit()

    def get_all(self) -> dict[str, float]:
        rows = self._conn.execute(
            "SELECT memory_id, score FROM scores WHERE namespace = ?", (self._ns,)
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def close(self) -> None:
        self._conn.close()
