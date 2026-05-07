"""
API key management for the extremis hosted server.

Keys format: extremis_sk_<32 url-safe base64 chars>
Keys are stored hashed (sha256). The plaintext is only shown once at creation.

Storage: SQLite file at {server_home}/keys.db
"""
from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_PREFIX = "extremis_sk_"


def generate_key() -> str:
    return _PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class KeyStore:
    def __init__(self, db_path: str) -> None:
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key_hash    TEXT PRIMARY KEY,
                namespace   TEXT NOT NULL,
                label       TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL,
                last_used   TEXT,
                call_count  INTEGER NOT NULL DEFAULT 0,
                revoked     INTEGER NOT NULL DEFAULT 0
            )
        """)
        self._conn.commit()

    def create(self, namespace: str, label: str = "") -> str:
        """Generate a new key, store its hash, return the plaintext (shown once)."""
        key = generate_key()
        self._conn.execute(
            "INSERT INTO api_keys (key_hash, namespace, label, created_at) VALUES (?, ?, ?, ?)",
            (hash_key(key), namespace, label, datetime.now(tz=timezone.utc).isoformat()),
        )
        self._conn.commit()
        return key

    def validate(self, key: str) -> Optional[str]:
        """Return the namespace if the key is valid and not revoked, else None."""
        row = self._conn.execute(
            "SELECT namespace, revoked FROM api_keys WHERE key_hash = ?",
            (hash_key(key),),
        ).fetchone()
        if not row or row["revoked"]:
            return None
        # touch last_used + increment counter
        self._conn.execute(
            "UPDATE api_keys SET last_used = ?, call_count = call_count + 1 WHERE key_hash = ?",
            (datetime.now(tz=timezone.utc).isoformat(), hash_key(key)),
        )
        self._conn.commit()
        return row["namespace"]

    def revoke(self, key_hash: str) -> bool:
        cursor = self._conn.execute(
            "UPDATE api_keys SET revoked = 1 WHERE key_hash = ?", (key_hash,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def list_keys(self, namespace: Optional[str] = None) -> list[dict]:
        if namespace:
            rows = self._conn.execute(
                "SELECT key_hash, namespace, label, created_at, last_used, call_count, revoked "
                "FROM api_keys WHERE namespace = ?", (namespace,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT key_hash, namespace, label, created_at, last_used, call_count, revoked "
                "FROM api_keys"
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
