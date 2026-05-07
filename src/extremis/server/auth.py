"""
API key management for the extremis hosted server.

Keys format: extremis_sk_<32 url-safe base64 chars>
Keys are stored hashed (sha256). The plaintext is only shown once at creation.

Storage:
  - SQLite (default): {server_home}/keys.db  — ephemeral on cloud hosts
  - Postgres: same DB as memories — persists across restarts (use this on Render/Railway)

Auto-selected: KeyStore.for_config(config) returns the right implementation.
"""

from __future__ import annotations

import hashlib
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
        cursor = self._conn.execute("UPDATE api_keys SET revoked = 1 WHERE key_hash = ?", (key_hash,))
        self._conn.commit()
        return cursor.rowcount > 0

    def list_keys(self, namespace: Optional[str] = None) -> list[dict]:
        if namespace:
            rows = self._conn.execute(
                "SELECT key_hash, namespace, label, created_at, last_used, call_count, revoked "
                "FROM api_keys WHERE namespace = ?",
                (namespace,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT key_hash, namespace, label, created_at, last_used, call_count, revoked FROM api_keys"
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()


class PostgresKeyStore:
    """
    API key store backed by the same Postgres DB used for memories.
    Keys survive server restarts and redeploys — use this on Render/Railway/Fly.
    """

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS api_keys (
            key_hash   TEXT PRIMARY KEY,
            namespace  TEXT NOT NULL,
            label      TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_used  TIMESTAMPTZ,
            call_count INTEGER NOT NULL DEFAULT 0,
            revoked    BOOLEAN NOT NULL DEFAULT FALSE
        );
    """

    def __init__(self, postgres_url: str) -> None:
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise ImportError("Postgres key store requires: pip install 'extremis[postgres]'") from None

        self._conn = psycopg2.connect(postgres_url, connect_timeout=10)
        self._conn.autocommit = False
        with self._conn.cursor() as cur:
            cur.execute(self._CREATE_TABLE)
        self._conn.commit()

    def _dict_cursor(self):
        import psycopg2.extras

        return psycopg2.extras.RealDictCursor

    def create(self, namespace: str, label: str = "") -> str:
        key = generate_key()
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO api_keys (key_hash, namespace, label) VALUES (%s, %s, %s)",
                (hash_key(key), namespace, label),
            )
        self._conn.commit()
        return key

    def validate(self, key: str) -> Optional[str]:
        with self._conn.cursor(cursor_factory=self._dict_cursor()) as cur:
            cur.execute(
                "SELECT namespace, revoked FROM api_keys WHERE key_hash = %s",
                (hash_key(key),),
            )
            row = cur.fetchone()
        if not row or row["revoked"]:
            return None
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE api_keys SET last_used = NOW(), call_count = call_count + 1 WHERE key_hash = %s",
                (hash_key(key),),
            )
        self._conn.commit()
        return row["namespace"]

    def revoke(self, key_hash: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute("UPDATE api_keys SET revoked = TRUE WHERE key_hash = %s", (key_hash,))
            affected = cur.rowcount
        self._conn.commit()
        return affected > 0

    def list_keys(self, namespace: Optional[str] = None) -> list[dict]:
        with self._conn.cursor(cursor_factory=self._dict_cursor()) as cur:
            if namespace:
                cur.execute(
                    "SELECT key_hash, namespace, label, created_at::text, last_used::text, call_count, revoked "
                    "FROM api_keys WHERE namespace = %s",
                    (namespace,),
                )
            else:
                cur.execute(
                    "SELECT key_hash, namespace, label, created_at::text, last_used::text, call_count, revoked "
                    "FROM api_keys"
                )
            return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()


def make_key_store(config) -> "KeyStore | PostgresKeyStore":
    """
    Return the right KeyStore for the current config.
    Uses Postgres when store=postgres so keys survive restarts.
    Falls back to SQLite otherwise.
    """
    import os
    from pathlib import Path

    if config.store == "postgres" and config.postgres_url:
        return PostgresKeyStore(config.postgres_url)

    server_home = os.environ.get("EXTREMIS_SERVER_HOME", "~/.extremis/server")
    db_path = str(Path(server_home).expanduser() / "keys.db")
    return KeyStore(db_path)
