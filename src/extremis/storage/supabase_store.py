"""
Supabase memory store — thin wrapper over PostgresMemoryStore
that reads from Supabase's standard environment variables.

Usage:
    EXTREMIS_STORE=supabase
    SUPABASE_DB_URL=postgresql://postgres.[ID]:[PASS]@aws-0-...pooler.supabase.com:5432/postgres

Or pass the Supabase connection string directly:
    EXTREMIS_STORE=supabase
    EXTREMIS_POSTGRES_URL=postgresql://...

Install: pip install "extremis[postgres]"
"""

from __future__ import annotations

import os

from ..config import Config
from .postgres import PostgresMemoryStore


def _resolve_supabase_url(config: Config) -> str:
    """
    Resolve Postgres URL from Supabase env vars or explicit config.
    Priority:
      1. EXTREMIS_POSTGRES_URL (explicit)
      2. SUPABASE_DB_URL
      3. DATABASE_URL (Supabase also sets this in some environments)
    """
    if config.postgres_url:
        return config.postgres_url
    for var in ("SUPABASE_DB_URL", "DATABASE_URL"):
        url = os.environ.get(var, "")
        if url and url.startswith("postgresql"):
            return url
    raise ValueError(
        "Supabase store requires a connection URL. Set one of:\n"
        "  EXTREMIS_POSTGRES_URL=postgresql://...\n"
        "  SUPABASE_DB_URL=postgresql://...\n"
        "  DATABASE_URL=postgresql://..."
    )


class SupabaseMemoryStore(PostgresMemoryStore):
    """
    Memory store backed by Supabase (Postgres + pgvector).
    Reads connection string from Supabase environment variables automatically.

    Your Supabase project already has pgvector — no manual setup needed.
    """

    def __init__(self, config: Config) -> None:
        url = _resolve_supabase_url(config)
        super().__init__(url, config)
