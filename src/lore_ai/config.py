from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LORE_", env_file=".env")

    # ── Backend ─────────────────────────────────────────────────────
    # "sqlite" | "postgres" | "chroma" | "pinecone"
    store: str = "sqlite"

    # ── Namespace ────────────────────────────────────────────────────
    # Isolates one user/agent's memories from another.
    # LORE_NAMESPACE=user_123 for per-user isolation on a shared server.
    namespace: str = "default"

    # ── SQLite ───────────────────────────────────────────────────────
    friday_home: str = "~/.lore"
    log_dir: str = ""           # defaults to {friday_home}/log
    local_db_path: str = ""     # defaults to {friday_home}/local.db

    # ── Postgres ─────────────────────────────────────────────────────
    postgres_url: str = ""

    # ── Chroma ───────────────────────────────────────────────────────
    chroma_path: str = ""       # defaults to {friday_home}/chroma

    # ── Pinecone ─────────────────────────────────────────────────────
    pinecone_api_key: str = ""
    pinecone_index: str = "lore-ai"
    pinecone_score_db: str = "" # defaults to {friday_home}/pinecone_scores.db

    # ── Embeddings ───────────────────────────────────────────────────
    # sentence-transformers model name  OR  OpenAI model name
    # e.g. "all-MiniLM-L6-v2" or "text-embedding-3-small"
    embedder: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    openai_api_key: str = ""    # used when embedder = "text-embedding-*"

    # ── Consolidation ────────────────────────────────────────────────
    consolidation_idle_minutes: int = 30
    consolidation_daily_hour: int = 4
    consolidation_model: str = "claude-haiku-4-5-20251001"
    consolidation_hard_model: str = "claude-sonnet-4-6"

    # ── Retrieval ranking ────────────────────────────────────────────
    rl_alpha: float = 0.5
    recency_half_life_days: int = 90

    # ── Attention scorer ─────────────────────────────────────────────
    attention_full_threshold: int = 75
    attention_standard_threshold: int = 50
    attention_minimal_threshold: int = 25

    # ── Resolved paths ───────────────────────────────────────────────
    def resolved_log_dir(self) -> str:
        return self.log_dir or f"{self.friday_home}/log"

    def resolved_local_db_path(self) -> str:
        return self.local_db_path or f"{self.friday_home}/local.db"

    def resolved_chroma_path(self) -> str:
        return self.chroma_path or f"{self.friday_home}/chroma"

    def resolved_pinecone_score_db(self) -> str:
        return self.pinecone_score_db or f"{self.friday_home}/pinecone_scores.db"
