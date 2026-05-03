from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRIDAY_", env_file=".env")

    # Storage backend: "sqlite" (default, local) or "postgres"
    store: str = "sqlite"

    # Namespace — isolates one user/agent's memories from another.
    # Set FRIDAY_NAMESPACE=<user_id> to get per-user isolation on a shared server.
    # Defaults to "default" (single-user / single-agent use).
    namespace: str = "default"

    # Storage paths (sqlite)
    friday_home: str = "~/.friday"
    log_dir: str = ""          # defaults to {friday_home}/log at runtime
    local_db_path: str = ""    # defaults to {friday_home}/local.db at runtime

    # Postgres — used when store = "postgres"
    postgres_url: str = ""

    # Attention scorer thresholds
    attention_full_threshold: int = 75
    attention_standard_threshold: int = 50
    attention_minimal_threshold: int = 25

    # Embeddings
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Consolidation
    consolidation_idle_minutes: int = 30
    consolidation_daily_hour: int = 4
    consolidation_model: str = "claude-haiku-4-5-20251001"
    consolidation_hard_model: str = "claude-sonnet-4-6"

    # Retrieval ranking
    rl_alpha: float = 0.5          # weight for utility score vs cosine similarity
    recency_half_life_days: int = 90

    def resolved_log_dir(self) -> str:
        return self.log_dir or f"{self.friday_home}/log"

    def resolved_local_db_path(self) -> str:
        return self.local_db_path or f"{self.friday_home}/local.db"
