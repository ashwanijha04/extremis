from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXTREMIS_", env_file=".env")

    # ── Backend ─────────────────────────────────────────────────────
    # "sqlite" | "postgres" | "chroma" | "pinecone" | "s3_vectors" | "supabase"
    store: str = "sqlite"

    # ── Namespace ────────────────────────────────────────────────────
    # Isolates one user/agent's memories from another.
    # EXTREMIS_NAMESPACE=user_123 for per-user isolation on a shared server.
    namespace: str = "default"

    # ── SQLite ───────────────────────────────────────────────────────
    extremis_home: str = "~/.extremis"
    log_dir: str = ""  # defaults to {extremis_home}/log
    local_db_path: str = ""  # defaults to {extremis_home}/local.db

    # ── Postgres ─────────────────────────────────────────────────────
    postgres_url: str = ""

    # ── Chroma ───────────────────────────────────────────────────────
    chroma_path: str = ""  # defaults to {extremis_home}/chroma

    # ── Pinecone ─────────────────────────────────────────────────────
    pinecone_api_key: str = ""
    pinecone_index: str = "extremis"
    pinecone_score_db: str = ""  # defaults to {extremis_home}/pinecone_scores.db

    # ── Amazon S3 Vectors ───────────────────────────────────────────
    # Credentials come from the standard boto3 chain (env vars, ~/.aws,
    # IAM role). Only the bucket/index/region need explicit config.
    s3_vectors_bucket: str = ""
    s3_vectors_index: str = "extremis"
    s3_vectors_region: str = ""  # falls back to AWS_REGION / boto3 default
    s3_vectors_score_db: str = ""  # defaults to {extremis_home}/s3_vectors_scores.db

    # ── Embeddings ───────────────────────────────────────────────────
    # sentence-transformers model name  OR  OpenAI model name
    # e.g. "all-MiniLM-L6-v2" or "text-embedding-3-small"
    embedder: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    openai_api_key: str = ""  # used when embedder = "text-embedding-*"

    # ── Consolidation ────────────────────────────────────────────────
    consolidation_idle_minutes: int = 30
    consolidation_daily_hour: int = 4
    consolidation_model: str = "claude-haiku-4-5-20251001"
    consolidation_hard_model: str = "claude-sonnet-4-6"

    # Counter-based auto-consolidation — off by default.
    # Fires every N remember() calls regardless of session boundaries.
    # Cost warning: each run calls the LLM on batches of 30 log entries.
    # A session with 500 turns will trigger ~10 consolidations at the default
    # threshold of 50 — multiplying your LLM cost by ~10x. Use
    # consolidate_on_session_end instead for predictable, session-scoped cost.
    auto_consolidate: bool = False
    auto_consolidate_every: int = 50

    # Signal-based consolidation — fires once when conversation_id changes.
    # One LLM consolidation pass per completed session. Predictable cost.
    # Requires ANTHROPIC_API_KEY. Off by default — enable in production agents.
    consolidate_on_session_end: bool = False

    # ── Chunking ─────────────────────────────────────────────────────
    chunk_size: int = 200  # max tokens per memory chunk (0 = disabled)

    # ── Observability (peekr) ─────────────────────────────────────────
    observe: bool = False  # set EXTREMIS_OBSERVE=true to enable
    traces_path: str = ""  # defaults to {extremis_home}/traces.jsonl

    # ── Retrieval ranking ────────────────────────────────────────────
    rl_alpha: float = 0.5
    recency_half_life_days: int = 90
    recall_min_relevance: float = 0.05  # drop results below this cosine similarity
    dedup_similarity_threshold: float = 0.92  # above this → supersede old instead of accumulating

    # ── Attention scorer ─────────────────────────────────────────────
    attention_full_threshold: int = 75
    attention_standard_threshold: int = 50
    attention_minimal_threshold: int = 25

    # ── Hallucination detection ──────────────────────────────────────
    # Tiered write-time check: extracted memories are verified against the
    # source conversation. NLI runs first; if score lands in the grey zone
    # the LLM judge is invoked. Failing memories are tagged + downranked,
    # never silently dropped — compaction can revisit them later.
    enable_faithfulness_check: bool = True
    faithfulness_nli_model: str = "cross-encoder/nli-deberta-v3-small"
    faithfulness_pass_threshold: float = 0.85  # ≥ this → store as-is
    faithfulness_grey_zone_low: float = 0.5  # below this → skip judge, mark unverified

    # Self-consistency: re-sample extraction N times for high-stakes layers
    # and keep only claims that converge in embedding space.
    # Set self_consistency_n=0 to disable entirely.
    self_consistency_n: int = 3
    self_consistency_temperature: float = 0.7
    consistency_threshold: float = 0.85
    # Layers that get self-consistency. Comma-separated env var: identity,semantic
    self_consistency_layers: str = "identity,semantic"

    # Half-life (days) for confidence temporal decay at recall time.
    # effective_confidence = confidence × layer_weight × 2^(-age_days/half_life)
    confidence_half_life_days: int = 180

    # ── Resolved paths ───────────────────────────────────────────────
    def resolved_log_dir(self) -> str:
        return self.log_dir or f"{self.extremis_home}/log"

    def resolved_local_db_path(self) -> str:
        return self.local_db_path or f"{self.extremis_home}/local.db"

    def resolved_chroma_path(self) -> str:
        return self.chroma_path or f"{self.extremis_home}/chroma"

    def resolved_pinecone_score_db(self) -> str:
        return self.pinecone_score_db or f"{self.extremis_home}/pinecone_scores.db"

    def resolved_s3_vectors_score_db(self) -> str:
        return self.s3_vectors_score_db or f"{self.extremis_home}/s3_vectors_scores.db"

    def resolved_traces_path(self) -> str:
        return self.traces_path or f"{self.extremis_home}/traces.jsonl"
