-- Postgres / pgvector schema
-- Run with: psql $DATABASE_URL -f 001_initial.sql

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS memories (
    id                  UUID PRIMARY KEY,
    layer               TEXT NOT NULL
                            CHECK (layer IN ('episodic', 'semantic', 'procedural', 'identity')),
    content             TEXT NOT NULL,
    embedding           vector(384),
    score               REAL NOT NULL DEFAULT 0.0,
    confidence          REAL NOT NULL DEFAULT 0.5
                            CHECK (confidence >= 0 AND confidence <= 1),
    metadata            JSONB NOT NULL DEFAULT '{}',
    source_memory_ids   UUID[] NOT NULL DEFAULT '{}',

    validity_start      TIMESTAMPTZ NOT NULL,
    validity_end        TIMESTAMPTZ,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed_at    TIMESTAMPTZ,
    access_count        INTEGER NOT NULL DEFAULT 0,

    do_not_consolidate  BOOLEAN NOT NULL DEFAULT FALSE,

    CONSTRAINT valid_window CHECK (
        validity_end IS NULL OR validity_end >= validity_start
    )
);

CREATE INDEX IF NOT EXISTS idx_memories_layer
    ON memories(layer);

CREATE INDEX IF NOT EXISTS idx_memories_validity_current
    ON memories(validity_start, validity_end)
    WHERE validity_end IS NULL;

CREATE INDEX IF NOT EXISTS idx_memories_score
    ON memories(score DESC)
    WHERE score > 0;

CREATE INDEX IF NOT EXISTS idx_memories_embedding
    ON memories USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_memories_content_trgm
    ON memories USING gin (content gin_trgm_ops);

-- Tracks where the consolidator left off; one row per completed pass
CREATE TABLE IF NOT EXISTS consolidation_checkpoints (
    id                  SERIAL PRIMARY KEY,
    log_offset          TEXT NOT NULL,
    completed_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    memories_created    INTEGER NOT NULL DEFAULT 0,
    memories_superseded INTEGER NOT NULL DEFAULT 0,
    notes               TEXT
);

-- Identity updates proposed by the consolidator; never auto-committed
CREATE TABLE IF NOT EXISTS pending_identity_updates (
    id                  UUID PRIMARY KEY,
    proposed_content    TEXT NOT NULL,
    reasoning           TEXT,
    source_memory_ids   UUID[] NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status              TEXT NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'accepted', 'rejected'))
);
