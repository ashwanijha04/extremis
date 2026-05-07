-- SQLite schema for the local store.
-- Mirrors 001_initial.sql with SQLite-compatible types.
-- Embeddings stored as BLOB (raw float32 bytes); cosine search done in Python.

CREATE TABLE IF NOT EXISTS memories (
    id                  TEXT PRIMARY KEY,          -- UUID as string
    namespace           TEXT NOT NULL DEFAULT 'default',
    layer               TEXT NOT NULL
                            CHECK (layer IN ('episodic', 'semantic', 'procedural', 'identity', 'working')),
    content             TEXT NOT NULL,
    embedding           BLOB,                      -- float32 bytes, numpy frombuffer
    score               REAL NOT NULL DEFAULT 0.0,
    confidence          REAL NOT NULL DEFAULT 0.5
                            CHECK (confidence >= 0.0 AND confidence <= 1.0),
    metadata            TEXT NOT NULL DEFAULT '{}', -- JSON string
    source_memory_ids   TEXT NOT NULL DEFAULT '[]', -- JSON array of UUID strings

    validity_start      TEXT NOT NULL,             -- ISO8601
    validity_end        TEXT,

    created_at          TEXT NOT NULL,
    last_accessed_at    TEXT,
    access_count        INTEGER NOT NULL DEFAULT 0,

    do_not_consolidate  INTEGER NOT NULL DEFAULT 0  -- 0 = false, 1 = true
);

CREATE INDEX IF NOT EXISTS idx_memories_namespace
    ON memories(namespace);

CREATE INDEX IF NOT EXISTS idx_memories_layer
    ON memories(layer);

CREATE INDEX IF NOT EXISTS idx_memories_score
    ON memories(score DESC)
    WHERE score > 0;

CREATE INDEX IF NOT EXISTS idx_memories_validity_current
    ON memories(validity_start)
    WHERE validity_end IS NULL;

-- Knowledge graph tables
CREATE TABLE IF NOT EXISTS kg_entities (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL DEFAULT 'default',
    name      TEXT NOT NULL,
    type      TEXT NOT NULL,
    metadata  TEXT NOT NULL DEFAULT '{}',
    UNIQUE(namespace, name)
);

CREATE TABLE IF NOT EXISTS kg_relationships (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace   TEXT NOT NULL DEFAULT 'default',
    from_entity TEXT NOT NULL,
    to_entity   TEXT NOT NULL,
    rel_type    TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0,
    metadata    TEXT NOT NULL DEFAULT '{}',
    UNIQUE(namespace, from_entity, to_entity, rel_type)
);

CREATE TABLE IF NOT EXISTS kg_attributes (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL DEFAULT 'default',
    entity    TEXT NOT NULL,
    key       TEXT NOT NULL,
    value     TEXT NOT NULL,
    UNIQUE(namespace, entity, key)
);

-- Consolidator checkpoint: one row, updated in-place
CREATE TABLE IF NOT EXISTS consolidation_checkpoints (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    log_offset          TEXT NOT NULL,
    completed_at        TEXT NOT NULL,
    memories_created    INTEGER NOT NULL DEFAULT 0,
    memories_superseded INTEGER NOT NULL DEFAULT 0,
    notes               TEXT
);

-- Identity updates proposed by the consolidator
CREATE TABLE IF NOT EXISTS pending_identity_updates (
    id                  TEXT PRIMARY KEY,          -- UUID
    proposed_content    TEXT NOT NULL,
    reasoning           TEXT,
    source_memory_ids   TEXT NOT NULL DEFAULT '[]',
    created_at          TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'accepted', 'rejected'))
);
