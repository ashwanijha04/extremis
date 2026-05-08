# SQLite (default)

Local storage with no infrastructure. Default backend — nothing to configure.

## Usage

```python
from extremis import Extremis

mem = Extremis()  # uses ~/.extremis/local.db
```

## Storage location

```
~/.extremis/
├── local.db         # memories, knowledge graph, RL scores
└── log/
    └── default/
        └── 2026-05-08.jsonl
```

Change the base directory:

```bash
EXTREMIS_FRIDAY_HOME=/data/my-memories
```

Or custom DB path:

```bash
EXTREMIS_LOCAL_DB_PATH=/path/to/custom.db
```

## Vector search

SQLite doesn't have a native vector type. extremis loads all embeddings into memory and computes cosine similarity with numpy. This is fine up to ~100K memories. For larger stores, switch to Postgres.

## WAL mode

The SQLite connection runs in WAL (Write-Ahead Logging) mode, which allows concurrent reads without blocking writes — safe for multi-threaded servers.

## Backup

The `local.db` file is a standard SQLite database. Copy it to back up all memories:

```bash
cp ~/.extremis/local.db ~/backup/extremis-$(date +%Y%m%d).db
```

## Limitations

- Single machine only (no multi-device sync)
- Linear scan for vector search (slow at >100K memories)
- No replication

For production or multi-device use, switch to [Postgres](postgres.md).
