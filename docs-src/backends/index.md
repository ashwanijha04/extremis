# Storage backends

All backends share the same API. Swap with one environment variable.

## Choosing a backend

| Backend | Best for | Infra needed | Vector search |
|---------|---------|-------------|--------------|
| **SQLite** (default) | Local dev, single machine | None | cosine in numpy |
| **Postgres + pgvector** | Production, multi-device | Postgres ≥ 15 | HNSW in SQL |
| **Chroma** | Local teams, offline | None | HNSW in Chroma |
| **Pinecone** | Scale, serverless | Pinecone account | HNSW in Pinecone |

## Switching backends

```bash
# SQLite (default)
EXTREMIS_STORE=sqlite

# Postgres
EXTREMIS_STORE=postgres
EXTREMIS_POSTGRES_URL=postgresql://user:pass@host/extremis

# Chroma
EXTREMIS_STORE=chroma
EXTREMIS_CHROMA_PATH=~/.extremis/chroma

# Pinecone
EXTREMIS_STORE=pinecone
EXTREMIS_PINECONE_API_KEY=pk_...
EXTREMIS_PINECONE_INDEX=extremis
```

## Migrating between backends

```bash
pip3.11 install "extremis[postgres,chroma,pinecone]"

# Escape Pinecone → local SQLite
extremis-migrate --from pinecone --to sqlite \
  --source-pinecone-api-key pk_... \
  --source-pinecone-index my-index

# SQLite → Postgres (upgrade to production)
extremis-migrate --from sqlite --to postgres \
  --dest-postgres-url postgresql://...

# Switch embedders at the same time
extremis-migrate --from sqlite --to chroma \
  --dest-embedder text-embedding-3-small

# Dry run
extremis-migrate --from sqlite --to chroma --dry-run
```

See individual backend pages:

- [SQLite](sqlite.md)
- [Postgres + pgvector](postgres.md)
- [Chroma](chroma.md)
- [Pinecone](pinecone.md)
- [Migrating](migration.md)
