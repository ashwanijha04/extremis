# Storage backends

All backends share the same API. Swap with one environment variable.

## Choosing a backend

| Backend | Best for | Infra needed | Vector search |
|---------|---------|-------------|--------------|
| **SQLite** (default) | Local dev, single machine | None | cosine in numpy |
| **Postgres + pgvector** | Production, multi-device | Postgres ≥ 15 | HNSW in SQL |
| **Chroma** | Local teams, offline | None | HNSW in Chroma |
| **Pinecone** | Scale, serverless | Pinecone account | HNSW in Pinecone |
| **Amazon S3 Vectors** | Cold/archival at extreme scale | AWS account | Managed ANN in S3 |

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

# Amazon S3 Vectors (boto3 credentials chain)
EXTREMIS_STORE=s3_vectors
EXTREMIS_S3_VECTORS_BUCKET=extremis-vectors
EXTREMIS_S3_VECTORS_INDEX=extremis
EXTREMIS_S3_VECTORS_REGION=us-east-1
```

## Migrating between backends

```bash
pip3.11 install "extremis[postgres,chroma,pinecone,s3-vectors]"

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

# Tier down to S3 Vectors for cheap, durable archival
extremis-migrate --from pinecone --to s3_vectors \
  --source-pinecone-api-key pk_... --source-pinecone-index my-index \
  --dest-s3-vectors-bucket extremis-vectors \
  --dest-s3-vectors-index extremis --dest-s3-vectors-region us-east-1

# Dry run
extremis-migrate --from sqlite --to chroma --dry-run
```

See individual backend pages:

- [SQLite](sqlite.md)
- [Postgres + pgvector](postgres.md)
- [Chroma](chroma.md)
- [Pinecone](pinecone.md)
- [Amazon S3 Vectors](s3-vectors.md)
- [Migrating](migration.md)
