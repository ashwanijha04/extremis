# Migrating backends

Move all memories between backends in one command. extremis re-embeds automatically if you switch embedding models.

## Basic migration

```bash
pip3.11 install "extremis[postgres,chroma,pinecone,s3-vectors]"

extremis-migrate --from <source> --to <dest>
```

Supported backends: `sqlite`, `postgres`, `chroma`, `pinecone`, `s3_vectors`

## Examples

```bash
# Escape Pinecone → local SQLite
extremis-migrate --from pinecone --to sqlite \
  --source-pinecone-api-key pk_... \
  --source-pinecone-index my-index

# Local SQLite → Postgres (upgrade to production)
extremis-migrate --from sqlite --to postgres \
  --dest-postgres-url postgresql://user:pass@host/extremis

# Switch embedding models at the same time
extremis-migrate --from sqlite --to chroma \
  --dest-embedder text-embedding-3-small

# Tier down to Amazon S3 Vectors (cheap, durable archival)
extremis-migrate --from pinecone --to s3_vectors \
  --source-pinecone-api-key pk_... --source-pinecone-index my-index \
  --dest-s3-vectors-bucket extremis-vectors \
  --dest-s3-vectors-index extremis --dest-s3-vectors-region us-east-1

# Dry run — count without writing
extremis-migrate --from pinecone --to sqlite \
  --source-pinecone-api-key pk_... \
  --dry-run
```

## Re-embedding

If source and destination use different embedding models, pass `--dest-embedder`:

```bash
# Migrating from all-MiniLM-L6-v2 (384d) to text-embedding-3-small (1536d)
extremis-migrate --from sqlite --to postgres \
  --dest-postgres-url postgresql://... \
  --dest-embedder text-embedding-3-small
```

extremis detects the model difference and re-embeds every memory. The `OPENAI_API_KEY` env var must be set for OpenAI embedders.

## What gets migrated

- All active memories (validity_end IS NULL)
- RL scores
- Metadata, layers, confidence values

What's **not** migrated:
- Conversation logs (JSONL files) — copy manually if needed
- Knowledge graph — will be added in a future version
