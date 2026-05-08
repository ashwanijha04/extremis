# Postgres + pgvector

Production-scale memory with HNSW indexing. Ranking runs entirely in SQL — no Python loop over rows.

## Install

```bash
pip3.11 install "extremis[postgres]"
```

## Configure

```bash
EXTREMIS_STORE=postgres
EXTREMIS_POSTGRES_URL=postgresql://user:pass@host:5432/extremis
```

Or in Python:

```python
from extremis import Extremis, Config

mem = Extremis(config=Config(
    store="postgres",
    postgres_url="postgresql://user:pass@host/extremis",
))
```

## Postgres requirements

- Postgres 15+
- pgvector extension

The schema and extension are created automatically on first start. No manual SQL needed.

## Free hosted options

| Provider | Free tier | pgvector | Setup |
|---------|-----------|---------|-------|
| **Supabase** | 500 MB | ✅ Built-in | [supabase.com](https://supabase.com) |
| **Neon** | 0.5 GB | ✅ Built-in | [neon.tech](https://neon.tech) |
| **Render** | Included with server deploy | ✅ Built-in | [render.com](https://render.com) |

### Supabase quickstart

1. Create a project at [supabase.com](https://supabase.com)
2. Go to **Settings** → **Database** → copy the connection string
3. In the SQL editor, run: `CREATE EXTENSION IF NOT EXISTS vector;`
4. Set `EXTREMIS_POSTGRES_URL=postgresql://...`

## How vectors are stored

Embeddings are stored as `vector(384)` — a pgvector type. The ranking query runs entirely in Postgres:

```sql
SELECT *,
  1 - (embedding <=> %(vec)s) AS relevance,
  (1 - (embedding <=> %(vec)s))
    * (1 + 0.5 * tanh(score))
    * exp(-0.693 * EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400 / 90)
  AS final_rank
FROM memories
WHERE validity_end IS NULL
ORDER BY final_rank DESC
LIMIT 10
```

The HNSW index (`CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)`) accelerates the `<=>` operator for large memory stores.

## Key management (hosted server)

When running `extremis-server` with Postgres, API keys are stored in the same Postgres database — they persist across restarts and redeploys.

```bash
EXTREMIS_STORE=postgres extremis-server create-key --namespace alice
```
