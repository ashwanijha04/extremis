# Config

All settings via environment variables (prefix `EXTREMIS_`) or a `.env` file.

```python
from extremis import Config

mem = Extremis(config=Config(
    store="postgres",
    postgres_url="postgresql://...",
    namespace="alice",
))
```

## Storage

| Variable | Python field | Default | Description |
|----------|-------------|---------|-------------|
| `EXTREMIS_STORE` | `store` | `"sqlite"` | `sqlite` \| `postgres` \| `chroma` \| `pinecone` |
| `EXTREMIS_NAMESPACE` | `namespace` | `"default"` | User/agent isolation scope |
| `EXTREMIS_FRIDAY_HOME` | `extremis_home` | `~/.extremis` | Base directory |
| `EXTREMIS_LOG_DIR` | `log_dir` | `{home}/log` | Conversation log directory |
| `EXTREMIS_LOCAL_DB_PATH` | `local_db_path` | `{home}/local.db` | SQLite path |
| `EXTREMIS_POSTGRES_URL` | `postgres_url` | `""` | Required when `store=postgres` |
| `EXTREMIS_CHROMA_PATH` | `chroma_path` | `{home}/chroma` | ChromaDB directory |
| `EXTREMIS_PINECONE_API_KEY` | `pinecone_api_key` | `""` | Required when `store=pinecone` |
| `EXTREMIS_PINECONE_INDEX` | `pinecone_index` | `"extremis"` | Pinecone index name |

## Embeddings

| Variable | Python field | Default | Description |
|----------|-------------|---------|-------------|
| `EXTREMIS_EMBEDDER` | `embedder` | `"all-MiniLM-L6-v2"` | Model name (sentence-transformers or OpenAI) |
| `EXTREMIS_EMBEDDING_DIM` | `embedding_dim` | `384` | Vector dimension (must match model) |
| `EXTREMIS_OPENAI_API_KEY` | `openai_api_key` | `""` | Required for OpenAI embedders |

## Retrieval

| Variable | Python field | Default | Description |
|----------|-------------|---------|-------------|
| `EXTREMIS_RL_ALPHA` | `rl_alpha` | `0.5` | Utility score weight in ranking |
| `EXTREMIS_RECENCY_HALF_LIFE_DAYS` | `recency_half_life_days` | `90` | Memory decay rate |
| `EXTREMIS_RECALL_MIN_RELEVANCE` | `recall_min_relevance` | `0.05` | Minimum cosine similarity |
| `EXTREMIS_DEDUP_SIMILARITY_THRESHOLD` | `dedup_similarity_threshold` | `0.92` | Write-time dedup threshold |

## Consolidation

| Variable | Python field | Default | Description |
|----------|-------------|---------|-------------|
| `EXTREMIS_CONSOLIDATION_MODEL` | `consolidation_model` | `claude-haiku-4-5-20251001` | LLM for consolidation |
| `EXTREMIS_CONSOLIDATION_HARD_MODEL` | `consolidation_hard_model` | `claude-sonnet-4-6` | LLM for compaction |

## Attention scoring

| Variable | Python field | Default | Description |
|----------|-------------|---------|-------------|
| `EXTREMIS_ATTENTION_FULL_THRESHOLD` | `attention_full_threshold` | `75` | Score ≥ this → full |
| `EXTREMIS_ATTENTION_STANDARD_THRESHOLD` | `attention_standard_threshold` | `50` | Score ≥ this → standard |
| `EXTREMIS_ATTENTION_MINIMAL_THRESHOLD` | `attention_minimal_threshold` | `25` | Score ≥ this → minimal |
