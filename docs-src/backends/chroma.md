# ChromaDB

Local vector database with HNSW indexing. Good for teams without cloud infrastructure.

## Install

```bash
pip3.11 install "extremis[chroma]"
```

## Configure

```bash
EXTREMIS_STORE=chroma
EXTREMIS_CHROMA_PATH=~/.extremis/chroma  # default
```

## How it differs from SQLite

| | SQLite | Chroma |
|--|--------|--------|
| Vector search | numpy cosine (O(n)) | HNSW (fast at scale) |
| Storage | Single `.db` file | Directory of files |
| Scale | ~100K memories | Millions of memories |

## Namespace isolation

Each namespace gets its own Chroma collection: `lore_ai_{namespace}`. They're stored in the same directory but completely isolated.
