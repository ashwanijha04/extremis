# Pinecone

Serverless hosted vector store. Vectors in Pinecone, RL scores in a local SQLite sidecar.

## Install

```bash
pip3.11 install "extremis[pinecone]"
```

## Create an index

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="pk_...")
pc.create_index(
    "extremis",
    dimension=384,    # match EXTREMIS_EMBEDDING_DIM
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

## Configure

```bash
EXTREMIS_STORE=pinecone
EXTREMIS_PINECONE_API_KEY=pk_...
EXTREMIS_PINECONE_INDEX=extremis
```

## RL scores

Pinecone doesn't support partial metadata updates efficiently. extremis stores RL scores in a local SQLite file alongside the Pinecone index:

```
~/.extremis/
└── pinecone_scores.db   # RL utility scores (tiny, ~KB per memory)
```

Change the path: `EXTREMIS_PINECONE_SCORE_DB=/custom/path.db`

## Namespace mapping

Pinecone namespaces map 1:1 to extremis namespaces.
