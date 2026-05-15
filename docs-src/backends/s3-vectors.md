# Amazon S3 Vectors

Cheap, durable, large-scale vector storage on AWS. Vectors in an S3 vector bucket + index, RL scores in a local SQLite sidecar.

Best for cold/archival or extreme-scale workloads. Query latency is in the 100s of ms range — for chat-rate recall, pair it with a hot tier (Pinecone, pgvector) or stick with Pinecone as the primary.

## Install

```bash
pip3.11 install "extremis[s3-vectors]"
```

## One-time AWS setup

S3 Vectors needs a vector bucket and an index. The index declares the embedding dimension, distance metric, and which metadata keys are filterable (max 10).

```bash
aws s3vectors create-vector-bucket --vector-bucket-name extremis-vectors

aws s3vectors create-index \
    --vector-bucket-name extremis-vectors \
    --index-name extremis \
    --data-type float32 \
    --dimension 384 \
    --distance-metric cosine \
    --metadata-configuration 'nonFilterableMetadataKeys=["content","extra_metadata","source_memory_ids","confidence","created_at","validity_start","last_accessed_at","access_count","do_not_consolidate"]'
```

extremis keeps only `namespace`, `layer`, and `validity_end` filterable — everything else is non-filterable to stay under the 10-key cap.

> Dimension must match your embedder. `all-MiniLM-L6-v2` → 384, `text-embedding-3-small` → 1536.

## Configure

```bash
EXTREMIS_STORE=s3_vectors
EXTREMIS_S3_VECTORS_BUCKET=extremis-vectors
EXTREMIS_S3_VECTORS_INDEX=extremis
EXTREMIS_S3_VECTORS_REGION=us-east-1   # optional; AWS_REGION also works
```

Credentials come from the standard boto3 chain — env vars (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`), `~/.aws/credentials`, or an EC2/Lambda IAM role. No API-key flag.

## RL scores

S3 Vectors charges per metadata write and limits filterable-key cardinality, so extremis stores RL scores in a local SQLite file alongside the index:

```
~/.extremis/
└── s3_vectors_scores.db   # RL utility scores (tiny, ~KB per memory)
```

Change the path: `EXTREMIS_S3_VECTORS_SCORE_DB=/custom/path.db`

## Namespace mapping

S3 Vectors has no per-index namespace concept, so extremis uses the configured namespace as a filterable metadata key. All queries are filtered by `namespace == EXTREMIS_NAMESPACE` for tenant isolation on a shared index.

## When to choose S3 Vectors

| Use case | Pick S3 Vectors? |
|---|---|
| Cold/archival memory at scale | ✓ |
| Cheap long-term retention, infrequent recall | ✓ |
| Real-time chat (sub-100 ms recall) | ✗ — use Pinecone / pgvector |
| Tight metadata-filter cardinality | ✗ — only 10 filterable keys |

A common pattern: hot tier in Pinecone (recent + frequently recalled), tier down to S3 Vectors with `extremis-migrate` for everything older than N days.
