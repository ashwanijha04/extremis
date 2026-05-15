"""
extremis migration tool.

Copies all memories from one backend to another.
Re-embeds if the source and destination embedders differ.

CLI:
    extremis-migrate --from sqlite --to chroma
    extremis-migrate --from sqlite --to pinecone --pinecone-api-key pk_... --pinecone-index my-index
    extremis-migrate --from chroma --to postgres --postgres-url postgresql://...
    extremis-migrate --from pinecone --to s3_vectors \\
        --source-pinecone-api-key pk_... --dest-s3-vectors-bucket extremis-vectors \\
        --dest-s3-vectors-index extremis --dest-s3-vectors-region us-east-1

Python:
    from extremis.migrate import Migrator
    result = Migrator().run(source_store, dest_store, source_embedder, dest_embedder)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class MigrationResult:
    memories_migrated: int = 0
    memories_skipped: int = 0
    re_embedded: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class Migrator:
    """
    Copies every valid memory from source to dest.

    Re-embeds if the embedder models differ (e.g. moving from
    sentence-transformers to OpenAI embeddings).
    """

    def run(
        self,
        source,  # MemoryStore
        dest,  # MemoryStore
        source_embedder=None,  # Embedder | None
        dest_embedder=None,  # Embedder | None
        batch_size: int = 100,
        dry_run: bool = False,
    ) -> MigrationResult:
        result = MigrationResult()
        start = datetime.now(tz=timezone.utc)

        reembed = (
            source_embedder is not None
            and dest_embedder is not None
            and _embedder_name(source_embedder) != _embedder_name(dest_embedder)
        )
        if reembed:
            log.info(
                "Embedding models differ (%s → %s) — will re-embed all memories",
                _embedder_name(source_embedder),
                _embedder_name(dest_embedder),
            )

        offset = 0
        while True:
            batch = source.list_recent(limit=batch_size)
            if not batch:
                break

            texts_to_embed = [m.content for m in batch] if reembed else []
            new_embeddings = dest_embedder.embed_batch(texts_to_embed) if reembed else []

            for i, memory in enumerate(batch):
                try:
                    if reembed:
                        memory = memory.model_copy(update={"embedding": new_embeddings[i]})
                        result.re_embedded += 1

                    if not dry_run:
                        dest.store(memory)
                    result.memories_migrated += 1
                except Exception as exc:
                    log.warning("Failed to migrate memory %s: %s", memory.id, exc)
                    result.errors.append(f"{memory.id}: {exc}")
                    result.memories_skipped += 1

            offset += len(batch)
            if len(batch) < batch_size:
                break

        result.duration_seconds = round((datetime.now(tz=timezone.utc) - start).total_seconds(), 2)
        log.info(
            "Migration complete: %d migrated, %d skipped, %d re-embedded in %.1fs",
            result.memories_migrated,
            result.memories_skipped,
            result.re_embedded,
            result.duration_seconds,
        )
        return result


def _embedder_name(embedder) -> str:
    if hasattr(embedder, "_model_name"):
        return embedder._model_name
    if hasattr(embedder, "_model"):
        return str(embedder._model)
    return type(embedder).__name__


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="extremis-migrate",
        description="Migrate extremis memories between storage backends",
    )
    parser.add_argument(
        "--from",
        dest="source",
        required=True,
        choices=["sqlite", "postgres", "chroma", "pinecone", "s3_vectors"],
        help="Source backend",
    )
    parser.add_argument(
        "--to",
        dest="dest",
        required=True,
        choices=["sqlite", "postgres", "chroma", "pinecone", "s3_vectors"],
        help="Destination backend",
    )
    parser.add_argument("--namespace", default="default", help="Namespace to migrate")
    parser.add_argument("--dry-run", action="store_true", help="Count memories without writing")

    # SQLite / Chroma paths
    parser.add_argument("--source-path", default="", help="Source SQLite/Chroma path")
    parser.add_argument("--dest-path", default="", help="Destination SQLite/Chroma path")

    # Postgres
    parser.add_argument("--source-postgres-url", default="", help="Source Postgres URL")
    parser.add_argument("--dest-postgres-url", default="", help="Destination Postgres URL")

    # Pinecone
    parser.add_argument("--source-pinecone-api-key", default="")
    parser.add_argument("--dest-pinecone-api-key", default="")
    parser.add_argument("--source-pinecone-index", default="extremis")
    parser.add_argument("--dest-pinecone-index", default="extremis")

    # Amazon S3 Vectors (boto3 credentials chain — no API key flag)
    parser.add_argument("--source-s3-vectors-bucket", default="")
    parser.add_argument("--dest-s3-vectors-bucket", default="")
    parser.add_argument("--source-s3-vectors-index", default="extremis")
    parser.add_argument("--dest-s3-vectors-index", default="extremis")
    parser.add_argument("--source-s3-vectors-region", default="")
    parser.add_argument("--dest-s3-vectors-region", default="")

    # Embedder override
    parser.add_argument("--dest-embedder", default="", help="Re-embed with this model (e.g. text-embedding-3-small)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    from .config import Config

    source_cfg = Config(namespace=args.namespace)
    dest_cfg = Config(namespace=args.namespace)

    source_store = _make_store(args.source, args, source_cfg, role="source")
    dest_store = _make_store(args.dest, args, dest_cfg, role="dest")

    source_embedder = None
    dest_embedder = None
    if args.dest_embedder:
        dest_embedder = _make_embedder(args.dest_embedder)

    result = Migrator().run(
        source_store,
        dest_store,
        source_embedder=source_embedder,
        dest_embedder=dest_embedder,
        dry_run=args.dry_run,
    )

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Migration complete")
    print(f"  Migrated:    {result.memories_migrated}")
    print(f"  Skipped:     {result.memories_skipped}")
    print(f"  Re-embedded: {result.re_embedded}")
    print(f"  Duration:    {result.duration_seconds}s")
    if result.errors:
        print(f"  Errors ({len(result.errors)}):")
        for e in result.errors[:5]:
            print(f"    {e}")


def _make_store(backend: str, args, config: Config, role: str):
    if backend == "sqlite":
        from .storage.sqlite import SQLiteMemoryStore

        path = getattr(args, f"{role}_path") or config.resolved_local_db_path()
        return SQLiteMemoryStore(path, config)
    elif backend == "postgres":
        from .storage.postgres import PostgresMemoryStore

        url = getattr(args, f"{role}_postgres_url") or config.postgres_url
        return PostgresMemoryStore(url, config)
    elif backend == "chroma":
        from .storage.chroma import ChromaMemoryStore

        path = getattr(args, f"{role}_path") or f"{config.extremis_home}/chroma"
        return ChromaMemoryStore(path, config)
    elif backend == "pinecone":
        from .storage.pinecone_store import PineconeMemoryStore

        api_key = getattr(args, f"{role}_pinecone_api_key")
        index = getattr(args, f"{role}_pinecone_index")
        return PineconeMemoryStore(api_key, index, config)
    elif backend == "s3_vectors":
        from .storage.s3_vectors import S3VectorsMemoryStore

        bucket = getattr(args, f"{role}_s3_vectors_bucket") or config.s3_vectors_bucket
        index = getattr(args, f"{role}_s3_vectors_index") or config.s3_vectors_index
        region = getattr(args, f"{role}_s3_vectors_region") or config.s3_vectors_region
        if not bucket:
            raise ValueError(f"--{role}-s3-vectors-bucket is required when {role} backend is s3_vectors")
        return S3VectorsMemoryStore(bucket, index, config, region=region)
    raise ValueError(f"Unknown backend: {backend}")


def _make_embedder(model: str):
    if model.startswith("text-embedding"):
        from .embeddings.openai import OpenAIEmbedder

        return OpenAIEmbedder(model)
    from .embeddings.sentence_transformers import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(model)


if __name__ == "__main__":
    cli()
