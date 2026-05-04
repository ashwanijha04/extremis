"""Shared fixtures for all test modules."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from friday_memory.api import FridayMemory
from friday_memory.config import Config
from friday_memory.storage.kg import SQLiteKGStore
from friday_memory.storage.log import FileLogStore
from friday_memory.storage.sqlite import SQLiteMemoryStore
from friday_memory.types import Memory, MemoryLayer


# ------------------------------------------------------------------ #
# Config + storage fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def tmp_config(tmp_path):
    return Config(
        friday_home=str(tmp_path),
        log_dir=str(tmp_path / "log"),
        local_db_path=str(tmp_path / "local.db"),
        namespace="test_ns",
    )


@pytest.fixture
def mock_embedder():
    e = MagicMock()
    e.embed.return_value = [0.1] * 384
    e.embed_batch.return_value = [[0.1] * 384]
    e.dim = 384
    return e


@pytest.fixture
def log_store(tmp_config):
    return FileLogStore(tmp_config.resolved_log_dir(), namespace=tmp_config.namespace)


@pytest.fixture
def mem_store(tmp_config):
    return SQLiteMemoryStore(tmp_config.resolved_local_db_path(), tmp_config)


@pytest.fixture
def kg_store(tmp_config):
    return SQLiteKGStore(tmp_config.resolved_local_db_path(), tmp_config)


@pytest.fixture
def api(tmp_config, mock_embedder):
    return FridayMemory(config=tmp_config, embedder=mock_embedder)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def make_memory(
    content: str,
    layer: MemoryLayer = MemoryLayer.EPISODIC,
    embedding: list[float] | None = None,
    score: float = 0.0,
    namespace: str = "test_ns",
) -> Memory:
    return Memory(
        namespace=namespace,
        layer=layer,
        content=content,
        embedding=embedding or [0.1] * 384,
        score=score,
        validity_start=datetime.now(tz=timezone.utc),
    )
