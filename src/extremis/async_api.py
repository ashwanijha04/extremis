"""AsyncExtremis — async drop-in for Extremis using a thread-pool executor."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional
from uuid import UUID

from .api import Extremis
from .config import Config
from .interfaces import Embedder, LogStore, MemoryStore
from .types import (
    AttentionResult,
    CompactionResult,
    EntityResult,
    EntityType,
    Memory,
    MemoryLayer,
    Observation,
    RecallResult,
)


class AsyncExtremis:
    """
    Async wrapper around Extremis. Embedding and store calls run in a thread-pool
    executor so they don't block the event loop.

    Drop-in for async apps:

        mem = AsyncExtremis()

        async def handler(message: str):
            await mem.remember(message, conversation_id="session_1")
            results = await mem.recall("what is the user building?")
            return results
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        log: Optional[LogStore] = None,
        local: Optional[MemoryStore] = None,
        embedder: Optional[Embedder] = None,
        executor=None,
    ) -> None:
        self._sync = Extremis(config=config, log=log, local=local, embedder=embedder)
        self._executor = executor  # None → default ThreadPoolExecutor

    async def _run(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))

    async def remember(
        self,
        content: str,
        role: str = "user",
        conversation_id: str = "default",
        metadata: Optional[dict] = None,
    ) -> None:
        await self._run(self._sync.remember, content, role, conversation_id, metadata)

    async def recall(
        self,
        query: str,
        limit: int = 10,
        layers: Optional[list[MemoryLayer]] = None,
        min_score: Optional[float] = None,
    ) -> list[RecallResult]:
        return await self._run(self._sync.recall, query, limit, layers, min_score)

    async def report_outcome(
        self,
        memory_ids: list[UUID],
        success: bool,
        weight: float = 1.0,
    ) -> None:
        await self._run(self._sync.report_outcome, memory_ids, success, weight)

    async def remember_now(
        self,
        content: str,
        layer: MemoryLayer,
        expires_at: Optional[datetime] = None,
        confidence: float = 0.9,
        metadata: Optional[dict] = None,
    ) -> Memory:
        return await self._run(self._sync.remember_now, content, layer, expires_at, confidence, metadata)

    async def compact(self, layer: MemoryLayer = MemoryLayer.SEMANTIC) -> CompactionResult:
        return await self._run(self._sync.compact, layer)

    async def observe(self, conversation_id: str = "default") -> list[Observation]:
        return await self._run(self._sync.observe, conversation_id)

    async def score_attention(
        self,
        message: str,
        sender: str = "",
        channel: str = "dm",
        owner_ids: Optional[set[str]] = None,
        allowlist: Optional[set[str]] = None,
        context: Optional[dict] = None,
    ) -> AttentionResult:
        return await self._run(self._sync.score_attention, message, sender, channel, owner_ids, allowlist, context)

    async def kg_add_entity(self, name: str, type: EntityType, metadata: Optional[dict] = None):
        return await self._run(self._sync.kg_add_entity, name, type, metadata)

    async def kg_add_relationship(
        self,
        from_entity: str,
        to_entity: str,
        rel_type: str,
        weight: float = 1.0,
        metadata: Optional[dict] = None,
    ):
        return await self._run(self._sync.kg_add_relationship, from_entity, to_entity, rel_type, weight, metadata)

    async def kg_add_attribute(self, entity: str, key: str, value: str):
        return await self._run(self._sync.kg_add_attribute, entity, key, value)

    async def kg_query(self, name: str) -> Optional[EntityResult]:
        return await self._run(self._sync.kg_query, name)

    async def kg_traverse(self, name: str, depth: int = 2) -> list[EntityResult]:
        return await self._run(self._sync.kg_traverse, name, depth)

    def get_local_store(self) -> MemoryStore:
        return self._sync.get_local_store()

    def get_log(self) -> LogStore:
        return self._sync.get_log()
