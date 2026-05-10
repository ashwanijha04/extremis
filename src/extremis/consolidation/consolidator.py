from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from itertools import islice

import anthropic

try:
    from peekr.decorators import trace as _trace
except ImportError:

    def _trace(_func=None, *, name=None, capture_io=True):  # type: ignore[misc]
        def decorator(fn):
            return fn

        return decorator(_func) if _func is not None else decorator


from ..config import Config
from ..interfaces import Embedder, LogStore, MemoryStore
from ..types import ConsolidationResult, LogEntry, Memory, MemoryLayer
from .prompts import EXTRACTION_SYSTEM, EXTRACTION_USER_TEMPLATE

log = logging.getLogger(__name__)

_BATCH_SIZE = 30  # max log entries per LLM call


def _batched(iterable, n: int):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


class LLMConsolidator:
    """
    Reads log entries since the last checkpoint, calls Claude to extract
    semantic and procedural memories, and writes them to the memory store.

    Identity-layer proposals are written to pending_identity_updates and
    never committed automatically — they require human review.
    """

    def __init__(self, config: Config, embedder: Embedder) -> None:
        self._config = config
        self._embedder = embedder
        self._client = anthropic.Anthropic()

    def run_pass(
        self,
        log_store: LogStore,
        local: MemoryStore,
        consolidated: MemoryStore,
    ) -> ConsolidationResult:
        checkpoint = log_store.get_checkpoint()
        entries = log_store.read_since(checkpoint)

        if not entries:
            log.info("Consolidation: no new log entries since checkpoint %s", checkpoint)
            return ConsolidationResult(log_checkpoint=checkpoint or "")

        log.info("Consolidation: processing %d log entries", len(entries))
        start = datetime.now(tz=timezone.utc)
        result = ConsolidationResult()

        conversations = self._group_by_conversation(entries)

        for conv_id, conv_entries in conversations.items():
            if len(conv_entries) < 2:
                continue  # too little context to be worth extracting

            for batch in _batched(conv_entries, _BATCH_SIZE):
                try:
                    extracted = self._extract(conv_id, batch)
                except Exception as exc:
                    log.warning("Consolidation extraction failed for %s: %s", conv_id, exc)
                    continue

                for item in extracted:
                    layer_str = item.get("layer", "semantic")
                    content = item.get("content", "").strip()
                    confidence = float(item.get("confidence", 0.7))

                    if not content:
                        continue

                    try:
                        layer = MemoryLayer(layer_str)
                    except ValueError:
                        log.debug("Unknown layer %r — skipping", layer_str)
                        continue

                    try:
                        embedding = self._embedder.embed(content)
                    except Exception as exc:
                        log.warning("Embedding failed: %s", exc)
                        continue

                    memory = Memory(
                        layer=layer,
                        content=content,
                        embedding=embedding,
                        confidence=confidence,
                        metadata={
                            "source": "consolidation",
                            "conversation_id": conv_id,
                            "model": self._config.consolidation_model,
                        },
                        validity_start=datetime.now(tz=timezone.utc),
                    )

                    # Contradiction detection: supersede conflicting existing memories
                    if layer in (MemoryLayer.SEMANTIC, MemoryLayer.PROCEDURAL):
                        superseded = self._supersede_contradictions(memory, embedding, consolidated, result)
                        if superseded:
                            continue  # supersede() already stored the new memory

                    consolidated.store(memory)
                    result.memories_created += 1

        # Advance checkpoint to the end of what we just processed
        new_checkpoint = self._end_checkpoint(log_store)
        log_store.set_checkpoint(new_checkpoint)
        result.log_checkpoint = new_checkpoint

        elapsed = (datetime.now(tz=timezone.utc) - start).total_seconds()
        result.duration_seconds = round(elapsed, 2)

        log.info(
            "Consolidation complete: %d created, %d superseded in %.1fs",
            result.memories_created,
            result.memories_superseded,
            result.duration_seconds,
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _supersede_contradictions(
        self,
        new_memory: Memory,
        embedding: list[float],
        store: MemoryStore,
        result: ConsolidationResult,
    ) -> bool:
        """Check existing memories for contradictions; supersede if found. Returns True if superseded."""
        if not hasattr(store, "find_similar"):
            return False
        try:
            similar = store.find_similar(  # type: ignore[union-attr]
                embedding,
                new_memory.layer,
                threshold=0.80,
                limit=3,
            )
        except Exception:
            return False

        for old_memory, similarity in similar:
            if old_memory.id == new_memory.id:
                continue
            try:
                verdict = self._check_contradiction(old_memory.content, new_memory.content)
            except Exception:
                continue
            if verdict:
                log.info(
                    "Contradiction detected (sim=%.2f): superseding %r with %r",
                    similarity,
                    old_memory.content[:60],
                    new_memory.content[:60],
                )
                store.supersede(old_memory.id, new_memory)
                result.memories_superseded += 1
                return True
        return False

    @_trace(name="extremis.consolidation.contradiction_check", capture_io=False)
    def _check_contradiction(self, old_content: str, new_content: str) -> bool:
        """Return True if new_content contradicts old_content."""
        response = self._client.messages.create(
            model=self._config.consolidation_model,
            max_tokens=5,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Old memory: {old_content}\n"
                        f"New memory: {new_content}\n\n"
                        "Does the new memory directly contradict or supersede the old memory? "
                        "Reply only 'yes' or 'no'."
                    ),
                }
            ],
        )
        return response.content[0].text.strip().lower().startswith("yes")

    @_trace(name="extremis.consolidation.extract", capture_io=False)
    def _extract(self, conv_id: str, entries: list[LogEntry]) -> list[dict]:
        log_text = "\n".join(f"[{e.role.upper()}] {e.content}" for e in entries)
        user_msg = EXTRACTION_USER_TEMPLATE.format(
            conversation_id=conv_id,
            count=len(entries),
            log_text=log_text,
        )

        response = self._client.messages.create(
            model=self._config.consolidation_model,
            max_tokens=1024,
            system=EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text.strip()
        return self._parse_response(raw)

    @staticmethod
    def _parse_response(raw: str) -> list[dict]:
        # Strip accidental markdown fences
        text = raw
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
        text = text.strip()

        try:
            data = json.loads(text)
            return data.get("memories", [])
        except json.JSONDecodeError:
            log.warning("Failed to parse LLM JSON response: %r", raw[:200])
            return []

    @staticmethod
    def _group_by_conversation(entries: list[LogEntry]) -> dict[str, list[LogEntry]]:
        groups: dict[str, list[LogEntry]] = {}
        for entry in entries:
            groups.setdefault(entry.conversation_id, []).append(entry)
        return groups

    @staticmethod
    def _end_checkpoint(log_store: LogStore) -> str:
        return log_store.current_checkpoint()
