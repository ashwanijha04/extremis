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
from ..verification import (
    LLMJudge,
    NLIChecker,
    recommend_for_verification,
    recommendations_to_metadata,
    self_consistency_filter,
    verify,
)
from .prompts import EXTRACTION_SYSTEM, EXTRACTION_USER_TEMPLATE

log = logging.getLogger(__name__)

_BATCH_SIZE = 30  # max log entries per LLM call
_MAX_PARENT_LINKS = 10  # cap episodic ancestors per consolidated memory


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

        # Verification components — lazy-init so tests that mock the LLM
        # don't pay the NLI model load cost.
        self._nli: NLIChecker | None = None
        self._judge: LLMJudge | None = None
        self._consistency_layers = {s.strip().lower() for s in config.self_consistency_layers.split(",") if s.strip()}

    def _get_nli(self) -> NLIChecker | None:
        if not self._config.enable_faithfulness_check:
            return None
        if self._nli is None:
            try:
                self._nli = NLIChecker(self._config.faithfulness_nli_model)
            except Exception as exc:
                log.info("NLI unavailable, falling back to judge-only: %s", exc)
                self._nli = None
        return self._nli

    def _get_judge(self) -> LLMJudge | None:
        if not self._config.enable_faithfulness_check:
            return None
        if self._judge is None:
            self._judge = LLMJudge(self._client, self._config.consolidation_model)
        return self._judge

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
                # Self-consistency: re-sample the extractor N times and keep
                # only claims that converge across samples. Layers outside
                # self_consistency_layers pass through untouched.
                extracted = self._extract_with_consistency(conv_id, batch)
                if extracted is None:
                    continue

                # Joined conversation text used for faithfulness check
                source_messages = [f"[{e.role.upper()}] {e.content}" for e in batch]

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

                    # Faithfulness check: NLI → judge on grey zone.
                    # Tag + downrank; never drop. False-positive verifiers
                    # reduce surfacing but don't erase the memory.
                    final_confidence = confidence
                    metadata: dict = {
                        "source": "consolidation",
                        "conversation_id": conv_id,
                        "model": self._config.consolidation_model,
                    }
                    if self._config.enable_faithfulness_check:
                        verification = verify(
                            content,
                            source_messages,
                            nli=self._get_nli(),
                            judge=self._get_judge(),
                            pass_threshold=self._config.faithfulness_pass_threshold,
                            grey_zone_low=self._config.faithfulness_grey_zone_low,
                        )
                        metadata["verification"] = verification.to_metadata()
                        final_confidence = min(confidence, verification.score)
                        if verification.best_source_idx is not None:
                            entry = batch[verification.best_source_idx]
                            metadata["source_message_ids"] = [str(entry.id)]

                        # Stamp actionable recommendations for issues found.
                        # Operators can inspect metadata.recommendations to see
                        # what to do about each flagged memory.
                        recs = recommend_for_verification(
                            metadata["verification"],
                            memory_refs={
                                "conversation_id": conv_id,
                                "source_message_ids": metadata.get("source_message_ids", []),
                            },
                        )
                        if recs:
                            metadata["recommendations"] = recommendations_to_metadata(recs)

                    # Consistency stats stamped by self_consistency_filter
                    consistency = item.pop("_consistency", None)
                    if consistency is not None:
                        metadata["consistency"] = consistency

                    # Parent-child lineage: link consolidated memory to the
                    # episodic ancestors from this conversation so callers
                    # can walk source_memory_ids back to the originating log.
                    source_memory_ids = []
                    if layer in (MemoryLayer.SEMANTIC, MemoryLayer.PROCEDURAL):
                        source_memory_ids = self._find_episodic_ancestors(embedding, conv_id, local)

                    memory = Memory(
                        layer=layer,
                        content=content,
                        embedding=embedding,
                        confidence=final_confidence,
                        metadata=metadata,
                        source_memory_ids=source_memory_ids,
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

    def _extract_with_consistency(self, conv_id: str, batch: list[LogEntry]) -> list[dict] | None:
        """Run extraction once normally, or N times under self-consistency.

        Returns the list of accepted claims (with _consistency stats stamped
        on those that went through the filter), or None on extraction error.
        """
        n = self._config.self_consistency_n
        if n <= 1 or not self._consistency_layers:
            try:
                return self._extract(conv_id, batch)
            except Exception as exc:
                log.warning("Consolidation extraction failed for %s: %s", conv_id, exc)
                return None

        def _do_extract() -> list[dict]:
            return self._extract(
                conv_id,
                batch,
                temperature=self._config.self_consistency_temperature,
            )

        try:
            kept, _stats = self_consistency_filter(
                _do_extract,
                embedder=self._embedder,
                n=n,
                threshold=self._config.consistency_threshold,
                layers_in_scope=self._consistency_layers,
            )
            return kept
        except Exception as exc:
            log.warning("Self-consistency failed for %s: %s", conv_id, exc)
            try:
                return self._extract(conv_id, batch)
            except Exception:
                return None

    def _find_episodic_ancestors(
        self,
        embedding: list[float],
        conv_id: str,
        local: MemoryStore,
    ) -> list:
        """Find episodic memories from the same conversation that ground this claim.

        Walks the embedding-nearest episodic memories and filters to the
        same conversation_id. Caps at _MAX_PARENT_LINKS to keep the list bounded.
        """
        try:
            candidates = local.search(
                embedding,
                layers=[MemoryLayer.EPISODIC],
                limit=_MAX_PARENT_LINKS * 3,
                min_score=0.0,
            )
        except Exception:
            return []

        ancestors = []
        for cand in candidates:
            if (cand.memory.metadata or {}).get("conversation_id") == conv_id:
                ancestors.append(cand.memory.id)
                if len(ancestors) >= _MAX_PARENT_LINKS:
                    break
        return ancestors

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
    def _extract(
        self,
        conv_id: str,
        entries: list[LogEntry],
        temperature: float = 0.0,
    ) -> list[dict]:
        log_text = "\n".join(f"[{e.role.upper()}] {e.content}" for e in entries)
        user_msg = EXTRACTION_USER_TEMPLATE.format(
            conversation_id=conv_id,
            count=len(entries),
            log_text=log_text,
        )

        kwargs: dict = {
            "model": self._config.consolidation_model,
            "max_tokens": 1024,
            "system": EXTRACTION_SYSTEM,
            "messages": [{"role": "user", "content": user_msg}],
        }
        if temperature > 0.0:
            kwargs["temperature"] = temperature

        response = self._client.messages.create(**kwargs)

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
