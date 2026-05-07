"""
LLMCompactor — reconcile contradictions in existing structured memories.

Unlike LLMConsolidator (which distils new log entries into structured memories),
the compactor works on memories that are *already* structured. It loads all
memories for a given layer, asks Claude to identify and resolve contradictions,
and supersedes the conflicting set with reconciled replacements.

Use memory_compact when:
- You've accumulated contradictory semantic/procedural memories over time
- You want a one-off cleanup of test/dev fixtures
- After significant new learning that may conflict with old beliefs

Do NOT confuse with memory_consolidate, which only processes new log entries.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from uuid import UUID

import anthropic

from ..config import Config
from ..interfaces import Embedder, MemoryStore
from ..types import CompactionResult, Memory, MemoryLayer

log = logging.getLogger(__name__)

_COMPACTION_SYSTEM = """\
You are reviewing a set of stored memories about a user or agent.
Your job is to identify contradictions and duplicates, then produce a reconciled set.

Rules:
- If two memories say opposite things about the same topic, keep the one with the
  higher confidence score (or the more recent one if equal), and mark the other as superseded.
- If two memories say essentially the same thing in different words, merge them into
  one cleaner statement and mark both originals as superseded.
- If a memory is clearly outdated (superseded by a more recent fact), mark it.
- Do NOT invent new content. Only reconcile what exists.
- Memories with do_not_consolidate=true must be left unchanged.

Return JSON only:
{
  "groups": [
    {
      "action": "keep" | "supersede" | "merge",
      "memory_ids": ["uuid1", "uuid2"],
      "reconciled_content": "...",   // only for merge/supersede actions
      "reconciled_confidence": 0.0,  // only for merge/supersede
      "reason": "..."
    }
  ]
}

If there is nothing to reconcile, return {"groups": []}.
"""


class LLMCompactor:
    def __init__(self, config: Config, embedder: Embedder) -> None:
        self._config = config
        self._embedder = embedder
        self._client = anthropic.Anthropic()

    def run(
        self,
        store: MemoryStore,
        layer: MemoryLayer = MemoryLayer.SEMANTIC,
        batch_size: int = 40,
    ) -> CompactionResult:
        start = datetime.now(tz=timezone.utc)
        result = CompactionResult()

        memories = store.list_recent(layer=layer, limit=1000)
        if len(memories) < 2:
            log.info("Compaction: fewer than 2 memories in layer %s — nothing to do", layer.value)
            return result

        log.info("Compaction: reviewing %d %s memories", len(memories), layer.value)

        # Process in batches to stay within context limits
        for i in range(0, len(memories), batch_size):
            batch = memories[i : i + batch_size]
            try:
                groups = self._review_batch(batch)
            except Exception as exc:
                log.warning("Compaction batch %d failed: %s", i // batch_size, exc)
                continue

            for group in groups:
                action = group.get("action", "keep")
                if action == "keep":
                    result.memories_unchanged += len(group.get("memory_ids", []))
                    continue

                ids = [UUID(mid) for mid in group.get("memory_ids", [])]
                reconciled_content = group.get("reconciled_content", "")
                reconciled_confidence = float(group.get("reconciled_confidence", 0.8))

                if not reconciled_content or not ids:
                    continue

                # Find the original memories to preserve scores/metadata
                originals = [m for m in batch if m.id in ids]
                if not originals:
                    continue

                best_score = max(m.score for m in originals)
                try:
                    new_embedding = self._embedder.embed(reconciled_content)
                except Exception as exc:
                    log.warning("Compaction embedding failed: %s", exc)
                    continue

                reconciled = Memory(
                    layer=layer,
                    content=reconciled_content,
                    embedding=new_embedding,
                    confidence=reconciled_confidence,
                    score=best_score,
                    metadata={
                        "source": "compaction",
                        "action": action,
                        "reason": group.get("reason", ""),
                    },
                    source_memory_ids=ids,
                    validity_start=datetime.now(tz=timezone.utc),
                )

                # Supersede all originals with the reconciled memory
                first = True
                for original in originals:
                    if first:
                        store.supersede(original.id, reconciled)
                        first = False
                    else:
                        # Close additional originals without creating duplicates
                        from datetime import timezone as _tz

                        closed = original.model_copy(update={"validity_end": datetime.now(tz=_tz.utc)})
                        store.store(closed)

                if action == "merge":
                    result.memories_deduped += len(ids)
                else:
                    result.memories_reconciled += len(ids)

        result.duration_seconds = round((datetime.now(tz=timezone.utc) - start).total_seconds(), 2)
        log.info(
            "Compaction complete: %d reconciled, %d deduped in %.1fs",
            result.memories_reconciled,
            result.memories_deduped,
            result.duration_seconds,
        )
        return result

    def _review_batch(self, memories: list[Memory]) -> list[dict]:
        memory_list = "\n".join(
            f"- id={m.id}  confidence={m.confidence:.2f}  score={m.score:+.1f}"
            f"  do_not_consolidate={m.do_not_consolidate}\n  content: {m.content}"
            for m in memories
        )
        response = self._client.messages.create(
            model=self._config.consolidation_model,
            max_tokens=2048,
            system=_COMPACTION_SYSTEM,
            messages=[{"role": "user", "content": f"Memories to review:\n\n{memory_list}"}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
        try:
            return json.loads(raw).get("groups", [])
        except json.JSONDecodeError:
            log.warning("Compactor failed to parse LLM response: %r", raw[:200])
            return []
