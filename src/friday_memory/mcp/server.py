"""
Friday Memory MCP server.

Add to claude_desktop_config.json:

  "mcpServers": {
    "friday-memory": {
      "command": "friday-memory-mcp",
      "env": { "FRIDAY_HOME": "~/.friday" }
    }
  }

Or with a custom db path:

  "mcpServers": {
    "friday-memory": {
      "command": "friday-memory-mcp",
      "env": {
        "FRIDAY_LOCAL_DB_PATH": "/path/to/local.db",
        "FRIDAY_LOG_DIR": "/path/to/logs"
      }
    }
  }
"""

from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID

from mcp.server.fastmcp import FastMCP

from ..api import FridayMemory
from ..config import Config
from ..consolidation.consolidator import LLMConsolidator
from ..types import MemoryLayer

log = logging.getLogger(__name__)


def create_server(config: Config | None = None) -> FastMCP:
    cfg = config or Config()
    mem = FridayMemory(config=cfg)

    mcp = FastMCP(
        "friday-memory",
        instructions=(
            "Use these tools to give yourself persistent memory across conversations. "
            "Call memory_recall at the start of every conversation to retrieve relevant context. "
            "Call memory_remember after learning something durable about the user. "
            "Call memory_report_outcome when the user tells you whether your answer was helpful. "
            "Call memory_consolidate periodically (e.g. every 20 conversations) to distil "
            "episodic logs into semantic and procedural memories."
        ),
    )

    # ------------------------------------------------------------------ #
    # Tool 1 — remember
    # ------------------------------------------------------------------ #
    @mcp.tool()
    def memory_remember(
        content: str,
        role: str = "user",
        conversation_id: str = "default",
    ) -> str:
        """
        Append a message or fact to the memory log and write it as an episodic memory.

        Use this to capture anything worth remembering from the current conversation:
        user statements, assistant replies, decisions made, context established.

        Args:
            content: The text to remember.
            role: Who said it — "user", "assistant", or "system".
            conversation_id: Identifier for the current conversation (use a stable ID
                             so related messages can be grouped during consolidation).
        """
        mem.remember(content, role=role, conversation_id=conversation_id)
        return f"Remembered ({role}): {content[:120]}{'...' if len(content) > 120 else ''}"

    # ------------------------------------------------------------------ #
    # Tool 2 — recall
    # ------------------------------------------------------------------ #
    @mcp.tool()
    def memory_recall(
        query: str,
        limit: int = 10,
        layers: str = "",
    ) -> str:
        """
        Retrieve memories relevant to a query using semantic search.

        Identity and procedural memories are always included (they define who the user
        is and how to behave). Semantic and episodic memories are ranked by relevance,
        utility score, and recency.

        Call this at the start of every conversation with the user's first message
        as the query.

        Args:
            query: What you're looking for — a question, topic, or the user's message.
            limit: Max memories to return (default 10).
            layers: Comma-separated layer filter, e.g. "semantic,procedural".
                    Leave empty to search all layers.
        """
        layer_list = None
        if layers:
            layer_list = [MemoryLayer(l.strip()) for l in layers.split(",") if l.strip()]

        results = mem.recall(query, limit=limit, layers=layer_list)

        if not results:
            return "No relevant memories found."

        lines = [f"Found {len(results)} memories:\n"]
        for i, r in enumerate(results, 1):
            layer_tag = f"[{r.memory.layer.value}]"
            score_tag = f"relevance={r.relevance:.2f}"
            lines.append(f"{i}. {layer_tag} {score_tag}  id={r.memory.id}")
            lines.append(f"   {r.memory.content}")
            if r.memory.validity_end:
                lines.append(f"   expires: {r.memory.validity_end.isoformat()}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Tool 3 — report_outcome
    # ------------------------------------------------------------------ #
    @mcp.tool()
    def memory_report_outcome(
        memory_ids: str,
        success: bool,
        weight: float = 1.0,
    ) -> str:
        """
        Apply a reinforcement signal to memories that were used in a response.

        Call this when the user explicitly rates a response (+1 / -1) or when
        you can clearly infer whether a recalled memory was helpful.

        Args:
            memory_ids: Comma-separated UUIDs of the memories that were recalled
                        and contributed to your answer.
            success: True for a positive signal (+1), False for negative (-1).
            weight: Magnitude of the signal (default 1.0). Use higher values
                    for strong explicit user feedback.
        """
        ids = [UUID(mid.strip()) for mid in memory_ids.split(",") if mid.strip()]
        if not ids:
            return "No valid memory IDs provided."

        mem.report_outcome(ids, success=success, weight=weight)
        signal = f"+{weight}" if success else f"-{weight}"
        return f"Applied {signal} signal to {len(ids)} memor{'y' if len(ids) == 1 else 'ies'}."

    # ------------------------------------------------------------------ #
    # Tool 4 — remember_now
    # ------------------------------------------------------------------ #
    @mcp.tool()
    def memory_remember_now(
        content: str,
        layer: str = "semantic",
        confidence: float = 0.9,
        expires_at: str = "",
    ) -> str:
        """
        Write directly to a structured memory layer, bypassing the log.

        Use this for:
        - Time-sensitive facts with a hard expiry ("flight departs Thursday 06:00")
        - High-confidence facts the user explicitly states about themselves
        - Procedural rules you want active immediately

        For most things, prefer memory_remember (which goes through the log).
        Use memory_remember_now only when the fact needs to be in structured
        memory right now, not after the next consolidation pass.

        Args:
            content: The memory content.
            layer: "episodic" | "semantic" | "procedural" | "identity"
            confidence: How certain you are this is true (0.0–1.0).
            expires_at: ISO8601 datetime when this memory should stop being valid.
                        Leave empty for no expiry.
        """
        try:
            ml = MemoryLayer(layer)
        except ValueError:
            valid = [l.value for l in MemoryLayer]
            return f"Unknown layer '{layer}'. Valid layers: {valid}"

        expiry = datetime.fromisoformat(expires_at) if expires_at else None
        stored = mem.remember_now(content, layer=ml, expires_at=expiry, confidence=confidence)
        return f"Stored [{layer}] memory {stored.id}: {content[:100]}"

    # ------------------------------------------------------------------ #
    # Tool 5 — consolidate
    # ------------------------------------------------------------------ #
    @mcp.tool()
    def memory_consolidate() -> str:
        """
        Run a consolidation pass over unprocessed log entries.

        Reads all log entries since the last checkpoint, calls an LLM to extract
        durable semantic and procedural memories, and writes them to the memory
        store. Updates the checkpoint so re-running is safe (idempotent).

        This is the system's "dream" pass — call it periodically to distil
        episodic logs into lasting knowledge. Typically every 20–50 conversations,
        or once a day.

        Requires ANTHROPIC_API_KEY to be set.
        """
        try:
            consolidator = LLMConsolidator(cfg, mem._embedder)
            result = consolidator.run_pass(
                mem.get_log(),
                mem.get_local_store(),
                mem.get_local_store(),  # same store for MVP; Postgres goes here later
            )
            return (
                f"Consolidation complete.\n"
                f"  Memories created:    {result.memories_created}\n"
                f"  Memories superseded: {result.memories_superseded}\n"
                f"  Duration:            {result.duration_seconds:.1f}s\n"
                f"  Checkpoint:          {result.log_checkpoint}"
            )
        except Exception as exc:
            log.exception("Consolidation failed")
            return f"Consolidation failed: {exc}"

    return mcp


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
