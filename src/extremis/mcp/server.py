"""
Friday Memory MCP server.

Add to claude_desktop_config.json:

  "mcpServers": {
    "extremis": {
      "command": "extremis-mcp",
      "env": { "FRIDAY_HOME": "~/.friday" }
    }
  }

Or with a custom db path:

  "mcpServers": {
    "extremis": {
      "command": "extremis-mcp",
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

from ..api import Extremis
from ..config import Config
from ..consolidation.consolidator import LLMConsolidator
from ..types import EntityType, MemoryLayer

log = logging.getLogger(__name__)


def create_server(config: Config | None = None) -> FastMCP:
    cfg = config or Config()
    mem = Extremis(config=cfg)

    mcp = FastMCP(
        "extremis",
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

    # ------------------------------------------------------------------ #
    # Tool 6 — knowledge graph: add entity / relationship / attribute
    # ------------------------------------------------------------------ #
    @mcp.tool()
    def memory_kg_write(
        operation: str,
        name: str,
        entity_type: str = "concept",
        from_entity: str = "",
        to_entity: str = "",
        rel_type: str = "",
        weight: float = 1.0,
        key: str = "",
        value: str = "",
        metadata: str = "{}",
    ) -> str:
        """
        Write to the knowledge graph.

        operation: "add_entity" | "add_relationship" | "add_attribute"

        add_entity:
          name        — entity name (e.g. "Alice")
          entity_type — person | org | project | group | concept | other

        add_relationship:
          from_entity — source entity name
          to_entity   — target entity name
          rel_type    — relationship label (e.g. "works_at", "friend", "owns")
          weight      — confidence 0.0–1.0 (default 1.0)

        add_attribute:
          name  — entity name
          key   — attribute key (e.g. "phone", "timezone", "tone")
          value — attribute value
        """
        import json as _json
        meta = _json.loads(metadata) if metadata and metadata != "{}" else {}
        if operation == "add_entity":
            try:
                etype = EntityType(entity_type)
            except ValueError:
                return f"Unknown entity_type '{entity_type}'. Valid: {[e.value for e in EntityType]}"
            entity = mem.kg_add_entity(name, etype, meta)
            return f"Entity [{entity.type.value}] '{entity.name}' added (namespace={cfg.namespace})"

        elif operation == "add_relationship":
            if not from_entity or not to_entity or not rel_type:
                return "add_relationship requires: from_entity, to_entity, rel_type"
            rel = mem.kg_add_relationship(from_entity, to_entity, rel_type, weight, meta)
            return f"Relationship {rel.from_entity} —[{rel.rel_type}]→ {rel.to_entity} (weight={rel.weight})"

        elif operation == "add_attribute":
            if not key or not value:
                return "add_attribute requires: name (entity), key, value"
            attr = mem.kg_add_attribute(name, key, value)
            return f"Attribute '{attr.key}' = '{attr.value}' set on '{attr.entity}'"

        return f"Unknown operation '{operation}'. Valid: add_entity, add_relationship, add_attribute"

    # ------------------------------------------------------------------ #
    # Tool 7 — knowledge graph: query
    # ------------------------------------------------------------------ #
    @mcp.tool()
    def memory_kg_query(name: str, traverse_depth: int = 0) -> str:
        """
        Query the knowledge graph for an entity and its connections.

        Args:
            name:           Entity name to look up.
            traverse_depth: BFS depth (0 = entity + direct connections only,
                            2 = two hops out). Keep ≤ 3 to avoid large results.
        """
        if traverse_depth > 0:
            results = mem.kg_traverse(name, depth=traverse_depth)
        else:
            result = mem.kg_query(name)
            results = [result] if result else []

        if not results:
            return f"No entity found for '{name}'."

        lines = []
        for er in results:
            lines.append(f"[{er.entity.type.value}] {er.entity.name}")
            if er.entity.metadata:
                lines.append(f"  metadata: {er.entity.metadata}")
            for rel in er.relationships:
                arrow = "→" if rel.from_entity == er.entity.name else "←"
                other = rel.to_entity if rel.from_entity == er.entity.name else rel.from_entity
                lines.append(f"  {arrow} [{rel.rel_type}] {other}  w={rel.weight:.1f}")
            for attr in er.attributes:
                lines.append(f"  {attr.key}: {attr.value}")
            lines.append("")
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------ #
    # Tool 8 — observe: compress log entries to priority observations
    # ------------------------------------------------------------------ #
    @mcp.tool()
    def memory_observe(conversation_id: str = "default") -> str:
        """
        Compress recent log entries for a conversation into priority-tagged observations.

        Classifies each log entry as:
          🔴 CRITICAL — decisions, errors, deadlines, reward signals
          🟡 CONTEXT  — reasons, insights, learnings
          🟢 INFO     — everything else

        Returns a markdown block suitable for injecting into context.
        No LLM call — pure heuristic, runs instantly.

        Args:
            conversation_id: Which conversation's entries to compress.
        """
        from ..observer.observer import HeuristicObserver
        observations = mem.observe(conversation_id)
        if not observations:
            return f"No log entries found for conversation '{conversation_id}'."
        return HeuristicObserver.format_markdown(observations)

    # ------------------------------------------------------------------ #
    # Tool 9 — attention scorer
    # ------------------------------------------------------------------ #
    @mcp.tool()
    def memory_score_attention(
        message: str,
        sender: str = "",
        channel: str = "dm",
        owner_ids: str = "",
        allowlist: str = "",
        ongoing: bool = False,
        already_answered: bool = False,
    ) -> str:
        """
        Score an incoming message to decide how much attention to give it.

        Returns a score (0–100) and a processing level:
          full     — engage fully
          standard — engage, balanced response
          minimal  — brief acknowledgement only
          ignore   — skip this message

        Use this before deciding whether to generate a full response, especially
        in group chat or broadcast contexts where not every message warrants a reply.

        Args:
            message:          The incoming message text.
            sender:           Sender identifier (user ID, phone number, etc.)
            channel:          "dm" | "group" | "broadcast"
            owner_ids:        Comma-separated sender IDs that always get full attention.
            allowlist:        Comma-separated sender IDs with elevated base score.
            ongoing:          True if this is part of an ongoing conversation thread.
            already_answered: True if someone else already replied.
        """
        owners = {s.strip() for s in owner_ids.split(",") if s.strip()}
        allowed = {s.strip() for s in allowlist.split(",") if s.strip()}
        ctx = {"ongoing": ongoing, "already_answered": already_answered}

        result = mem.score_attention(
            message, sender=sender, channel=channel,
            owner_ids=owners, allowlist=allowed, context=ctx,
        )
        return (
            f"Score: {result.score}/100  Level: {result.level}\n"
            f"Reason: {result.reason}\n"
            f"Breakdown: {result.breakdown}"
        )

    return mcp


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Friday Memory MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default=os.environ.get("FRIDAY_TRANSPORT", "stdio"),
        help="Transport mode: stdio (default, for Claude Desktop/Code) or sse (HTTP server)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host for SSE mode (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="Port for SSE mode (default: 8765)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    server = create_server()

    if args.transport == "sse":
        log.info("Starting extremis MCP server on http://%s:%d", args.host, args.port)
        server.run(transport="sse", host=args.host, port=args.port)
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
