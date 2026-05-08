# MCP tools reference

extremis exposes 10 tools to Claude via the MCP protocol.

## memory_remember

Append a message or fact to the memory log and write it as an episodic memory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | string | required | The text to remember |
| `role` | string | `"user"` | Who said it — `"user"`, `"assistant"`, or `"system"` |
| `conversation_id` | string | `"default"` | Groups messages for consolidation |

**LLM cost:** None

## memory_recall

Retrieve memories relevant to a query using semantic search.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | What you're looking for |
| `limit` | int | `10` | Max memories to return |
| `layers` | string | `""` | Comma-separated filter e.g. `"semantic,procedural"` |

**LLM cost:** None

## memory_report_outcome

Apply a reinforcement signal to recalled memories.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_ids` | string | required | Comma-separated UUIDs from recall results |
| `success` | bool | required | `true` = positive signal, `false` = negative |
| `weight` | float | `1.0` | Signal magnitude |

**LLM cost:** None

## memory_remember_now

Write directly to a structured memory layer, bypassing the log.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | string | required | The memory content |
| `layer` | string | `"semantic"` | `episodic`, `semantic`, `procedural`, `identity`, `working` |
| `confidence` | float | `0.9` | Certainty (0.0–1.0) |
| `expires_at` | string | `""` | ISO8601 expiry for working memories |

**LLM cost:** None

## memory_consolidate

Distil **new log entries** into structured semantic and procedural memories.

!!! note
    Only processes log entries since the last checkpoint. Does not touch existing structured memories. See `memory_compact` for reconciling existing memories.

**LLM cost:** Claude Haiku per conversation batch

## memory_compact

Reconcile contradictions in **existing structured memories**.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layer` | string | `"semantic"` | Which layer to compact |

**LLM cost:** Claude Haiku per batch of memories

## memory_kg_write

Add entity, relationship, or attribute to the knowledge graph.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `operation` | string | required | `add_entity`, `add_relationship`, `add_attribute` |
| `name` | string | `""` | Entity name (for all operations) |
| `entity_type` | string | `"concept"` | `person`, `org`, `project`, `group`, `concept`, `other` |
| `from_entity` | string | `""` | Source entity (for relationships) |
| `to_entity` | string | `""` | Target entity (for relationships) |
| `rel_type` | string | `""` | Relationship label |
| `weight` | float | `1.0` | Confidence |
| `key` | string | `""` | Attribute key |
| `value` | string | `""` | Attribute value |

**LLM cost:** None

## memory_kg_query

Query the knowledge graph for an entity and its connections.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | required | Entity to look up |
| `traverse_depth` | int | `0` | BFS depth (0 = entity + direct connections) |

**LLM cost:** None

## memory_observe

Compress recent log entries into priority-tagged observations.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conversation_id` | string | `"default"` | Which conversation to observe |

Returns entries tagged 🔴 CRITICAL / 🟡 CONTEXT / 🟢 INFO.

**LLM cost:** None (pure heuristic)

## memory_score_attention

Score an incoming message 0–100 to decide how much attention to give it.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | string | required | The incoming message |
| `sender` | string | `""` | Sender identifier |
| `channel` | string | `"dm"` | `dm`, `group`, or `broadcast` |
| `owner_ids` | string | `""` | Comma-separated owner IDs (always full attention) |
| `allowlist` | string | `""` | Comma-separated elevated senders |
| `ongoing` | bool | `false` | Part of an active conversation |
| `already_answered` | bool | `false` | Someone else already replied |

**LLM cost:** None (pure heuristic)
