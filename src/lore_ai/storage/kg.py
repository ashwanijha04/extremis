"""
SQLite knowledge graph — entities, relationships, attributes.
Ported and generalised from friday-saas/base/scripts/knowledge-graph.py.
"""
from __future__ import annotations

import json
import sqlite3
from collections import deque
from pathlib import Path
from typing import Optional

from ..config import Config
from ..types import (
    Entity,
    EntityResult,
    EntityType,
    KGAttribute,
    Relationship,
)


class SQLiteKGStore:
    """
    Lightweight knowledge graph stored in the same SQLite DB as memories.
    Supports entities, directed relationships, and key-value attributes.
    All data is scoped to a namespace.
    """

    def __init__(self, db_path: str, config: Config) -> None:
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._config = config
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        schema_path = Path(__file__).parent.parent / "migrations" / "001_initial_sqlite.sql"
        self._conn.executescript(schema_path.read_text())
        self._conn.commit()

    @property
    def _ns(self) -> str:
        return self._config.namespace

    # ------------------------------------------------------------------ #
    # Write operations
    # ------------------------------------------------------------------ #

    def add_entity(
        self,
        name: str,
        type: EntityType,
        metadata: Optional[dict] = None,
    ) -> Entity:
        self._conn.execute(
            """
            INSERT INTO kg_entities (namespace, name, type, metadata)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(namespace, name) DO UPDATE SET
                type = EXCLUDED.type,
                metadata = EXCLUDED.metadata
            """,
            (self._ns, name, type.value, json.dumps(metadata or {})),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM kg_entities WHERE namespace = ? AND name = ?",
            (self._ns, name),
        ).fetchone()
        return self._row_to_entity(row)

    def add_relationship(
        self,
        from_entity: str,
        to_entity: str,
        rel_type: str,
        weight: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> Relationship:
        self._conn.execute(
            """
            INSERT INTO kg_relationships (namespace, from_entity, to_entity, rel_type, weight, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, from_entity, to_entity, rel_type) DO UPDATE SET
                weight = EXCLUDED.weight,
                metadata = EXCLUDED.metadata
            """,
            (self._ns, from_entity, to_entity, rel_type, weight, json.dumps(metadata or {})),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM kg_relationships WHERE namespace = ? AND from_entity = ? AND to_entity = ? AND rel_type = ?",
            (self._ns, from_entity, to_entity, rel_type),
        ).fetchone()
        return self._row_to_rel(row)

    def add_attribute(self, entity: str, key: str, value: str) -> KGAttribute:
        self._conn.execute(
            """
            INSERT INTO kg_attributes (namespace, entity, key, value)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(namespace, entity, key) DO UPDATE SET value = EXCLUDED.value
            """,
            (self._ns, entity, key, value),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM kg_attributes WHERE namespace = ? AND entity = ? AND key = ?",
            (self._ns, entity, key),
        ).fetchone()
        return self._row_to_attr(row)

    # ------------------------------------------------------------------ #
    # Read operations
    # ------------------------------------------------------------------ #

    def query_entity(self, name: str) -> Optional[EntityResult]:
        row = self._conn.execute(
            "SELECT * FROM kg_entities WHERE namespace = ? AND name = ?",
            (self._ns, name),
        ).fetchone()
        if not row:
            return None

        entity = self._row_to_entity(row)

        rel_rows = self._conn.execute(
            "SELECT * FROM kg_relationships WHERE namespace = ? AND (from_entity = ? OR to_entity = ?)",
            (self._ns, name, name),
        ).fetchall()

        attr_rows = self._conn.execute(
            "SELECT * FROM kg_attributes WHERE namespace = ? AND entity = ?",
            (self._ns, name),
        ).fetchall()

        return EntityResult(
            entity=entity,
            relationships=[self._row_to_rel(r) for r in rel_rows],
            attributes=[self._row_to_attr(r) for r in attr_rows],
        )

    def traverse(self, entity_name: str, depth: int = 2) -> list[EntityResult]:
        """BFS traversal up to `depth` hops from entity_name."""
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(entity_name, 0)])
        results: list[EntityResult] = []

        while queue:
            name, current_depth = queue.popleft()
            if name in visited or current_depth > depth:
                continue
            visited.add(name)

            result = self.query_entity(name)
            if result:
                results.append(result)
                if current_depth < depth:
                    for rel in result.relationships:
                        neighbour = rel.to_entity if rel.from_entity == name else rel.from_entity
                        if neighbour not in visited:
                            queue.append((neighbour, current_depth + 1))

        return results

    def query_by_attribute(self, key: str, value: Optional[str] = None) -> list[Entity]:
        if value:
            rows = self._conn.execute(
                "SELECT e.* FROM kg_entities e JOIN kg_attributes a ON e.name = a.entity AND e.namespace = a.namespace WHERE e.namespace = ? AND a.key = ? AND a.value = ?",
                (self._ns, key, value),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT e.* FROM kg_entities e JOIN kg_attributes a ON e.name = a.entity AND e.namespace = a.namespace WHERE e.namespace = ? AND a.key = ?",
                (self._ns, key),
            ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def list_entities(self, type: Optional[EntityType] = None) -> list[Entity]:
        if type:
            rows = self._conn.execute(
                "SELECT * FROM kg_entities WHERE namespace = ? AND type = ?",
                (self._ns, type.value),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM kg_entities WHERE namespace = ?", (self._ns,)
            ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def export_markdown(self) -> str:
        lines = [f"# Knowledge Graph — namespace: {self._ns}\n"]
        entities = self.list_entities()
        for entity in entities:
            result = self.query_entity(entity.name)
            if not result:
                continue
            lines.append(f"## {entity.name} ({entity.type.value})")
            if entity.metadata:
                lines.append(f"  metadata: {json.dumps(entity.metadata)}")
            for rel in result.relationships:
                arrow = "→" if rel.from_entity == entity.name else "←"
                other = rel.to_entity if rel.from_entity == entity.name else rel.from_entity
                lines.append(f"  {arrow} [{rel.rel_type}] {other}  (weight={rel.weight:.1f})")
            for attr in result.attributes:
                lines.append(f"  {attr.key}: {attr.value}")
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Row converters
    # ------------------------------------------------------------------ #

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> Entity:
        return Entity(
            id=row["id"],
            namespace=row["namespace"],
            name=row["name"],
            type=EntityType(row["type"]),
            metadata=json.loads(row["metadata"]),
        )

    @staticmethod
    def _row_to_rel(row: sqlite3.Row) -> Relationship:
        return Relationship(
            id=row["id"],
            namespace=row["namespace"],
            from_entity=row["from_entity"],
            to_entity=row["to_entity"],
            rel_type=row["rel_type"],
            weight=row["weight"],
            metadata=json.loads(row["metadata"]),
        )

    @staticmethod
    def _row_to_attr(row: sqlite3.Row) -> KGAttribute:
        return KGAttribute(
            id=row["id"],
            namespace=row["namespace"],
            entity=row["entity"],
            key=row["key"],
            value=row["value"],
        )

    def close(self) -> None:
        self._conn.close()
