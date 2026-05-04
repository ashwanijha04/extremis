"""SQLiteKGStore — knowledge graph tests."""
from __future__ import annotations

import pytest

from lore_ai.types import EntityType


class TestAddEntity:
    def test_add_and_query(self, kg_store):
        kg_store.add_entity("Alice", EntityType.PERSON)
        result = kg_store.query_entity("Alice")
        assert result is not None
        assert result.entity.name == "Alice"
        assert result.entity.type == EntityType.PERSON

    def test_add_with_metadata(self, kg_store):
        kg_store.add_entity("Acme", EntityType.ORG, metadata={"founded": 2010})
        result = kg_store.query_entity("Acme")
        assert result.entity.metadata == {"founded": 2010}

    def test_upsert_updates_type(self, kg_store):
        kg_store.add_entity("Node", EntityType.CONCEPT)
        kg_store.add_entity("Node", EntityType.PROJECT)
        result = kg_store.query_entity("Node")
        assert result.entity.type == EntityType.PROJECT

    def test_unknown_entity_returns_none(self, kg_store):
        assert kg_store.query_entity("DoesNotExist") is None

    def test_list_entities(self, kg_store):
        kg_store.add_entity("Alice", EntityType.PERSON)
        kg_store.add_entity("Acme", EntityType.ORG)
        all_entities = kg_store.list_entities()
        names = {e.name for e in all_entities}
        assert "Alice" in names
        assert "Acme" in names

    def test_list_entities_by_type(self, kg_store):
        kg_store.add_entity("Alice", EntityType.PERSON)
        kg_store.add_entity("Acme", EntityType.ORG)
        people = kg_store.list_entities(type=EntityType.PERSON)
        assert all(e.type == EntityType.PERSON for e in people)
        assert len(people) == 1


class TestRelationships:
    def test_add_and_query_relationship(self, kg_store):
        kg_store.add_entity("Alice", EntityType.PERSON)
        kg_store.add_entity("Acme", EntityType.ORG)
        kg_store.add_relationship("Alice", "Acme", "works_at", weight=0.9)

        result = kg_store.query_entity("Alice")
        rels = {(r.from_entity, r.to_entity, r.rel_type) for r in result.relationships}
        assert ("Alice", "Acme", "works_at") in rels

    def test_relationship_shows_on_both_sides(self, kg_store):
        kg_store.add_entity("Alice", EntityType.PERSON)
        kg_store.add_entity("Acme", EntityType.ORG)
        kg_store.add_relationship("Alice", "Acme", "works_at")

        acme_result = kg_store.query_entity("Acme")
        rel_names = {(r.from_entity, r.rel_type) for r in acme_result.relationships}
        assert ("Alice", "works_at") in rel_names

    def test_relationship_weight(self, kg_store):
        kg_store.add_entity("A", EntityType.CONCEPT)
        kg_store.add_entity("B", EntityType.CONCEPT)
        kg_store.add_relationship("A", "B", "linked", weight=0.42)

        result = kg_store.query_entity("A")
        rel = next(r for r in result.relationships if r.rel_type == "linked")
        assert rel.weight == pytest.approx(0.42)

    def test_upsert_updates_weight(self, kg_store):
        kg_store.add_entity("A", EntityType.CONCEPT)
        kg_store.add_entity("B", EntityType.CONCEPT)
        kg_store.add_relationship("A", "B", "linked", weight=0.5)
        kg_store.add_relationship("A", "B", "linked", weight=0.9)

        result = kg_store.query_entity("A")
        rel = next(r for r in result.relationships if r.rel_type == "linked")
        assert rel.weight == pytest.approx(0.9)


class TestAttributes:
    def test_add_and_query_attribute(self, kg_store):
        kg_store.add_entity("Alice", EntityType.PERSON)
        kg_store.add_attribute("Alice", "timezone", "Asia/Dubai")

        result = kg_store.query_entity("Alice")
        attr_map = {a.key: a.value for a in result.attributes}
        assert attr_map["timezone"] == "Asia/Dubai"

    def test_upsert_updates_value(self, kg_store):
        kg_store.add_entity("Alice", EntityType.PERSON)
        kg_store.add_attribute("Alice", "timezone", "UTC")
        kg_store.add_attribute("Alice", "timezone", "Asia/Dubai")

        result = kg_store.query_entity("Alice")
        attr_map = {a.key: a.value for a in result.attributes}
        assert attr_map["timezone"] == "Asia/Dubai"

    def test_query_by_attribute(self, kg_store):
        kg_store.add_entity("Alice", EntityType.PERSON)
        kg_store.add_entity("Bob", EntityType.PERSON)
        kg_store.add_attribute("Alice", "team", "backend")
        kg_store.add_attribute("Bob", "team", "frontend")

        backend_team = kg_store.query_by_attribute("team", "backend")
        assert len(backend_team) == 1
        assert backend_team[0].name == "Alice"

    def test_query_by_attribute_key_only(self, kg_store):
        kg_store.add_entity("Alice", EntityType.PERSON)
        kg_store.add_entity("Bob", EntityType.PERSON)
        kg_store.add_attribute("Alice", "phone", "+1234")
        kg_store.add_attribute("Bob", "phone", "+5678")

        with_phone = kg_store.query_by_attribute("phone")
        assert len(with_phone) == 2


class TestTraverse:
    def test_traverse_depth_zero_returns_entity(self, kg_store):
        kg_store.add_entity("Alice", EntityType.PERSON)
        results = kg_store.traverse("Alice", depth=0)
        assert len(results) == 1
        assert results[0].entity.name == "Alice"

    def test_traverse_depth_one_returns_neighbours(self, kg_store):
        kg_store.add_entity("Alice", EntityType.PERSON)
        kg_store.add_entity("Acme", EntityType.ORG)
        kg_store.add_entity("Bob", EntityType.PERSON)
        kg_store.add_relationship("Alice", "Acme", "works_at")
        kg_store.add_relationship("Alice", "Bob", "friend")

        results = kg_store.traverse("Alice", depth=1)
        names = {r.entity.name for r in results}
        assert "Alice" in names
        assert "Acme" in names
        assert "Bob" in names

    def test_traverse_no_duplicates(self, kg_store):
        kg_store.add_entity("Alice", EntityType.PERSON)
        kg_store.add_entity("Bob", EntityType.PERSON)
        kg_store.add_relationship("Alice", "Bob", "friend")
        kg_store.add_relationship("Bob", "Alice", "friend")

        results = kg_store.traverse("Alice", depth=2)
        names = [r.entity.name for r in results]
        assert len(names) == len(set(names))

    def test_traverse_unknown_entity(self, kg_store):
        results = kg_store.traverse("Nobody", depth=2)
        assert results == []


class TestExportMarkdown:
    def test_export_contains_entity_name(self, kg_store):
        kg_store.add_entity("Alice", EntityType.PERSON)
        md = kg_store.export_markdown()
        assert "Alice" in md

    def test_export_contains_relationship(self, kg_store):
        kg_store.add_entity("Alice", EntityType.PERSON)
        kg_store.add_entity("Acme", EntityType.ORG)
        kg_store.add_relationship("Alice", "Acme", "works_at")
        md = kg_store.export_markdown()
        assert "works_at" in md
        assert "Acme" in md

    def test_export_contains_attribute(self, kg_store):
        kg_store.add_entity("Alice", EntityType.PERSON)
        kg_store.add_attribute("Alice", "phone", "+971")
        md = kg_store.export_markdown()
        assert "phone" in md
        assert "+971" in md
