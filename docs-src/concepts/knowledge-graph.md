# Knowledge graph

extremis maintains a structured knowledge graph alongside the vector store. Where vectors answer "what's related to this topic?", the graph answers "who works for who?", "what does this entity own?", "how are these things connected?"

## When to use it

| Question type | Tool |
|--------------|------|
| "What does the user prefer?" | `recall()` — semantic search |
| "Who is Alice's manager?" | `kg_query()` — graph traversal |
| "What timezone is Bob in?" | `kg_query()` — attribute lookup |
| "What projects is this team working on?" | `kg_traverse()` — BFS walk |

## Entities

An entity is a named node with a type:

```python
from extremis.types import EntityType

mem.kg_add_entity("Ashwani", EntityType.PERSON)
mem.kg_add_entity("PropertyFinder", EntityType.ORG)
mem.kg_add_entity("extremis", EntityType.PROJECT)
```

Available types: `PERSON`, `ORG`, `PROJECT`, `GROUP`, `CONCEPT`, `OTHER`

## Relationships

A directed edge between two entities:

```python
mem.kg_add_relationship(
    "Ashwani",          # from
    "PropertyFinder",   # to
    "works_at",         # relationship type (any string)
    weight=1.0,         # confidence 0.0–1.0
)
mem.kg_add_relationship("Ashwani", "extremis", "building")
```

## Attributes

Key-value tags on an entity:

```python
mem.kg_add_attribute("Ashwani", "timezone", "Asia/Dubai")
mem.kg_add_attribute("Ashwani", "language", "Python")
mem.kg_add_attribute("Ashwani", "phone", "+971...")
```

## Querying

### Single entity

```python
result = mem.kg_query("Ashwani")

print(result.entity.name)    # "Ashwani"
print(result.entity.type)    # EntityType.PERSON

for rel in result.relationships:
    print(f"{rel.from_entity} → [{rel.rel_type}] → {rel.to_entity}")

for attr in result.attributes:
    print(f"{attr.key}: {attr.value}")
```

### BFS traversal

Walk the graph up to N hops from an entity:

```python
# Everything reachable in 2 hops from Ashwani
graph = mem.kg_traverse("Ashwani", depth=2)

for entity_result in graph:
    print(f"[{entity_result.entity.type.value}] {entity_result.entity.name}")
    for rel in entity_result.relationships:
        print(f"  → [{rel.rel_type}] {rel.to_entity}")
```

### Find by attribute

```python
entities = mem.get_kg().query_by_attribute("timezone", "Asia/Dubai")
for e in entities:
    print(e.name)  # → "Ashwani"
```

## Upsert behaviour

All write operations are upserts — calling them multiple times is safe:

```python
# Adding the same entity twice updates it
mem.kg_add_entity("Ashwani", EntityType.PERSON)
mem.kg_add_entity("Ashwani", EntityType.PERSON, metadata={"seniority": "senior"})

# Adding the same relationship updates weight/metadata
mem.kg_add_relationship("A", "B", "knows", weight=0.5)
mem.kg_add_relationship("A", "B", "knows", weight=0.9)  # updates weight
```

## Namespace isolation

The knowledge graph respects namespaces — entities and relationships are scoped per namespace:

```python
alice_mem = Extremis(config=Config(namespace="alice"))
bob_mem   = Extremis(config=Config(namespace="bob"))

alice_mem.kg_add_entity("Project X", EntityType.PROJECT)

# Bob's graph has no Project X
assert bob_mem.kg_query("Project X") is None
```
