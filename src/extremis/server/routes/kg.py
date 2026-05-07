from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ...types import EntityType
from ..deps import Memory

router = APIRouter(tags=["knowledge-graph"])


class KGWriteRequest(BaseModel):
    operation: str                  # add_entity | add_relationship | add_attribute
    name: str = ""
    entity_type: str = "concept"
    from_entity: str = ""
    to_entity: str = ""
    rel_type: str = ""
    weight: float = 1.0
    key: str = ""
    value: str = ""
    metadata: dict = {}


class KGQueryRequest(BaseModel):
    name: str
    traverse_depth: int = 0


@router.post("/write")
def kg_write(req: KGWriteRequest, mem: Memory) -> dict:
    if req.operation == "add_entity":
        entity = mem.kg_add_entity(req.name, EntityType(req.entity_type), req.metadata or None)
        return entity.model_dump(mode="json")
    elif req.operation == "add_relationship":
        rel = mem.kg_add_relationship(
            req.from_entity, req.to_entity, req.rel_type, req.weight, req.metadata or None
        )
        return rel.model_dump(mode="json")
    elif req.operation == "add_attribute":
        attr = mem.kg_add_attribute(req.name, req.key, req.value)
        return attr.model_dump(mode="json")
    from fastapi import HTTPException
    raise HTTPException(400, detail=f"Unknown operation: {req.operation}")


@router.post("/query")
def kg_query(req: KGQueryRequest, mem: Memory) -> dict:
    if req.traverse_depth > 0:
        results = mem.kg_traverse(req.name, depth=req.traverse_depth)
        return {"results": [r.model_dump(mode="json") for r in results]}
    result = mem.kg_query(req.name)
    if result is None:
        return {"result": None}
    return {"result": result.model_dump(mode="json")}
