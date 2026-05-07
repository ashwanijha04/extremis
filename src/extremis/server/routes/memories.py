from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel

from ...types import MemoryLayer
from ..deps import Memory

router = APIRouter(tags=["memories"])


class RememberRequest(BaseModel):
    content: str
    role: str = "user"
    conversation_id: str = "default"
    metadata: dict = {}


class RecallRequest(BaseModel):
    query: str
    limit: int = 10
    layers: Optional[list[str]] = None
    min_score: float = 0.0


class ReportRequest(BaseModel):
    memory_ids: list[UUID]
    success: bool
    weight: float = 1.0


class RememberNowRequest(BaseModel):
    content: str
    layer: str = "semantic"
    confidence: float = 0.9
    expires_at: Optional[datetime] = None
    metadata: dict = {}


class ConsolidateResponse(BaseModel):
    memories_created: int
    memories_superseded: int
    duration_seconds: float
    log_checkpoint: str


@router.post("/remember", status_code=204)
def remember(req: RememberRequest, mem: Memory) -> None:
    mem.remember(req.content, role=req.role, conversation_id=req.conversation_id, metadata=req.metadata)


@router.post("/recall")
def recall(req: RecallRequest, mem: Memory) -> dict:
    layers = [MemoryLayer(lyr) for lyr in req.layers] if req.layers else None
    results = mem.recall(req.query, limit=req.limit, layers=layers, min_score=req.min_score)
    return {"results": [r.model_dump(mode="json") for r in results]}


@router.post("/report", status_code=204)
def report(req: ReportRequest, mem: Memory) -> None:
    mem.report_outcome(req.memory_ids, success=req.success, weight=req.weight)


@router.post("/store")
def store(req: RememberNowRequest, mem: Memory) -> dict:
    layer = MemoryLayer(req.layer)
    memory = mem.remember_now(
        req.content,
        layer=layer,
        expires_at=req.expires_at,
        confidence=req.confidence,
        metadata=req.metadata,
    )
    return memory.model_dump(mode="json", exclude={"embedding"})


@router.post("/consolidate")
def consolidate(mem: Memory) -> ConsolidateResponse:
    from ...consolidation.consolidator import LLMConsolidator

    consolidator = LLMConsolidator(mem._config, mem._embedder)
    result = consolidator.run_pass(mem.get_log(), mem.get_local_store(), mem.get_local_store())
    return ConsolidateResponse(
        memories_created=result.memories_created,
        memories_superseded=result.memories_superseded,
        duration_seconds=result.duration_seconds,
        log_checkpoint=result.log_checkpoint,
    )


@router.get("/observe")
def observe(conversation_id: str, mem: Memory) -> dict:
    observations = mem.observe(conversation_id)
    return {"observations": [o.model_dump(mode="json") for o in observations]}
