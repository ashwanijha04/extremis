from __future__ import annotations

from fastapi import APIRouter

from ..deps import Memory, Namespace

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@router.post("/attention/score")
def score_attention(
    message: str,
    mem: Memory,
    sender: str = "",
    channel: str = "dm",
    owner_ids: str = "",
    allowlist: str = "",
    ongoing: bool = False,
    already_answered: bool = False,
) -> dict:
    owners = {s.strip() for s in owner_ids.split(",") if s.strip()}
    allowed = {s.strip() for s in allowlist.split(",") if s.strip()}
    ctx = {"ongoing": ongoing, "already_answered": already_answered}
    result = mem.score_attention(message, sender=sender, channel=channel,
                                  owner_ids=owners, allowlist=allowed, context=ctx)
    return result.model_dump()


@router.get("/usage")
def usage(namespace: Namespace) -> dict:
    return {"namespace": namespace, "note": "detailed usage metrics coming in v2"}
