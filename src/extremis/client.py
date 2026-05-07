"""
HostedClient — drop-in replacement for Memory that talks to the extremis hosted API.

All computation (embedding, search, RL scoring, KG, consolidation) runs server-side.
No local database. No 90 MB model download.

Usage:
    from extremis import HostedClient

    mem = HostedClient(api_key="extremis_sk_...")

    # Exact same API as Memory
    mem.remember("User is building a WhatsApp AI", conversation_id="conv_001")
    results = mem.recall("WhatsApp product")
    mem.report_outcome([r.memory.id for r in results], success=True)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from .types import (
    AttentionResult,
    EntityResult,
    EntityType,
    Memory,
    MemoryLayer,
    Observation,
    RecallResult,
)

_CLOUD_URL = "https://api.extremis.com"  # not yet live — self-host with extremis-server


class HostedClient:
    """
    Stateless HTTP client. Every call is a round-trip to a extremis server.

    Self-host:
        extremis-server serve --host 0.0.0.0 --port 8000
        extremis-server create-key --namespace alice
        mem = HostedClient(api_key="extremis_sk_...", base_url="http://localhost:8000")

    Cloud (coming soon — join the waitlist at github.com/ashwanijha04/extremis):
        mem = HostedClient(api_key="extremis_sk_...")

    Install: pip install "extremis[client]"
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = _CLOUD_URL,
        timeout: float = 30.0,
    ) -> None:
        try:
            import httpx
        except ImportError:
            raise ImportError("HostedClient requires httpx: pip install 'extremis[client]'") from None

        self._base = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self._base,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    # ── Core memory ─────────────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        role: str = "user",
        conversation_id: str = "default",
        metadata: Optional[dict] = None,
    ) -> None:
        self._post(
            "/v1/memories/remember",
            {
                "content": content,
                "role": role,
                "conversation_id": conversation_id,
                "metadata": metadata or {},
            },
        )

    def recall(
        self,
        query: str,
        limit: int = 10,
        layers: Optional[list[MemoryLayer]] = None,
        min_score: float = 0.0,
    ) -> list[RecallResult]:
        data = self._post(
            "/v1/memories/recall",
            {
                "query": query,
                "limit": limit,
                "layers": [layer.value for layer in layers] if layers else None,
                "min_score": min_score,
            },
        )
        return [RecallResult(**r) for r in data["results"]]

    def report_outcome(
        self,
        memory_ids: list[UUID],
        success: bool,
        weight: float = 1.0,
    ) -> None:
        self._post(
            "/v1/memories/report",
            {
                "memory_ids": [str(m) for m in memory_ids],
                "success": success,
                "weight": weight,
            },
        )

    def remember_now(
        self,
        content: str,
        layer: MemoryLayer,
        expires_at: Optional[datetime] = None,
        confidence: float = 0.9,
        metadata: Optional[dict] = None,
    ) -> Memory:
        data = self._post(
            "/v1/memories/store",
            {
                "content": content,
                "layer": layer.value,
                "confidence": confidence,
                "expires_at": expires_at.isoformat() if expires_at else None,
                "metadata": metadata or {},
            },
        )
        return Memory(**data)

    def observe(self, conversation_id: str = "default") -> list[Observation]:
        data = self._get("/v1/memories/observe", {"conversation_id": conversation_id})
        return [Observation(**o) for o in data["observations"]]

    def consolidate(self) -> dict:
        return self._post("/v1/memories/consolidate", {})

    # ── Knowledge graph ──────────────────────────────────────────────────────

    def kg_add_entity(self, name: str, type: EntityType, metadata: Optional[dict] = None):
        return self._post(
            "/v1/kg/write",
            {
                "operation": "add_entity",
                "name": name,
                "entity_type": type.value,
                "metadata": metadata or {},
            },
        )

    def kg_add_relationship(
        self,
        from_entity: str,
        to_entity: str,
        rel_type: str,
        weight: float = 1.0,
        metadata: Optional[dict] = None,
    ):
        return self._post(
            "/v1/kg/write",
            {
                "operation": "add_relationship",
                "from_entity": from_entity,
                "to_entity": to_entity,
                "rel_type": rel_type,
                "weight": weight,
                "metadata": metadata or {},
            },
        )

    def kg_add_attribute(self, entity: str, key: str, value: str):
        return self._post(
            "/v1/kg/write",
            {
                "operation": "add_attribute",
                "name": entity,
                "key": key,
                "value": value,
            },
        )

    def kg_query(self, name: str) -> Optional[EntityResult]:
        data = self._post("/v1/kg/query", {"name": name, "traverse_depth": 0})
        if data.get("result") is None:
            return None
        return EntityResult(**data["result"])

    def kg_traverse(self, name: str, depth: int = 2) -> list[EntityResult]:
        data = self._post("/v1/kg/query", {"name": name, "traverse_depth": depth})
        return [EntityResult(**r) for r in data.get("results", [])]

    # ── Attention ────────────────────────────────────────────────────────────

    def score_attention(
        self,
        message: str,
        sender: str = "",
        channel: str = "dm",
        owner_ids: Optional[set[str]] = None,
        allowlist: Optional[set[str]] = None,
        context: Optional[dict] = None,
    ) -> AttentionResult:
        ctx = context or {}
        data = self._get(
            "/v1/attention/score",
            {
                "message": message,
                "sender": sender,
                "channel": channel,
                "owner_ids": ",".join(owner_ids or []),
                "allowlist": ",".join(allowlist or []),
                "ongoing": str(ctx.get("ongoing", False)).lower(),
                "already_answered": str(ctx.get("already_answered", False)).lower(),
            },
        )
        return AttentionResult(**data)

    # ── HTTP helpers ─────────────────────────────────────────────────────────

    def _post(self, path: str, body: dict) -> dict:
        resp = self._http.post(path, json=body)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def _get(self, path: str, params: dict) -> dict:
        resp = self._http.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
