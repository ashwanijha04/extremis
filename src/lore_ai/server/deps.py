"""FastAPI dependencies — auth, per-namespace FridayMemory instances."""
from __future__ import annotations

import os
from typing import Annotated

from fastapi import Depends, HTTPException, Header, status

from ..api import FridayMemory
from ..config import Config
from .auth import KeyStore

# ── singletons ───────────────────────────────────────────────────────────────
_key_store: KeyStore | None = None
_instances: dict[str, FridayMemory] = {}
_server_config: Config | None = None


def init(key_store: KeyStore, server_config: Config) -> None:
    global _key_store, _server_config
    _key_store = key_store
    _server_config = server_config


# ── auth dependency ───────────────────────────────────────────────────────────

def _get_namespace(authorization: Annotated[str, Header()] = "") -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
    key = authorization.removeprefix("Bearer ").strip()
    if not key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Empty API key")
    assert _key_store is not None, "Server not initialised"
    namespace = _key_store.validate(key)
    if namespace is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or revoked API key")
    return namespace


Namespace = Annotated[str, Depends(_get_namespace)]


# ── per-namespace FridayMemory instance (cached) ──────────────────────────────

def get_memory(namespace: Namespace) -> FridayMemory:
    if namespace not in _instances:
        assert _server_config is not None
        cfg = _server_config.model_copy(update={"namespace": namespace})
        _instances[namespace] = FridayMemory(config=cfg)
    return _instances[namespace]


Memory = Annotated[FridayMemory, Depends(get_memory)]
