"""
lore-ai hosted API server.

Run locally:
    lore-server

Run with custom config:
    LORE_STORE=postgres LORE_POSTGRES_URL=postgresql://... lore-server

Create an API key:
    lore-server create-key --namespace alice --label "alice's dev key"

Docker:
    docker build -t lore-ai-server .
    docker run -p 8000:8000 -e LORE_STORE=postgres -e LORE_POSTGRES_URL=... lore-ai-server
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .auth import KeyStore
from .deps import init
from .routes import memories, kg, health

log = logging.getLogger(__name__)

_SERVER_HOME = os.environ.get("LORE_SERVER_HOME", "~/.lore/server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    home = Path(_SERVER_HOME).expanduser()
    home.mkdir(parents=True, exist_ok=True)

    from ..config import Config
    server_cfg = Config()
    key_store = KeyStore(str(home / "keys.db"))
    init(key_store, server_cfg)

    log.info("lore-ai server started  store=%s  home=%s", server_cfg.store, str(home))
    yield
    key_store.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="lore-ai API",
        description="Hosted memory layer for AI agents",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("LORE_CORS_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(memories.router, prefix="/v1/memories")
    app.include_router(kg.router,       prefix="/v1/kg")
    app.include_router(health.router,   prefix="/v1")

    return app


app = create_app()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse, sys

    parser = argparse.ArgumentParser(prog="lore-server")
    sub = parser.add_subparsers(dest="cmd")

    # serve (default)
    serve_p = sub.add_parser("serve", help="Start the API server")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--reload", action="store_true")

    # create-key
    key_p = sub.add_parser("create-key", help="Generate an API key")
    key_p.add_argument("--namespace", required=True)
    key_p.add_argument("--label", default="")

    # list-keys
    list_p = sub.add_parser("list-keys", help="List API keys")
    list_p.add_argument("--namespace", default="")

    # revoke-key
    rev_p = sub.add_parser("revoke-key", help="Revoke an API key by hash")
    rev_p.add_argument("--key-hash", required=True)

    args = parser.parse_args()

    if args.cmd == "create-key":
        store = KeyStore(str(Path(_SERVER_HOME).expanduser() / "keys.db"))
        key = store.create(args.namespace, args.label)
        store.close()
        print(f"\nAPI key created for namespace '{args.namespace}':")
        print(f"\n  {key}\n")
        print("Store this somewhere safe — it won't be shown again.\n")
        return

    if args.cmd == "list-keys":
        store = KeyStore(str(Path(_SERVER_HOME).expanduser() / "keys.db"))
        keys = store.list_keys(args.namespace or None)
        store.close()
        if not keys:
            print("No keys found.")
            return
        print(f"{'namespace':<20} {'label':<20} {'calls':>8}  {'last_used':<26}  {'hash[:12]'}")
        print("-" * 90)
        for k in keys:
            status = " [revoked]" if k["revoked"] else ""
            print(f"{k['namespace']:<20} {k['label']:<20} {k['call_count']:>8}  {(k['last_used'] or 'never'):<26}  {k['key_hash'][:12]}{status}")
        return

    if args.cmd == "revoke-key":
        store = KeyStore(str(Path(_SERVER_HOME).expanduser() / "keys.db"))
        ok = store.revoke(args.key_hash)
        store.close()
        print("Revoked." if ok else "Key not found.")
        return

    # default: serve
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install 'lore-ai[server]'")
        sys.exit(1)

    host = getattr(args, "host", "0.0.0.0")
    port = getattr(args, "port", 8000)
    reload = getattr(args, "reload", False)
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("lore_ai.server.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
