"""
Live tenant-isolation test for the hosted server.

Mints two API keys against the running server (two distinct namespaces),
writes a unique marker through each, and asserts neither namespace's recall
surfaces the other's marker.

Skipped unless EXTREMIS_LIVE_URL is set. Uses the server's CLI to mint keys
directly against its keystore so the test stays self-contained.

Run locally:
    extremis-server serve --port 8000           # in another shell
    EXTREMIS_LIVE_URL=http://localhost:8000 \\
        pytest tests/test_tenant_isolation_live.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid

import pytest

pytest.importorskip("httpx", reason="httpx not installed")

from extremis import HostedClient

LIVE_URL = os.environ.get("EXTREMIS_LIVE_URL")

pytestmark = pytest.mark.skipif(
    not LIVE_URL,
    reason="set EXTREMIS_LIVE_URL to run live tests",
)


def _mint_key(namespace: str) -> str:
    """Mint an API key against the server's keystore via the CLI."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "extremis.server.app",
            "create-key",
            "--namespace",
            namespace,
            "--label",
            "isolation-test",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    # Output format: blank line, "API key created for namespace 'X':", blank line, "  <key>"
    for line in proc.stdout.splitlines():
        s = line.strip()
        if s.startswith("extremis_sk_"):
            return s
    raise AssertionError(f"could not parse key from CLI output: {proc.stdout!r}")


def test_two_tenants_cannot_see_each_others_memories() -> None:
    ns_a = f"extremis:{uuid.uuid4()}"
    ns_b = f"extremis:{uuid.uuid4()}"
    key_a = _mint_key(ns_a)
    key_b = _mint_key(ns_b)

    marker_a = f"isolation-A-{int(time.time() * 1000)}"
    marker_b = f"isolation-B-{int(time.time() * 1000)}"

    with HostedClient(api_key=key_a, base_url=LIVE_URL) as mem_a:
        mem_a.remember(f"Tenant A secret data. Token: {marker_a}")
    with HostedClient(api_key=key_b, base_url=LIVE_URL) as mem_b:
        mem_b.remember(f"Tenant B secret data. Token: {marker_b}")

    with HostedClient(api_key=key_a, base_url=LIVE_URL) as mem_a:
        results_a = mem_a.recall("secret data", limit=20)
    with HostedClient(api_key=key_b, base_url=LIVE_URL) as mem_b:
        results_b = mem_b.recall("secret data", limit=20)

    contents_a = [r.memory.content for r in results_a]
    contents_b = [r.memory.content for r in results_b]

    assert any(marker_a in c for c in contents_a), f"tenant A missing its own marker: {contents_a}"
    assert not any(marker_b in c for c in contents_a), f"LEAK: tenant A saw tenant B's data. A's results: {contents_a}"

    assert any(marker_b in c for c in contents_b), f"tenant B missing its own marker: {contents_b}"
    assert not any(marker_a in c for c in contents_b), f"LEAK: tenant B saw tenant A's data. B's results: {contents_b}"
