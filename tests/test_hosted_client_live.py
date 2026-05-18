"""
Live roundtrip test for HostedClient.

Skipped unless EXTREMIS_LIVE_URL and EXTREMIS_API_KEY are both set in the
environment, so CI stays green when no server is reachable.

Run locally:
    extremis-server serve --port 8000          # in another shell
    EXTREMIS_LIVE_URL=http://localhost:8000 \\
    EXTREMIS_API_KEY=extremis_sk_... \\
        pytest tests/test_hosted_client_live.py -v
"""

from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("httpx", reason="httpx not installed")

from extremis import HostedClient

LIVE_URL = os.environ.get("EXTREMIS_LIVE_URL")
LIVE_KEY = os.environ.get("EXTREMIS_API_KEY")

pytestmark = pytest.mark.skipif(
    not (LIVE_URL and LIVE_KEY),
    reason="set EXTREMIS_LIVE_URL and EXTREMIS_API_KEY to run live tests",
)


def test_remember_recall_roundtrip() -> None:
    marker = f"hosted-live-marker-{int(time.time() * 1000)}"
    phrase = f"HostedClient live regression test. Token: {marker}"

    with HostedClient(api_key=LIVE_KEY, base_url=LIVE_URL) as mem:
        mem.remember(phrase, conversation_id="hosted-live-roundtrip")
        results = mem.recall("HostedClient live regression", limit=10)

    contents = [r.memory.content for r in results]
    assert any(marker in c for c in contents), f"marker {marker!r} not surfaced in recall; got {contents!r}"
