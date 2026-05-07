"""
Hosted API server tests — uses FastAPI TestClient (no real HTTP, no LLM).
All memory operations use a mocked embedder so sentence-transformers isn't needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx", reason="httpx not installed")

from fastapi.testclient import TestClient

from extremis.server.app import create_app
from extremis.server.auth import KeyStore
import extremis.server.deps as deps


@pytest.fixture
def key_store(tmp_path):
    store = KeyStore(str(tmp_path / "keys.db"))
    yield store
    store.close()


@pytest.fixture
def api_key(key_store):
    return key_store.create("test_ns", "test key")


@pytest.fixture
def mock_embedder():
    e = MagicMock()
    e.embed.return_value = [0.1] * 384
    e.embed_batch.return_value = [[0.1] * 384]
    e.dim = 384
    return e


@pytest.fixture
def client(tmp_path, key_store, mock_embedder):
    from extremis.config import Config
    server_cfg = Config(
        extremis_home=str(tmp_path),
        log_dir=str(tmp_path / "log"),
        local_db_path=str(tmp_path / "local.db"),
        namespace="test_ns",
    )

    # Patch the embedder so no model is downloaded
    with patch("extremis.api._build_embedder", return_value=mock_embedder):
        deps.init(key_store, server_cfg)
        deps._instances.clear()
        app = create_app()
        with TestClient(app) as c:
            yield c
    deps._instances.clear()


@pytest.fixture
def auth(api_key):
    return {"Authorization": f"Bearer {api_key}"}


class TestAuth:
    def test_missing_key_returns_401(self, client):
        resp = client.post("/v1/memories/remember", json={"content": "test"})
        assert resp.status_code == 401

    def test_invalid_key_returns_401(self, client):
        resp = client.post(
            "/v1/memories/remember",
            json={"content": "test"},
            headers={"Authorization": "Bearer extremis_sk_invalid"},
        )
        assert resp.status_code == 401

    def test_valid_key_accepted(self, client, auth):
        resp = client.post("/v1/memories/remember", json={"content": "test"}, headers=auth)
        assert resp.status_code == 204


class TestMemoryEndpoints:
    def test_remember_and_recall(self, client, auth):
        client.post("/v1/memories/remember", json={
            "content": "User is building a WhatsApp AI",
            "conversation_id": "c1",
        }, headers=auth)

        resp = client.post("/v1/memories/recall", json={"query": "WhatsApp", "limit": 5}, headers=auth)
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert any("WhatsApp" in r["memory"]["content"] for r in results)

    def test_store_returns_memory(self, client, auth):
        resp = client.post("/v1/memories/store", json={
            "content": "User is a Python developer",
            "layer": "semantic",
            "confidence": 0.95,
        }, headers=auth)
        assert resp.status_code == 200
        memory = resp.json()
        assert memory["content"] == "User is a Python developer"
        assert memory["layer"] == "semantic"
        assert memory["confidence"] == pytest.approx(0.95)

    def test_report_outcome(self, client, auth):
        store_resp = client.post("/v1/memories/store", json={
            "content": "Concise answers work well",
            "layer": "procedural",
        }, headers=auth)
        memory_id = store_resp.json()["id"]

        resp = client.post("/v1/memories/report", json={
            "memory_ids": [memory_id],
            "success": True,
            "weight": 2.0,
        }, headers=auth)
        assert resp.status_code == 204

    def test_observe(self, client, auth):
        client.post("/v1/memories/remember", json={
            "content": "We decided to launch tomorrow",
            "conversation_id": "obs_test",
        }, headers=auth)
        resp = client.get("/v1/memories/observe", params={"conversation_id": "obs_test"}, headers=auth)
        assert resp.status_code == 200
        obs = resp.json()["observations"]
        assert len(obs) >= 1


class TestKGEndpoints:
    def test_add_entity_and_query(self, client, auth):
        client.post("/v1/kg/write", json={
            "operation": "add_entity",
            "name": "Alice",
            "entity_type": "person",
        }, headers=auth)
        resp = client.post("/v1/kg/query", json={"name": "Alice", "traverse_depth": 0}, headers=auth)
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result is not None
        assert result["entity"]["name"] == "Alice"

    def test_add_attribute(self, client, auth):
        client.post("/v1/kg/write", json={
            "operation": "add_entity", "name": "Bob", "entity_type": "person",
        }, headers=auth)
        client.post("/v1/kg/write", json={
            "operation": "add_attribute", "name": "Bob", "key": "timezone", "value": "UTC",
        }, headers=auth)
        resp = client.post("/v1/kg/query", json={"name": "Bob"}, headers=auth)
        attrs = {a["key"]: a["value"] for a in resp.json()["result"]["attributes"]}
        assert attrs["timezone"] == "UTC"

    def test_unknown_entity_returns_null(self, client, auth):
        resp = client.post("/v1/kg/query", json={"name": "Nobody"}, headers=auth)
        assert resp.status_code == 200
        assert resp.json()["result"] is None


class TestHealthEndpoint:
    def test_health_no_auth_needed(self, client):
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestKeyStore:
    def test_create_and_validate(self, key_store):
        key = key_store.create("alice", "test")
        ns = key_store.validate(key)
        assert ns == "alice"

    def test_invalid_key_returns_none(self, key_store):
        assert key_store.validate("extremis_sk_invalid") is None

    def test_revoke(self, key_store):
        from extremis.server.auth import hash_key
        key = key_store.create("alice")
        key_store.revoke(hash_key(key))
        assert key_store.validate(key) is None

    def test_call_count_increments(self, key_store):
        key = key_store.create("alice")
        key_store.validate(key)
        key_store.validate(key)
        keys = key_store.list_keys("alice")
        assert keys[0]["call_count"] == 2
