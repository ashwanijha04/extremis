# Self-hosted server

Run extremis-server on your own infrastructure.

## Install

```bash
pip3.11 install "extremis[server,postgres]"
```

## Start

```bash
# Local dev
extremis-server serve

# With Postgres
EXTREMIS_STORE=postgres \
EXTREMIS_POSTGRES_URL=postgresql://user:pass@host/extremis \
extremis-server serve --host 0.0.0.0 --port 8000

# With Docker Compose (includes Postgres + pgvector)
docker compose up
```

## Create API keys

```bash
extremis-server create-key --namespace alice --label "alice prod"
# → extremis_sk_xxxx  (shown once, store it)

extremis-server list-keys
extremis-server list-keys --namespace alice
extremis-server revoke-key --key-hash abc123...
```

## Docker

```dockerfile
# Uses the included Dockerfile
docker build -t extremis-server .
docker run -p 8000:8000 \
  -e EXTREMIS_STORE=postgres \
  -e EXTREMIS_POSTGRES_URL=postgresql://... \
  extremis-server
```

## Docker Compose

```bash
docker compose up
```

Starts:
- `extremis` web service on port 8000
- `extremis-db` Postgres with pgvector on port 5432

## Health check

```bash
curl https://your-server:8000/v1/health
# → {"status":"ok","version":"0.1.0"}
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/health` | Health check |
| POST | `/v1/memories/remember` | Append to log + episodic store |
| POST | `/v1/memories/recall` | Semantic search |
| POST | `/v1/memories/report` | RL signal |
| POST | `/v1/memories/store` | Direct write to any layer |
| POST | `/v1/memories/consolidate` | LLM consolidation pass |
| GET | `/v1/memories/observe` | Priority-tagged log compression |
| POST | `/v1/kg/write` | Add entity/relationship/attribute |
| POST | `/v1/kg/query` | Query + BFS traverse |
| POST | `/v1/attention/score` | 0–100 message priority |
| GET | `/v1/usage` | Usage info (namespace) |

All endpoints except `/v1/health` require `Authorization: Bearer extremis_sk_...`.
