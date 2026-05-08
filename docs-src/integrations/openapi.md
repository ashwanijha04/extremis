# OpenAPI / REST API

extremis-server exposes a full REST API. Any tool that speaks HTTP can use it — no SDK required.

## OpenAPI spec

The live spec is always available at your server:

```
GET https://your-server/openapi.json
```

Or download the static spec:

- [openapi.json](https://ashwanijha04.github.io/extremis/openapi.json)

## Using with no-code tools

=== "Make (Integromat)"

    1. Add an **HTTP module**
    2. URL: `https://your-server.onrender.com/v1/memories/recall`
    3. Method: `POST`
    4. Headers: `Authorization: Bearer extremis_sk_...`
    5. Body (JSON): `{"query": "{{input}}", "limit": 5}`

=== "Zapier"

    Use the **Webhooks by Zapier** action:
    - Method: `POST`
    - URL: `https://your-server/v1/memories/remember`
    - Headers: `Authorization: Bearer extremis_sk_...`
    - Data: `{"content": "{{text}}", "conversation_id": "zapier"}`

=== "n8n"

    See the [n8n community node](n8n.md) for a visual interface without writing HTTP requests manually.

=== "Retool / AppSmith"

    Import the OpenAPI spec directly. Retool will generate all endpoints automatically.

## Authentication

All endpoints (except `/v1/health`) require:

```
Authorization: Bearer extremis_sk_...
```

## Core endpoints

```
POST /v1/memories/remember      Store a memory
POST /v1/memories/recall        Search memories
POST /v1/memories/store         Write to a specific layer
POST /v1/memories/report        Give +1/-1 feedback
POST /v1/memories/consolidate   Run LLM consolidation pass
GET  /v1/memories/observe       Priority-tagged log compression
POST /v1/kg/write               Add entity/relationship/attribute
POST /v1/kg/query               Query knowledge graph
POST /v1/attention/score        Score a message 0-100
GET  /v1/health                 Health check (no auth)
```

Full schema: [openapi.json](/extremis/openapi.json)
