# n8n integration

Add persistent memory to any n8n workflow using the extremis community node.

## Install the node

In n8n: **Settings** → **Community Nodes** → **Install** → search for `n8n-nodes-extremis`

Or install manually:

```bash
npm install n8n-nodes-extremis
```

## Setup

1. **Deploy extremis server** — [Deploy to Render](../deployment/render.md) (free, takes 5 minutes)
2. **Get your API key** — copy from your server's Logs tab on first start
3. **Add credentials in n8n** — **Credentials** → **New** → **Extremis API** → paste server URL + key

## Operations

| Operation | What it does |
|-----------|-------------|
| **Remember** | Store a message or fact in memory |
| **Recall** | Search memories by query — returns ranked results |
| **Remember Now** | Write directly to a memory layer (semantic, procedural, etc.) |
| **Report Outcome** | Give +1/-1 feedback to improve future rankings |
| **KG: Add Entity** | Add a person, org, or project to the knowledge graph |
| **KG: Add Relationship** | Connect two entities (e.g. "Alice works_at Acme") |
| **KG: Query** | Look up an entity and all its connections |
| **Consolidate** | Distil conversation logs into structured memories (calls LLM) |

## Example workflow: Customer support with memory

```
Webhook (incoming message)
  → Extremis: Recall (query = message content)
  → HTTP Request: Call Claude API (inject recalled memories as context)
  → Extremis: Remember (store the conversation)
  → HTTP Request: Send reply
```

## Example workflow: Lead enrichment

```
HubSpot: New Contact
  → Extremis: KG Add Entity (name, type=person)
  → Extremis: KG Add Relationship (person → org)
  → Extremis: Remember Now (durable facts about the lead)
```
