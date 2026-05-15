# SDKs

Extremis ships first-class SDKs in multiple languages. All SDKs talk to the same `/v1/*` HTTP API, return the same shapes, and expose the same hallucination-detection signals — `effective_confidence`, `verification.verdict`, and per-issue `recommendations` — as first-class typed fields.

## Available SDKs

| Language | Package | Source | Status |
|---|---|---|---|
| **Python** | [`pip install extremis`](https://pypi.org/project/extremis/) | [`src/extremis/`](https://github.com/ashwanijha04/extremis/tree/main/src/extremis) | stable |
| **TypeScript** | [`npm install @extremis/sdk`](https://www.npmjs.com/package/@extremis/sdk) | [`sdk/typescript/`](https://github.com/ashwanijha04/extremis/tree/main/sdk/typescript) | stable |

The Python SDK also includes a local-only mode (no server required) — `Extremis()` for fully embedded use, `HostedClient` for remote API access. The TypeScript SDK is remote-only.

## Choosing an SDK

- **Building a Python agent** — use `Extremis()` directly for the simplest setup. Switch to `HostedClient` when you want to share memory across processes.
- **Building a JS/TS agent, MCP server, edge function, or web frontend** — use [`@extremis/sdk`](typescript.md).
- **Need a language we don't have** — the `/v1/*` HTTP API is documented; the TS SDK is a good reference for how to wrap it. PRs welcome.

## Shared design

Every SDK aims for the same ergonomics:

```ts
// TypeScript
await mem.remember("User builds WhatsApp AI");
const results = await mem.recall("WhatsApp");
```

```python
# Python
mem.remember("User builds WhatsApp AI")
results = mem.recall("WhatsApp")
```

Both return memories carrying:

- `effective_confidence` — `confidence × layer_weight × temporal_decay`, the hedging signal
- `sources.verification` — write-time faithfulness verdict (SUPPORTED / CONTRADICTED / UNVERIFIABLE)
- `sources.recommendations` — actionable items for any detected issues

See [Hallucination detection](../concepts/hallucination-detection.md) for the full story on those signals.
