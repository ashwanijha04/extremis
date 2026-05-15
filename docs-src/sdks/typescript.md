# TypeScript SDK

`@extremis/sdk` is a zero-runtime-dependency client that talks to an extremis server over HTTP. Works on Node 18+, Bun, Deno, Cloudflare Workers, and browsers (with the appropriate CORS setup on the server).

## Install

```bash
npm install @extremis/sdk
# or
pnpm add @extremis/sdk
# or
bun add @extremis/sdk
```

## Quick start

```ts
import { ExtremisClient } from "@extremis/sdk";

const mem = new ExtremisClient({
  apiKey: "extremis_sk_...",
  baseUrl: "http://localhost:8000",
});

await mem.remember("User is building a WhatsApp AI product", {
  conversationId: "conv_001",
});

const results = await mem.recall("WhatsApp product");
for (const r of results) {
  console.log(r.memory.content, r.effective_confidence);
}
```

## Hallucination detection

Every `RecallResult` carries typed verification + recommendation fields. The signals show up in intellisense rather than being buried in opaque metadata.

```ts
const results = await mem.recall("Where does the user work?");

for (const r of results) {
  // Hedging signal: confidence × layer_weight × temporal_decay
  if (r.effective_confidence && r.effective_confidence < 0.3) {
    console.warn("Stale — hedge:", r.memory.content);
  }

  // Actionable recommendations
  for (const rec of r.sources?.recommendations ?? []) {
    console.warn(`[${rec.severity}] ${rec.issue}`);
    console.warn("  Action:    ", rec.action);
    console.warn("  Suggestion:", rec.suggestion);
  }
}
```

Detected issue types match the Python side exactly:

| Issue | Severity | When |
|---|---|---|
| `claim_contradicts_source` | high | NLI / judge says memory contradicts the source conversation |
| `claim_unverifiable` | medium | Judge says the claim isn't grounded |
| `borderline_support` | low | Passed verification but only weakly |
| `memory_expired` | high | `validity_end` in the past but still surfacing |
| `surfacing_contradicted_memory` | high | Previously-flagged contradicted memory recalled |
| `stale_confidence` | medium | `effective_confidence < 0.3` |

See [Hallucination detection](../concepts/hallucination-detection.md) for how these are computed.

## API reference

The TypeScript SDK mirrors Python's `HostedClient` exactly. Method names are camelCased (`rememberNow` instead of `remember_now`), options are passed as objects.

### Memory core

```ts
await mem.remember(content, { role?, conversationId?, metadata? });
await mem.recall(query, { limit?, layers?, minScore? });
await mem.rememberNow(content, { layer, expiresAt?, confidence?, metadata? });
await mem.reportOutcome(memoryIds, success, weight?);
await mem.consolidate();
await mem.compact(layer?);
await mem.observe(conversationId?);
```

### Knowledge graph

```ts
await mem.kgAddEntity(name, type, metadata?);
await mem.kgAddRelationship(from, to, relType, weight?, metadata?);
await mem.kgAddAttribute(entity, key, value);
await mem.kgQuery(name);
await mem.kgTraverse(name, depth?);
```

### Attention scoring

```ts
await mem.scoreAttention(message, {
  sender?, channel?, ownerIds?, allowlist?, ongoing?, alreadyAnswered?,
});
```

## Configuration

```ts
new ExtremisClient({
  apiKey: "extremis_sk_...",
  baseUrl: "https://api.example.com",
  timeoutMs: 30_000,
  maxRetries: 3,         // retry on 429 / 5xx
  baseDelayMs: 250,      // initial backoff
  maxDelayMs: 8_000,     // backoff cap
  fetch: customFetch,    // optional — defaults to globalThis.fetch
});
```

The SDK applies **exponential backoff with full jitter** on retryable failures (429, 5xx). `Retry-After` headers are honored when present. Set `maxRetries: 0` to disable retries entirely.

## Error handling

```ts
import {
  ExtremisError,
  ExtremisAuthError,
  ExtremisRateLimitError,
  ExtremisNetworkError,
} from "@extremis/sdk";

try {
  await mem.recall("query");
} catch (err) {
  if (err instanceof ExtremisAuthError) {
    // 401 / 403 — bad or revoked key
  } else if (err instanceof ExtremisRateLimitError) {
    // 429 after retries; err.retryAfterMs may be set
  } else if (err instanceof ExtremisNetworkError) {
    // connection refused, DNS, timeout
  } else if (err instanceof ExtremisError) {
    // any other HTTP failure — err.status, err.body
  }
}
```

## Runtimes

| Runtime | Supported | Notes |
|---|---|---|
| Node 18+ | ✅ | Uses native `fetch`. |
| Bun | ✅ | Native fetch. |
| Deno | ✅ | Native fetch. |
| Cloudflare Workers | ✅ | Native fetch. |
| Vercel Edge | ✅ | Native fetch. |
| Browsers | ✅ | Set CORS on the server (`EXTREMIS_CORS_ORIGINS`). |

## Self-hosting the server

```bash
pip install "extremis[server]"
extremis-server serve --host 0.0.0.0 --port 8000
extremis-server create-key --namespace alice --label "alice's key"
```

Then in your TypeScript code:

```ts
const mem = new ExtremisClient({
  apiKey: "extremis_sk_...",  // from create-key output
  baseUrl: "http://localhost:8000",
});
```

## Links

- [npm package](https://www.npmjs.com/package/@extremis/sdk)
- [Source on GitHub](https://github.com/ashwanijha04/extremis/tree/main/sdk/typescript)
- [Python SDK comparison](../api/hosted-client.md)
- [OpenAPI spec](../integrations/openapi.md)
