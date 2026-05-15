# @extremis/sdk

TypeScript SDK for [extremis](https://github.com/ashwanijha04/extremis) — layered memory platform for AI agents, with production hallucination detection built in.

```bash
npm install @extremis/sdk
```

Zero runtime dependencies. Works on Node 18+, Bun, Deno, Cloudflare Workers, browsers.

## Quick start

```ts
import { ExtremisClient } from "@extremis/sdk";

const mem = new ExtremisClient({
  apiKey: "extremis_sk_...",
  baseUrl: "http://localhost:8000", // or your hosted endpoint
});

// Append to the conversation log
await mem.remember("User is building a WhatsApp AI product", {
  conversationId: "conv_001",
});

// Semantic recall — returns memories ranked by relevance × utility × recency
const results = await mem.recall("WhatsApp product");

for (const r of results) {
  console.log(r.memory.content, "→ effective confidence:", r.effective_confidence);
}
```

## Hallucination detection

Every `RecallResult` carries first-class typed verification + recommendation fields. Operators see *what's wrong* and *what to do about it* without rummaging through opaque metadata.

```ts
const results = await mem.recall("Where does the user work?");

for (const r of results) {
  // Hedging signal: confidence × layer_weight × temporal_decay
  if (r.effective_confidence && r.effective_confidence < 0.3) {
    console.warn("Stale memory — hedge the response:", r.memory.content);
  }

  // Actionable recommendations attached at write-time + read-time
  for (const rec of r.sources?.recommendations ?? []) {
    console.warn(`[${rec.severity}] ${rec.issue}`);
    console.warn("  Action:    ", rec.action);
    console.warn("  Suggestion:", rec.suggestion);
  }

  // Full provenance trail
  console.log("Came from conversation:", r.sources?.conversation_id);
  console.log("Source message IDs:", r.sources?.source_message_ids);
  console.log("Parent memories:", r.sources?.source_memory_ids);
}
```

**Detected issues** (each with `severity`, `action`, `suggestion`, `refs`):

| Issue | When |
|---|---|
| `claim_contradicts_source` | NLI / judge says the extracted memory contradicts the source conversation |
| `claim_unverifiable` | Judge says the claim isn't grounded in the source |
| `borderline_support` | Memory passed verification but only weakly |
| `memory_expired` | `validity_end` is in the past but memory is still surfacing |
| `surfacing_contradicted_memory` | A memory previously flagged CONTRADICTED is still being recalled |
| `stale_confidence` | `effective_confidence < 0.3` — old or low-trust memory |

## Full API surface

Mirrors the Python `HostedClient` exactly.

### Memory

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
  baseUrl: "https://api.extremis.com",
  timeoutMs: 30_000,
  maxRetries: 3,              // retry on 429 / 5xx
  baseDelayMs: 250,           // initial backoff, doubled per attempt
  maxDelayMs: 8_000,          // backoff cap
  fetch: customFetch,         // optional — defaults to globalThis.fetch
});
```

Exponential backoff with full jitter is enabled by default. Honors `Retry-After` headers on 429s. Set `maxRetries: 0` to disable.

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
    // 401 / 403 — bad or revoked API key
  } else if (err instanceof ExtremisRateLimitError) {
    // 429 after exhausting retries; err.retryAfterMs may be set
  } else if (err instanceof ExtremisNetworkError) {
    // connection refused, DNS failure, timeout, etc.
  } else if (err instanceof ExtremisError) {
    // any other HTTP failure — err.status, err.body
  }
}
```

## Self-hosting

```bash
pip install extremis[server]
extremis-server serve --host 0.0.0.0 --port 8000
extremis-server create-key --namespace alice --label "alice's key"
```

Then point the TypeScript client at it:

```ts
new ExtremisClient({ apiKey: "extremis_sk_...", baseUrl: "http://localhost:8000" });
```

## Links

- [extremis on GitHub](https://github.com/ashwanijha04/extremis)
- [Python package on PyPI](https://pypi.org/project/extremis/)
- [Full documentation](https://ashwanijha04.github.io/extremis/)

## License

MIT
