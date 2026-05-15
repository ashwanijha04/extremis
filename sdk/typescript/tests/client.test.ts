/**
 * SDK tests with a fake fetch — no network, no server required.
 */

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { ExtremisClient } from "../src/client.js";
import {
  ExtremisAuthError,
  ExtremisError,
  ExtremisRateLimitError,
} from "../src/errors.js";

type FetchCall = {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: unknown;
};

function makeFetch(
  handlers: Array<{ status: number; body?: unknown; headers?: Record<string, string> }>,
): { fetch: typeof fetch; calls: FetchCall[] } {
  const calls: FetchCall[] = [];
  let i = 0;
  const fetchImpl: typeof fetch = async (input, init) => {
    const url = typeof input === "string" ? input : (input as URL).toString();
    const method = init?.method ?? "GET";
    // Normalize to lowercase keys — matches fetch's case-insensitive spec
    const headers: Record<string, string> = {};
    const hraw = init?.headers ?? {};
    if (hraw instanceof Headers) {
      hraw.forEach((v, k) => (headers[k.toLowerCase()] = v));
    } else if (Array.isArray(hraw)) {
      for (const [k, v] of hraw) headers[k.toLowerCase()] = v;
    } else {
      for (const [k, v] of Object.entries(hraw as Record<string, string>)) {
        headers[k.toLowerCase()] = v;
      }
    }
    let body: unknown = init?.body;
    if (typeof body === "string") {
      try {
        body = JSON.parse(body);
      } catch {
        /* leave as string */
      }
    }
    calls.push({ url, method, headers, body });

    const handler = handlers[i] ?? handlers[handlers.length - 1];
    i++;
    const responseHeaders = new Headers(handler?.headers ?? {});
    return new Response(
      handler?.body !== undefined ? JSON.stringify(handler.body) : null,
      { status: handler?.status ?? 200, headers: responseHeaders },
    );
  };
  return { fetch: fetchImpl, calls };
}

const baseOpts = {
  apiKey: "extremis_sk_test",
  baseUrl: "https://api.example.com",
  baseDelayMs: 1,
  maxDelayMs: 5,
};

// ── Construction ─────────────────────────────────────────────────────────

describe("ExtremisClient construction", () => {
  it("throws without apiKey", () => {
    assert.throws(
      () => new ExtremisClient({ apiKey: "" } as unknown as { apiKey: string }),
      /apiKey is required/,
    );
  });

  it("normalizes trailing slash in baseUrl", async () => {
    const { fetch, calls } = makeFetch([{ status: 204 }]);
    const c = new ExtremisClient({ ...baseOpts, baseUrl: "https://api.example.com/", fetch });
    await c.remember("hi");
    assert.equal(calls[0]?.url, "https://api.example.com/v1/memories/remember");
  });
});

// ── remember ─────────────────────────────────────────────────────────────

describe("remember", () => {
  it("posts content + default role to /v1/memories/remember", async () => {
    const { fetch, calls } = makeFetch([{ status: 204 }]);
    const c = new ExtremisClient({ ...baseOpts, fetch });
    await c.remember("hello", { conversationId: "c1" });

    assert.equal(calls.length, 1);
    const call = calls[0]!;
    assert.equal(call.method, "POST");
    assert.equal(call.headers.authorization, "Bearer extremis_sk_test");
    assert.deepEqual(call.body, {
      content: "hello",
      role: "user",
      conversation_id: "c1",
      metadata: {},
    });
  });
});

// ── recall ───────────────────────────────────────────────────────────────

describe("recall", () => {
  it("returns typed RecallResults with verification + recommendations", async () => {
    const { fetch } = makeFetch([
      {
        status: 200,
        body: {
          results: [
            {
              memory: {
                id: "mem-1",
                namespace: "default",
                layer: "semantic",
                content: "User likes Python",
                score: 0,
                confidence: 0.95,
                metadata: {
                  verification: {
                    score: 0.94,
                    verdict: "SUPPORTED",
                    method: "nli",
                  },
                  recommendations: [],
                },
                source_memory_ids: [],
                validity_start: "2025-01-01T00:00:00Z",
                created_at: "2025-01-01T00:00:00Z",
                access_count: 0,
                do_not_consolidate: false,
              },
              relevance: 0.9,
              final_rank: 0.85,
              reason: "",
              effective_confidence: 0.78,
              sources: {
                conversation_id: "c1",
                source_message_ids: ["m1"],
                source_memory_ids: [],
                layer: "semantic",
                created_at: "2025-01-01T00:00:00Z",
                verification: {
                  score: 0.94,
                  verdict: "SUPPORTED",
                  method: "nli",
                },
                consistency: null,
                recommendations: [
                  {
                    issue: "borderline_support",
                    severity: "low",
                    action: "monitor",
                    suggestion: "split template",
                    refs: {},
                  },
                ],
              },
            },
          ],
        },
      },
    ]);
    const c = new ExtremisClient({ ...baseOpts, fetch });
    const out = await c.recall("python", { limit: 5 });

    assert.equal(out.length, 1);
    const r = out[0]!;
    assert.equal(r.memory.content, "User likes Python");
    assert.equal(r.effective_confidence, 0.78);
    assert.equal(r.sources?.verification?.verdict, "SUPPORTED");
    assert.equal(r.sources?.recommendations[0]?.issue, "borderline_support");
  });

  it("sends layers and min_score in body", async () => {
    const { fetch, calls } = makeFetch([{ status: 200, body: { results: [] } }]);
    const c = new ExtremisClient({ ...baseOpts, fetch });
    await c.recall("q", { layers: ["semantic", "episodic"], minScore: 0.3 });
    assert.deepEqual(calls[0]?.body, {
      query: "q",
      limit: 10,
      layers: ["semantic", "episodic"],
      min_score: 0.3,
    });
  });

  it("returns empty array when server returns empty results", async () => {
    const { fetch } = makeFetch([{ status: 200, body: { results: [] } }]);
    const c = new ExtremisClient({ ...baseOpts, fetch });
    const out = await c.recall("q");
    assert.deepEqual(out, []);
  });
});

// ── rememberNow ──────────────────────────────────────────────────────────

describe("rememberNow", () => {
  it("converts Date to ISO string for expires_at", async () => {
    const { fetch, calls } = makeFetch([
      {
        status: 200,
        body: {
          id: "mem-2",
          namespace: "default",
          layer: "working",
          content: "x",
          score: 0,
          confidence: 0.9,
          metadata: {},
          source_memory_ids: [],
          validity_start: "2025-01-01T00:00:00Z",
          created_at: "2025-01-01T00:00:00Z",
          access_count: 0,
          do_not_consolidate: false,
        },
      },
    ]);
    const c = new ExtremisClient({ ...baseOpts, fetch });
    const expires = new Date("2025-02-01T00:00:00Z");
    await c.rememberNow("x", { layer: "working", expiresAt: expires });
    assert.equal((calls[0]!.body as Record<string, unknown>).expires_at, expires.toISOString());
  });
});

// ── auth + errors ────────────────────────────────────────────────────────

describe("errors", () => {
  it("throws ExtremisAuthError on 401", async () => {
    const { fetch } = makeFetch([{ status: 401, body: { detail: "bad key" } }]);
    const c = new ExtremisClient({ ...baseOpts, fetch, maxRetries: 0 });
    await assert.rejects(
      () => c.remember("hi"),
      (err: unknown) => err instanceof ExtremisAuthError && err.status === 401,
    );
  });

  it("throws ExtremisError on 400 (non-retryable)", async () => {
    const { fetch, calls } = makeFetch([{ status: 400, body: { detail: "bad request" } }]);
    const c = new ExtremisClient({ ...baseOpts, fetch });
    await assert.rejects(
      () => c.remember("hi"),
      (err: unknown) => err instanceof ExtremisError && err.status === 400,
    );
    // 400 must NOT retry
    assert.equal(calls.length, 1);
  });
});

// ── retries ──────────────────────────────────────────────────────────────

describe("retries", () => {
  it("retries 429 then succeeds", async () => {
    const { fetch, calls } = makeFetch([
      { status: 429, body: { detail: "rate limited" }, headers: { "Retry-After": "0" } },
      { status: 204 },
    ]);
    const c = new ExtremisClient({ ...baseOpts, fetch, maxRetries: 2 });
    await c.remember("hi");
    assert.equal(calls.length, 2);
  });

  it("retries 503 then succeeds", async () => {
    const { fetch, calls } = makeFetch([
      { status: 503 },
      { status: 200, body: { results: [] } },
    ]);
    const c = new ExtremisClient({ ...baseOpts, fetch, maxRetries: 1 });
    await c.recall("q");
    assert.equal(calls.length, 2);
  });

  it("throws ExtremisRateLimitError after exhausting retries on 429", async () => {
    const { fetch, calls } = makeFetch([
      { status: 429, headers: { "Retry-After": "0" } },
      { status: 429, headers: { "Retry-After": "0" } },
      { status: 429, headers: { "Retry-After": "0" } },
    ]);
    const c = new ExtremisClient({ ...baseOpts, fetch, maxRetries: 2 });
    await assert.rejects(
      () => c.remember("hi"),
      (err: unknown) => err instanceof ExtremisRateLimitError,
    );
    assert.equal(calls.length, 3);
  });

  it("does not retry when maxRetries=0", async () => {
    const { fetch, calls } = makeFetch([{ status: 503 }]);
    const c = new ExtremisClient({ ...baseOpts, fetch, maxRetries: 0 });
    await assert.rejects(() => c.remember("hi"));
    assert.equal(calls.length, 1);
  });
});

// ── reportOutcome ────────────────────────────────────────────────────────

describe("reportOutcome", () => {
  it("posts memory_ids + success + weight", async () => {
    const { fetch, calls } = makeFetch([{ status: 204 }]);
    const c = new ExtremisClient({ ...baseOpts, fetch });
    await c.reportOutcome(["mem-a", "mem-b"], true, 2.0);
    assert.deepEqual(calls[0]?.body, {
      memory_ids: ["mem-a", "mem-b"],
      success: true,
      weight: 2.0,
    });
  });
});

// ── KG + attention ────────────────────────────────────────────────────────

describe("kgQuery", () => {
  it("returns null when server says no match", async () => {
    const { fetch } = makeFetch([{ status: 200, body: { result: null } }]);
    const c = new ExtremisClient({ ...baseOpts, fetch });
    const out = await c.kgQuery("unknown");
    assert.equal(out, null);
  });
});

describe("scoreAttention", () => {
  it("sends params as query string on POST", async () => {
    const { fetch, calls } = makeFetch([
      { status: 200, body: { score: 80, level: "full", reason: "", breakdown: {} } },
    ]);
    const c = new ExtremisClient({ ...baseOpts, fetch });
    await c.scoreAttention("hello", { sender: "alice", ownerIds: ["a", "b"] });
    const url = calls[0]?.url ?? "";
    assert.ok(url.includes("/v1/attention/score?"), `expected query string, got ${url}`);
    assert.ok(url.includes("message=hello"));
    assert.ok(url.includes("owner_ids=a%2Cb"));
    assert.equal(calls[0]?.method, "POST");
  });
});
