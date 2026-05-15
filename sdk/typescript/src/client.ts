/**
 * ExtremisClient — TypeScript SDK with full HostedClient parity.
 *
 * Verification + recommendations are first-class typed fields, so
 * production hallucination-detection signals show up in intellisense
 * rather than being buried in untyped metadata.
 *
 * Usage:
 *   import { ExtremisClient } from "@extremis/sdk";
 *   const mem = new ExtremisClient({ apiKey: "extremis_sk_..." });
 *   await mem.remember("User is building a WhatsApp AI");
 *   const results = await mem.recall("WhatsApp product");
 *   for (const r of results) {
 *     if (r.sources?.recommendations.length) {
 *       console.warn(r.sources.recommendations);
 *     }
 *   }
 */

import { Transport } from "./transport.js";
import type {
  AttentionResult,
  ClientOptions,
  CompactionResult,
  ConsolidationResult,
  EntityResult,
  EntityType,
  Memory,
  MemoryLayer,
  Observation,
  RecallResult,
} from "./types.js";

const DEFAULT_BASE_URL = "https://api.extremis.com";

export interface RememberOptions {
  role?: string;
  conversationId?: string;
  metadata?: Record<string, unknown>;
}

export interface RecallOptions {
  limit?: number;
  layers?: MemoryLayer[];
  minScore?: number;
}

export interface RememberNowOptions {
  layer: MemoryLayer;
  /** ISO 8601 timestamp or Date. */
  expiresAt?: string | Date;
  confidence?: number;
  metadata?: Record<string, unknown>;
}

export interface AttentionOptions {
  sender?: string;
  channel?: string;
  ownerIds?: Iterable<string>;
  allowlist?: Iterable<string>;
  ongoing?: boolean;
  alreadyAnswered?: boolean;
}

export class ExtremisClient {
  private readonly transport: Transport;

  constructor(options: ClientOptions) {
    if (!options.apiKey) {
      throw new Error("ExtremisClient: apiKey is required");
    }
    const fetchImpl =
      options.fetch ?? (typeof globalThis !== "undefined" ? globalThis.fetch : undefined);
    if (!fetchImpl) {
      throw new Error(
        "ExtremisClient: no fetch implementation found. Provide options.fetch or run on Node 18+.",
      );
    }
    this.transport = new Transport({
      apiKey: options.apiKey,
      baseUrl: (options.baseUrl ?? DEFAULT_BASE_URL).replace(/\/$/, ""),
      timeoutMs: options.timeoutMs ?? 30_000,
      maxRetries: options.maxRetries ?? 3,
      baseDelayMs: options.baseDelayMs ?? 250,
      maxDelayMs: options.maxDelayMs ?? 8_000,
      fetch: fetchImpl.bind(globalThis),
    });
  }

  // ── Core memory ──────────────────────────────────────────────────────

  async remember(content: string, options: RememberOptions = {}): Promise<void> {
    await this.transport.postEmpty("/v1/memories/remember", {
      content,
      role: options.role ?? "user",
      conversation_id: options.conversationId ?? "default",
      metadata: options.metadata ?? {},
    });
  }

  async recall(query: string, options: RecallOptions = {}): Promise<RecallResult[]> {
    const data = await this.transport.post<{ results: RecallResult[] }>(
      "/v1/memories/recall",
      {
        query,
        limit: options.limit ?? 10,
        layers: options.layers ?? null,
        min_score: options.minScore ?? 0.0,
      },
    );
    return data.results ?? [];
  }

  async reportOutcome(
    memoryIds: string[],
    success: boolean,
    weight = 1.0,
  ): Promise<void> {
    await this.transport.postEmpty("/v1/memories/report", {
      memory_ids: memoryIds,
      success,
      weight,
    });
  }

  async rememberNow(content: string, options: RememberNowOptions): Promise<Memory> {
    const expires =
      options.expiresAt instanceof Date ? options.expiresAt.toISOString() : options.expiresAt ?? null;
    return this.transport.post<Memory>("/v1/memories/store", {
      content,
      layer: options.layer,
      confidence: options.confidence ?? 0.9,
      expires_at: expires,
      metadata: options.metadata ?? {},
    });
  }

  async observe(conversationId = "default"): Promise<Observation[]> {
    const data = await this.transport.get<{ observations: Observation[] }>(
      "/v1/memories/observe",
      { conversation_id: conversationId },
    );
    return data.observations ?? [];
  }

  async consolidate(): Promise<ConsolidationResult> {
    return this.transport.post<ConsolidationResult>("/v1/memories/consolidate", {});
  }

  async compact(layer: MemoryLayer = "semantic"): Promise<CompactionResult> {
    // Server endpoint is part of memories router but currently routed
    // through consolidate logic; if a dedicated /compact endpoint is added
    // later, swap the path here.
    return this.transport.post<CompactionResult>("/v1/memories/consolidate", {
      compact: true,
      layer,
    });
  }

  // ── Knowledge graph ──────────────────────────────────────────────────

  async kgAddEntity(
    name: string,
    type: EntityType,
    metadata: Record<string, unknown> = {},
  ): Promise<unknown> {
    return this.transport.post("/v1/kg/write", {
      operation: "add_entity",
      name,
      entity_type: type,
      metadata,
    });
  }

  async kgAddRelationship(
    fromEntity: string,
    toEntity: string,
    relType: string,
    weight = 1.0,
    metadata: Record<string, unknown> = {},
  ): Promise<unknown> {
    return this.transport.post("/v1/kg/write", {
      operation: "add_relationship",
      from_entity: fromEntity,
      to_entity: toEntity,
      rel_type: relType,
      weight,
      metadata,
    });
  }

  async kgAddAttribute(entity: string, key: string, value: string): Promise<unknown> {
    return this.transport.post("/v1/kg/write", {
      operation: "add_attribute",
      name: entity,
      key,
      value,
    });
  }

  async kgQuery(name: string): Promise<EntityResult | null> {
    const data = await this.transport.post<{ result: EntityResult | null }>(
      "/v1/kg/query",
      { name, traverse_depth: 0 },
    );
    return data.result ?? null;
  }

  async kgTraverse(name: string, depth = 2): Promise<EntityResult[]> {
    const data = await this.transport.post<{ results: EntityResult[] }>(
      "/v1/kg/query",
      { name, traverse_depth: depth },
    );
    return data.results ?? [];
  }

  // ── Attention ────────────────────────────────────────────────────────

  async scoreAttention(message: string, options: AttentionOptions = {}): Promise<AttentionResult> {
    const ownerIds = options.ownerIds ? Array.from(options.ownerIds).join(",") : "";
    const allowlist = options.allowlist ? Array.from(options.allowlist).join(",") : "";
    return this.transport.postWithQuery<AttentionResult>("/v1/attention/score", {
      message,
      sender: options.sender ?? "",
      channel: options.channel ?? "dm",
      owner_ids: ownerIds,
      allowlist,
      ongoing: options.ongoing ? "true" : "false",
      already_answered: options.alreadyAnswered ? "true" : "false",
    });
  }
}
