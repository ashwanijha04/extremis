/**
 * Type definitions mirroring the Python extremis types over the wire.
 *
 * Verification + recommendations are first-class typed fields, not opaque
 * metadata — they're the load-bearing signals for production hallucination
 * detection and should be discoverable through TypeScript's intellisense.
 */

export type MemoryLayer =
  | "episodic"
  | "semantic"
  | "procedural"
  | "identity"
  | "working";

export type EntityType =
  | "person"
  | "org"
  | "project"
  | "group"
  | "concept"
  | "other";

export type ObservationPriority = "critical" | "context" | "info";

// ─── Verification & recommendations ──────────────────────────────────────

export type FaithfulnessVerdict =
  | "SUPPORTED"
  | "CONTRADICTED"
  | "UNVERIFIABLE";

export type VerificationMethod =
  | "nli"
  | "nli+judge"
  | "judge-only"
  | "skipped";

/** Output of the write-time tiered faithfulness check. */
export interface Verification {
  score: number;
  verdict: FaithfulnessVerdict;
  method: VerificationMethod;
  nli_score?: number | null;
  judge_score?: number | null;
  judge_reason?: string;
  best_source_idx?: number | null;
  verified_at?: string;
}

/** Self-consistency stats for claims that passed the convergence filter. */
export interface Consistency {
  mean_similarity: number;
  samples: number;
}

export type RecommendationSeverity = "low" | "medium" | "high";

export type RecommendationIssue =
  | "claim_contradicts_source"
  | "claim_unverifiable"
  | "borderline_support"
  | "memory_expired"
  | "surfacing_contradicted_memory"
  | "stale_confidence";

/**
 * Actionable item for an operator when verification detects an issue.
 * Every Recommendation carries both an immediate `action` and a systemic
 * `suggestion` so you can fix the symptom and the cause.
 */
export interface Recommendation {
  issue: RecommendationIssue;
  severity: RecommendationSeverity;
  action: string;
  suggestion: string;
  refs: Record<string, unknown>;
}

// ─── Memory core ─────────────────────────────────────────────────────────

/** Metadata bag — typed access to known verification fields. */
export interface MemoryMetadata {
  source?: string;
  conversation_id?: string;
  model?: string;
  source_message_ids?: string[];
  verification?: Verification;
  consistency?: Consistency;
  recommendations?: Recommendation[];
  [key: string]: unknown;
}

export interface Memory {
  id: string;
  namespace: string;
  layer: MemoryLayer;
  content: string;
  embedding?: number[] | null;
  score: number;
  confidence: number;
  metadata: MemoryMetadata;
  source_memory_ids: string[];
  validity_start: string;
  validity_end?: string | null;
  created_at: string;
  last_accessed_at?: string | null;
  access_count: number;
  do_not_consolidate: boolean;
}

/** Structured provenance trail surfaced on every RecallResult. */
export interface RecallSources {
  conversation_id: string | null;
  source_message_ids: string[];
  source_memory_ids: string[];
  layer: MemoryLayer;
  created_at: string | null;
  verification: Verification | null;
  consistency: Consistency | null;
  recommendations: Recommendation[];
}

export interface RecallResult {
  memory: Memory;
  relevance: number;
  final_rank: number;
  reason: string;
  /** confidence × layer_weight × temporal_decay. Use to hedge ("as of X…"). */
  effective_confidence: number | null;
  /** "Where did this memory come from?" trail. */
  sources: RecallSources | null;
}

// ─── Other returned shapes ───────────────────────────────────────────────

export interface ConsolidationResult {
  memories_created: number;
  memories_superseded: number;
  log_checkpoint: string;
  duration_seconds: number;
  notes: string;
}

export interface CompactionResult {
  memories_reconciled: number;
  memories_deduped: number;
  memories_unchanged: number;
  duration_seconds: number;
}

export interface Observation {
  id: string;
  namespace: string;
  content: string;
  priority: ObservationPriority;
  timestamp: string;
  conversation_id: string | null;
  tags: string[];
}

export interface AttentionResult {
  score: number;
  level: "full" | "standard" | "minimal" | "ignore";
  reason: string;
  breakdown: Record<string, unknown>;
}

export interface Entity {
  id: number;
  namespace: string;
  name: string;
  type: EntityType;
  metadata: Record<string, unknown>;
}

export interface Relationship {
  id: number;
  namespace: string;
  from_entity: string;
  to_entity: string;
  rel_type: string;
  weight: number;
  metadata: Record<string, unknown>;
}

export interface KGAttribute {
  id: number;
  namespace: string;
  entity: string;
  key: string;
  value: string;
}

export interface EntityResult {
  entity: Entity;
  relationships: Relationship[];
  attributes: KGAttribute[];
}

// ─── Client options ──────────────────────────────────────────────────────

export interface ClientOptions {
  apiKey: string;
  baseUrl?: string;
  /** Total request timeout in ms. Default 30000. */
  timeoutMs?: number;
  /** Max retry attempts on 429 / 5xx. Default 3. Set 0 to disable. */
  maxRetries?: number;
  /** Initial backoff in ms (doubled per attempt with jitter). Default 250. */
  baseDelayMs?: number;
  /** Cap on backoff delay. Default 8000. */
  maxDelayMs?: number;
  /** Custom fetch implementation. Defaults to globalThis.fetch. */
  fetch?: typeof fetch;
}
