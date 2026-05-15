# Hallucination detection

Most teams ship AI memory systems with zero runtime hallucination checks. extremis runs a three-layer detection stack at write time and exposes hedging signals at read time so production responses can be honest about what they actually know.

## The problem

When an LLM extracts a "fact" from a conversation, it can sound completely confident about something the user never said. That fact gets stored, recalled later, and surfaced as truth. The system has no idea it was wrong.

Without runtime checks, you find out via support tickets.

## What extremis detects

There are five hallucination types that surface in production memory systems:

1. **Factual fabrication** — invented dates, names, numbers
2. **Source fabrication** — citing a message or document that doesn't exist
3. **Context contradiction** — extracted memory directly conflicts with the source conversation
4. **Instruction drift** — extracting facts despite "only respond from context" rules
5. **Confident speculation** — guess presented as certainty

In a memory system, type 3 is the load-bearing one — and the only one you can catch programmatically at write time because the source conversation is right there.

## The detection stack

```
extraction
    │
    ├─►  Self-consistency  (IDENTITY, SEMANTIC layers only)
    │       sample N times at temperature > 0; keep claims
    │       that converge in embedding space
    │
    ├─►  NLI faithfulness
    │       local cross-encoder checks entailment against
    │       source conversation messages
    │
    ├─►  LLM-as-judge (grey zone only, score 0.5–0.85)
    │       Claude Haiku returns structured verdict:
    │       SUPPORTED / CONTRADICTED / UNVERIFIABLE
    │
    └─►  Store with metadata.verification + downranked confidence
         + actionable recommendations
```

### Self-consistency

For high-stakes layers (`identity`, `semantic`), the extractor runs N times at `temperature > 0`. Claims are kept only if they converge — measured by mean pairwise cosine similarity of their embeddings.

If the model is confident in the underlying fact, every sample says roughly the same thing. If it's interpolating, samples diverge.

Configure via `Config.self_consistency_n` (default 3), `Config.consistency_threshold` (default 0.85), and `Config.self_consistency_layers` (default `identity,semantic`).

### NLI faithfulness

A local NLI model (`cross-encoder/nli-deberta-v3-small` by default) checks entailment of each extracted memory against each source message in the conversation. The faithfulness score is the max entailment probability across all sources.

NLI is essential here because **embedding similarity gets negation wrong**. "Payment must be made within 30 days" and "Payment does not need to be made within 30 days" have near-identical embeddings. NLI correctly flags them as CONTRADICTION.

Install with the verification extra:

```bash
pip install "extremis[verification]"
```

Without it, the stack falls back to judge-only mode automatically.

### LLM-as-judge

When NLI lands in the grey zone (`0.5–0.85`), the system escalates to a Claude Haiku call that returns a structured JSON verdict:

```json
{
  "verdict": "CONTRADICTED",
  "score": 0.1,
  "reason": "Source says Acme; memory says Globex."
}
```

NLI scores below `0.5` skip the judge — saves cost on obvious fails. NLI scores `≥ 0.85` skip the judge — clean passes don't need adjudication. So judge cost is bounded to the grey zone, not every extraction.

## What happens to flagged memories

**Tag and downrank, never drop.** Every detection path stores the memory. Failure modes:

- `metadata.verification.verdict` is stamped (`SUPPORTED` / `CONTRADICTED` / `UNVERIFIABLE`)
- `confidence` is reduced to `min(original, faithfulness_score)`
- `metadata.recommendations` carries actionable items for operators

This means false-positive verifiers reduce surfacing but never erase memories. Compaction or manual review can recover. The audit trail stays intact.

## Recall-time signals

Every `RecallResult` carries:

```python
RecallResult.effective_confidence  # confidence × layer_weight × temporal_decay
RecallResult.sources               # structured provenance + recommendations
```

### Effective confidence

```
effective_confidence = confidence × LAYER_WEIGHT × 2^(-age_days / half_life_days)
```

Layer weights (from the production stack):

| Layer | Weight |
|---|---|
| `identity` | 0.95 |
| `semantic` | 0.80 |
| `procedural` | 0.70 |
| `episodic` | 0.60 |
| `working` | 0.40 |

Expired memories (past `validity_end`) decay to `0` immediately.

Use this to hedge responses: a memory with `effective_confidence < 0.3` should be presented as "as of N days ago, …" rather than present tense.

### Sources

```python
RecallResult.sources = {
    "conversation_id": "conv_001",
    "source_message_ids": ["msg_42"],
    "source_memory_ids": [<UUID of episodic ancestor>, ...],
    "layer": "semantic",
    "created_at": "2025-03-15T10:00:00Z",
    "verification": { "verdict": "SUPPORTED", "score": 0.94, "method": "nli" },
    "consistency": { "mean_similarity": 0.91, "samples": 3 },
    "recommendations": [...],
}
```

Walk `source_memory_ids` backwards to find the originating episodic memories. Walk those memories' `metadata.conversation_id` to find the log entries that started it all.

## Recommendations

Every flagged memory carries `metadata.recommendations` — structured, actionable items so operators see *what to do* about each detected issue, not just *that* something was detected.

```python
@dataclass
class Recommendation:
    issue: str          # machine-readable identifier
    severity: str       # "low" | "medium" | "high"
    action: str         # imperative — what to do now
    suggestion: str     # systemic fix
    refs: dict          # pointers (source_message_ids, memory_id, ...)
```

| Issue | Severity | Surfaced at |
|---|---|---|
| `claim_contradicts_source` | high | Write time (NLI / judge says CONTRADICTED) |
| `claim_unverifiable` | medium | Write time (judge says UNVERIFIABLE) |
| `borderline_support` | low | Write time (score in grey zone but passed) |
| `memory_expired` | high | Recall time (`validity_end` in the past) |
| `surfacing_contradicted_memory` | high | Recall time (memory was flagged but is still recallable) |
| `stale_confidence` | medium | Recall time (`effective_confidence < 0.3`) |

## Parent-child lineage

Consolidated `semantic` and `procedural` memories carry `source_memory_ids` pointing back to the `episodic` memories from the same conversation that grounded them. When a flagged memory needs investigation, this chain takes you from the consolidated fact → its episodic ancestors → the originating log entries → the user's actual words.

```
LogEntry ───▶ Episodic Memory ───▶ Semantic Memory ───▶ (post-compact) Reconciled Memory
                  ▲                       │
                  └───── source_memory_ids points back
```

## Observability

If [peekr](https://github.com/ashwanijha04/peekr) is installed and `EXTREMIS_OBSERVE=true` is set, every verification call emits trace spans:

- `extremis.verification.nli`
- `extremis.verification.judge`
- `extremis.verification.consistency`
- `extremis.consolidation.extract`
- `extremis.consolidation.contradiction_check`

This gives you the production audit trail: which memories got downranked, by which check, and how often the judge was invoked.

Peekr is **optional, not required** — the stack has graceful no-op fallbacks for `@trace` throughout. Strongly recommended in production for the audit story.

## Configuration

```python
from extremis import Config

config = Config(
    enable_faithfulness_check=True,
    faithfulness_pass_threshold=0.85,
    faithfulness_grey_zone_low=0.5,
    self_consistency_n=3,
    self_consistency_layers="identity,semantic",
    consistency_threshold=0.85,
    confidence_half_life_days=180,
)
```

Set `enable_faithfulness_check=False` to disable the stack. Set `self_consistency_n=0` to disable sampling but keep NLI + judge.

## SDK access

All hallucination-detection signals flow through the HTTP API and are first-class in every SDK.

### Python

```python
results = mem.recall("Where does the user work?")
for r in results:
    if r.effective_confidence and r.effective_confidence < 0.3:
        print("hedge:", r.memory.content)
    for rec in (r.sources or {}).get("recommendations", []):
        print(f"[{rec['severity']}] {rec['issue']}: {rec['action']}")
```

### TypeScript

```ts
const results = await mem.recall("Where does the user work?");
for (const r of results) {
  if (r.effective_confidence && r.effective_confidence < 0.3) {
    console.warn("hedge:", r.memory.content);
  }
  for (const rec of r.sources?.recommendations ?? []) {
    console.warn(`[${rec.severity}] ${rec.issue}: ${rec.action}`);
  }
}
```
