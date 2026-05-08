# RL scoring

extremis learns which memories are actually useful through a feedback loop — not just which ones are similar.

## The problem with cosine-only retrieval

Most memory systems rank by cosine similarity: the most *similar* memory wins. But similar ≠ useful.

- A memory you've recalled 50 times and confirmed helpful should rank higher than an equally similar but untested one
- A memory that consistently led to bad responses should fade out
- Old memories should decay naturally unless they're still being used

## The ranking formula

Every recalled memory gets a `final_rank`:

```
final_rank = cosine_similarity
           × (1 + α · tanh(utility_score))
           × exp(−ln2 · age_days / half_life)
```

| Factor | What it measures | Controlled by |
|--------|-----------------|--------------|
| `cosine_similarity` | Semantic relevance to the query | Embedding model |
| `utility_score` | Accumulated feedback signal | `report_outcome()` |
| `recency_decay` | How fresh the memory is | `EXTREMIS_RECENCY_HALF_LIFE_DAYS` |

`α` (alpha) controls the balance between relevance and utility. Default: `0.5`.

## Giving feedback

```python
results = mem.recall("what does the user prefer?")

# After using these memories in a response:
mem.report_outcome(
    memory_ids=[r.memory.id for r in results[:2]],
    success=True,   # True = +1, False = -1
    weight=1.0,     # amplify for strong signals (e.g. explicit user feedback)
)
```

### Asymmetric weighting

Negative signals apply **1.5× weight** — the same asymmetry human threat-learning uses. Mistakes fade faster than successes accumulate.

```python
# Positive: score += weight * 1.0  (e.g. +1.0)
mem.report_outcome([mem_id], success=True, weight=1.0)

# Negative: score += -(weight * 1.5)  (e.g. -1.5)
mem.report_outcome([mem_id], success=False, weight=1.0)
```

### When to call it

- After any response where you can assess quality
- When the user explicitly says "that was wrong" or "that was helpful"
- Use higher `weight` for explicit user ratings vs implicit signals

## Reading the reason

Every `RecallResult` has a `reason` string that explains the ranking:

```python
for r in results:
    print(r.reason)
```

```
similarity 0.91 · score +4.0 · used 8× · 3d old
similarity 0.54 · score -1.5 · first recall · 45d old
procedural (always included) · similarity 0.32 · used 2× · today
```

The reason tells you: semantic match strength, accumulated feedback, how many times recalled, and age.

## Recency decay

Memories decay in ranking over time. A memory from 90 days ago (default half-life) ranks at ~50% of what it would rank if it were today.

```python
# Configure half-life
from extremis import Config, Extremis

mem = Extremis(config=Config(recency_half_life_days=30))  # faster decay
mem = Extremis(config=Config(recency_half_life_days=365)) # slower decay
```

Or via env var: `EXTREMIS_RECENCY_HALF_LIFE_DAYS=30`

Memories with high `utility_score` resist decay — the formula multiplies recency by utility, so a memory with score +10 stays relevant much longer than one with score 0.

## Tuning α

`α` controls how much feedback influences ranking vs raw similarity:

```python
# α = 0.0: pure cosine similarity (feedback ignored)
# α = 0.5: balanced (default)
# α = 1.0: feedback dominates, similarity secondary

mem = Extremis(config=Config(rl_alpha=0.8))
```

Or: `EXTREMIS_RL_ALPHA=0.8`

## Relevance floor

Memories below a minimum cosine similarity are filtered out to prevent noise:

```python
# Default: 0.05 (5% cosine similarity)
# Pinned layers (identity/procedural) bypass this floor
mem = Extremis(config=Config(recall_min_relevance=0.1))
```

Or: `EXTREMIS_RECALL_MIN_RELEVANCE=0.1`
