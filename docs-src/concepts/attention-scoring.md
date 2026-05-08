# Attention scoring

The attention scorer grades incoming messages 0–100 and assigns a processing level. It's a pure heuristic — no LLM, no network, runs in microseconds.

## Use case

In group chats, broadcast channels, or any context where you receive many messages, most don't warrant a full LLM response. The attention scorer tells you whether to engage fully, respond briefly, or skip.

## Levels

| Level | Score | What to do |
|-------|-------|-----------|
| `full` | ≥ 75 | Engage fully — rich response |
| `standard` | ≥ 50 | Engage — balanced response |
| `minimal` | ≥ 25 | Brief acknowledgement only |
| `ignore` | < 25 | Skip |

## Usage

```python
result = mem.score_attention(
    message="URGENT: the API is down!",
    sender="user_123",
    channel="dm",           # "dm" | "group" | "broadcast"
    owner_ids={"user_123"}, # these always get full attention
    allowlist={"user_456"}, # elevated base score
    context={
        "ongoing": True,          # part of an active thread
        "already_answered": False,
    },
)

print(result.score)    # → 85
print(result.level)    # → "full"
print(result.reason)   # → "unknown sender; channel=dm; urgent keyword; question"
```

## Scoring factors

```
score = sender_score + channel_score + content_score + context_score
```

**Sender:**
- Owner ID → 100 (immediate `full` return, no further scoring)
- Allowlisted → +60
- Unknown → +10

**Channel:**
- DM → +25
- Group → +15
- Broadcast → +5

**Content signals:**
- Urgent keywords (urgent, asap, emergency, broken, down) → +20
- Action request (do, please, can you, fix, check) → +15
- Question mark → +10
- Casual banter (short message, casual words) → +8
- Single emoji → −5

**Context:**
- Ongoing conversation → +10
- New thread → +10
- Already answered → −15

## Configuring thresholds

```python
from extremis import Config, Extremis

mem = Extremis(config=Config(
    attention_full_threshold=80,      # default: 75
    attention_standard_threshold=55,  # default: 50
    attention_minimal_threshold=30,   # default: 25
))
```

Or via env vars: `EXTREMIS_ATTENTION_FULL_THRESHOLD=80`

## Via MCP

```
memory_score_attention(
    message="can you help me?",
    sender="user_123",
    channel="group",
    ongoing=true
)
```

Returns: `"Score: 60/100  Level: standard\nReason: ..."`
