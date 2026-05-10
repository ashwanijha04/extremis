# Benchmarks

## LongMemEval-S

Standard benchmark for long-term memory in LLM systems. 500 QA instances, each
backed by ~40 timestamped conversation sessions.

**Metrics:**
- **QA accuracy** — did the agent answer correctly given recalled context?
- **Retrieval R@5** — did the top-5 recalled memories include content from the answer session?

### 1. Get the dataset

```bash
git clone https://github.com/xiaowu0162/LongMemEval /tmp/longmemeval
# dataset file: /tmp/longmemeval/data/longmemeval_s.json
```

### 2. Install dependencies

```bash
pip install "extremis[mcp,observe]" tqdm anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run

```bash
# Quick smoke test — first 20 instances
python benchmarks/longmemeval_s.py \
    --dataset /tmp/longmemeval/data/longmemeval_s.json \
    --limit 20

# Full 500-instance run (~$2 in Haiku API calls, ~30 min)
python benchmarks/longmemeval_s.py \
    --dataset /tmp/longmemeval/data/longmemeval_s.json \
    --output  results/longmemeval_s_results.jsonl

# With LLM consolidation enabled (~$7 total, higher accuracy)
python benchmarks/longmemeval_s.py \
    --dataset /tmp/longmemeval/data/longmemeval_s.json \
    --consolidate \
    --output  results/longmemeval_s_consolidated.jsonl

# Resume an interrupted run
python benchmarks/longmemeval_s.py \
    --dataset /tmp/longmemeval/data/longmemeval_s.json \
    --resume \
    --output  results/longmemeval_s_results.jsonl
```

### Cost

Measured with peekr on a 10-instance sample run:

| Mode | LLM calls | Actual cost per instance |
|------|-----------|--------------------------|
| Default (auto_consolidate=off) | 2 Haiku calls | ~$0.003 |
| With `--consolidate` | ~20 Haiku calls | ~$0.058 |

Full 500-instance run: **~$1.50** default · **~$29** with consolidation.

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | required | Path to `longmemeval_s.json` |
| `--output` | `results/longmemeval_s_results.jsonl` | Where to write per-instance results |
| `--recall-k` | 5 | Memories retrieved per question |
| `--limit` | None | Cap number of instances (for testing) |
| `--consolidate` | off | Run LLM consolidation after feeding each instance |
| `--resume` | off | Skip question_ids already in the output file |
