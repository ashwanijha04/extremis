"""
LongMemEval-S benchmark runner for extremis.

Measures:
  - QA accuracy   : LLM answers correctly given recalled context
  - Retrieval R@5 : top-5 recalled memories include content from the answer session

Usage:
    pip install extremis tqdm anthropic
    python benchmarks/longmemeval_s.py \
        --dataset path/to/longmemeval_s.json \
        --output  results/longmemeval_s_results.jsonl

Optional flags:
    --consolidate   run LLM consolidation after feeding each instance (costs ~$0.01/instance)
    --recall-k 5    number of memories to retrieve per question (default 5)
    --limit 50      only run first N instances (useful for a quick smoke test)
    --resume        skip already-evaluated question_ids in the output file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import anthropic

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_):  # type: ignore
        return it

# ── make sure the local src/ tree is importable when run from the repo root ──
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from extremis import Extremis
from extremis.config import Config
from extremis.consolidation.consolidator import LLMConsolidator


# ── helpers ──────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def feed_sessions(mem: Extremis, sessions: list[list[dict]], session_ids: list[str]) -> None:
    """Store every turn of every session into extremis."""
    for session, sid in zip(sessions, session_ids):
        for turn in session:
            mem.remember(
                content=turn["content"],
                role=turn.get("role", "user"),
                conversation_id=sid,
            )


def build_context(recall_results) -> str:
    return "\n".join(
        f"[{r.memory.layer.value}] {r.memory.content}"
        for r in recall_results
    )


def answer_question(client: anthropic.Anthropic, context: str, question: str) -> str:
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": (
                f"You have access to the following memory context:\n\n{context}\n\n"
                f"Answer this question using only the context above. "
                f"If the answer is not in the context, say exactly: I don't know\n\n"
                f"Question: {question}"
            ),
        }],
    )
    return resp.content[0].text.strip()


def judge_answer(client: anthropic.Anthropic, question: str, prediction: str, ground_truth: str) -> bool:
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5,
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Ground truth: {ground_truth}\n"
                f"Predicted: {prediction}\n\n"
                "Is the predicted answer correct or semantically equivalent to the ground truth? "
                "Reply with only 'yes' or 'no'."
            ),
        }],
    )
    return resp.content[0].text.strip().lower().startswith("yes")


def retrieval_hit(recall_results, answer_session_ids: list[str]) -> bool:
    """Return True if any recalled memory came from an answer session."""
    if not answer_session_ids:
        return False
    recalled_sids = {
        r.memory.metadata.get("conversation_id", "")
        for r in recall_results
    }
    return bool(recalled_sids & set(answer_session_ids))


def load_completed_ids(output_path: str) -> set[str]:
    done = set()
    p = Path(output_path)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["question_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


# ── main ─────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    client = anthropic.Anthropic()
    dataset = load_dataset(args.dataset)

    if args.limit:
        dataset = dataset[: args.limit]

    completed = load_completed_ids(args.output) if args.resume else set()
    if completed:
        print(f"Resuming — skipping {len(completed)} already-evaluated instances.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_file = open(args.output, "a")

    qa_correct = 0
    ret_hits = 0
    evaluated = 0

    # load the embedding model once — reused across all instances
    from extremis.embeddings.sentence_transformers import SentenceTransformerEmbedder
    shared_embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    print("Embedder loaded. Starting evaluation...")

    for instance in tqdm(dataset, desc="LongMemEval-S"):
        qid = instance["question_id"]
        if qid in completed:
            continue

        question        = instance["question"]
        ground_truth    = instance["answer"]
        sessions        = instance["haystack_sessions"]
        session_ids     = instance["haystack_session_ids"]
        answer_sids     = instance.get("answer_session_ids", [])

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(namespace=qid, extremis_home=tmpdir)
            mem = Extremis(config=config, embedder=shared_embedder)

            feed_sessions(mem, sessions, session_ids)

            if args.consolidate:
                consolidator = LLMConsolidator(config, mem._embedder)
                consolidator.run_pass(mem.get_log(), mem.get_local_store(), mem.get_local_store())

            recall_results = mem.recall(question, limit=args.recall_k)
            context = build_context(recall_results)

            predicted = answer_question(client, context, question)
            correct   = judge_answer(client, question, predicted, ground_truth)
            hit       = retrieval_hit(recall_results, answer_sids)

        if correct:
            qa_correct += 1
        if hit:
            ret_hits += 1
        evaluated += 1

        row = {
            "question_id":     qid,
            "question":        question,
            "ground_truth":    ground_truth,
            "predicted":       predicted,
            "correct":         correct,
            "retrieval_hit":   hit,
            "recalled":        [r.memory.content for r in recall_results],
            "final_ranks":     [round(r.final_rank, 4) for r in recall_results],
        }
        out_file.write(json.dumps(row) + "\n")
        out_file.flush()

        if evaluated % 50 == 0:
            print(
                f"  [{evaluated}/{len(dataset)}] "
                f"QA={qa_correct/evaluated:.1%}  "
                f"R@{args.recall_k}={ret_hits/evaluated:.1%}"
            )

    out_file.close()

    print("\n" + "=" * 52)
    print(f"  LongMemEval-S  —  extremis results")
    print("=" * 52)
    print(f"  Instances evaluated : {evaluated}")
    print(f"  QA accuracy         : {qa_correct/evaluated:.1%}")
    print(f"  Retrieval R@{args.recall_k:<2}       : {ret_hits/evaluated:.1%}")
    print(f"  Output              : {args.output}")
    print("=" * 52)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LongMemEval-S benchmark for extremis")
    p.add_argument("--dataset",     required=True,  help="Path to longmemeval_s.json")
    p.add_argument("--output",      default="results/longmemeval_s_results.jsonl")
    p.add_argument("--recall-k",    type=int, default=5,   dest="recall_k")
    p.add_argument("--limit",       type=int, default=None, help="Only run first N instances")
    p.add_argument("--consolidate", action="store_true",   help="Run LLM consolidation pass per instance")
    p.add_argument("--resume",      action="store_true",   help="Skip already-evaluated question_ids")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
