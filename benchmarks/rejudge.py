"""Re-judge existing results with the updated judge prompt. No re-retrieval, no re-answering."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import anthropic

client = anthropic.Anthropic()


def judge_answer(question: str, prediction: str, ground_truth: str) -> bool:
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5,
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Ground truth: {ground_truth}\n"
                f"Predicted: {prediction}\n\n"
                "Mark as correct (yes) if the predicted answer:\n"
                "- Is semantically equivalent to the ground truth, OR\n"
                "- Contains the ground truth as part of a longer correct answer, OR\n"
                "- Expresses the same fact with different wording\n"
                "Mark as incorrect (no) if the predicted answer contradicts, omits, or gets the ground truth wrong.\n"
                "Reply with only 'yes' or 'no'."
            ),
        }],
    )
    return resp.content[0].text.strip().lower().startswith("yes")


input_path = "results/longmemeval_s_results.jsonl"
output_path = "results/longmemeval_s_rejudged.jsonl"

data = [json.loads(l) for l in open(input_path)]
flipped_to_correct = 0
flipped_to_wrong = 0
correct = 0

out = open(output_path, "w")
for i, row in enumerate(data):
    new_correct = judge_answer(row["question"], row["predicted"], row["ground_truth"])
    if new_correct != row["correct"]:
        if new_correct:
            flipped_to_correct += 1
        else:
            flipped_to_wrong += 1
    row["correct"] = new_correct
    correct += new_correct
    out.write(json.dumps(row) + "\n")
    if (i + 1) % 50 == 0:
        print(f"[{i+1}/500] QA={correct/(i+1):.1%}  flipped+={flipped_to_correct}  flipped-={flipped_to_wrong}")

out.close()
print(f"\n{'='*50}")
print(f"  QA accuracy (rejudged) : {correct/len(data):.1%}  (was 38.8%)")
print(f"  Flipped correct→wrong  : {flipped_to_wrong}")
print(f"  Flipped wrong→correct  : {flipped_to_correct}")
print(f"  Output                 : {output_path}")
print(f"{'='*50}")
