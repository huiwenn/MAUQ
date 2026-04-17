#!/usr/bin/env python3
"""
Phase 0: Non-determinism pilot.
Per-model: 50 questions × 30 reps at temperature 0.
Measures baseline variance to determine required n per condition.

Usage:
  uv run python scripts/run_phase0_pilot.py --model claude-haiku
  uv run python scripts/run_phase0_pilot.py --merge   # merge per-model files
"""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topology_tax.config import CORE_MODELS, RESULTS_DIR
from topology_tax.bedrock_client import BedrockAgent
from topology_tax.datasets import load_questions, extract_answer


def run_single_model(model_name, model_id, output_dir):
    questions = load_questions("mmlu", n=50, seed=42)
    n_reps = 30

    agent = BedrockAgent(agent_id=model_name, model_id=model_id, temperature=0.0)

    model_results = []
    for qi, q in enumerate(questions):
        answers = []
        raw_responses = []
        for rep in range(n_reps):
            try:
                resp = agent.invoke(q["question"])
                ans = extract_answer(resp.text, task_type="mcq")
                answers.append(ans)
                raw_responses.append({
                    "text": resp.text,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "latency_ms": resp.latency_ms,
                })
            except Exception as e:
                print(f"  Error on {q['question_id']} rep {rep}: {e}")
                answers.append("ERROR")
                raw_responses.append({"text": f"ERROR: {e}", "input_tokens": 0, "output_tokens": 0, "latency_ms": 0})

        unique = len(set(a for a in answers if a != "ERROR"))
        most_common = Counter(answers).most_common(1)[0]

        model_results.append({
            "question_id": q["question_id"],
            "correct_answer": q["correct_answer"],
            "answers": answers,
            "raw_responses": raw_responses,
            "n_unique": unique,
            "most_common": most_common[0],
            "most_common_count": most_common[1],
            "deterministic": unique == 1,
        })

        det_so_far = sum(1 for r in model_results if r["deterministic"])
        print(f"  [{qi+1}/{len(questions)}] {q['question_id']}: "
              f"{unique} unique, det_rate={det_so_far}/{qi+1}")

        # Incremental save after every question
        out_file = output_dir / f"phase0_{model_name}.json"
        with open(out_file, "w") as f:
            json.dump({"model": model_name, "model_id": model_id,
                        "results": model_results}, f, indent=2)

    det_count = sum(1 for r in model_results if r["deterministic"])
    print(f"\n  {model_name}: {det_count}/{len(model_results)} deterministic "
          f"({det_count/len(model_results)*100:.1f}%)")
    return model_results


def merge_results(output_dir):
    merged = {}
    for f in sorted(output_dir.glob("phase0_*.json")):
        if f.name == "phase0_results.json":
            continue
        data = json.loads(f.read_text())
        merged[data["model"]] = data["results"]
        print(f"  Loaded {f.name}: {len(data['results'])} questions")

    if not merged:
        print("No per-model files found to merge.")
        return

    out = output_dir / "phase0_results.json"
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged {len(merged)} models -> {out}")

    print("\n=== RECOMMENDATION ===")
    for model_name, model_results in merged.items():
        det_rate = np.mean([r["deterministic"] for r in model_results])
        avg_unique = np.mean([r["n_unique"] for r in model_results])
        print(f"  {model_name}: {det_rate*100:.0f}% deterministic, "
              f"avg {avg_unique:.1f} unique answers per question")

    overall_det = np.mean([
        r["deterministic"]
        for model_results in merged.values()
        for r in model_results
    ])
    if overall_det > 0.95:
        print(f"\n  -> High determinism ({overall_det*100:.0f}%): n=5 runs sufficient")
    elif overall_det > 0.80:
        print(f"\n  -> Moderate determinism ({overall_det*100:.0f}%): n=10-15 runs recommended")
    else:
        print(f"\n  -> Low determinism ({overall_det*100:.0f}%): n=20+ runs needed")


def main():
    parser = argparse.ArgumentParser(description="Phase 0 non-determinism pilot")
    parser.add_argument("--model", type=str, help="Single model name from CORE_MODELS")
    parser.add_argument("--merge", action="store_true", help="Merge per-model results")
    args = parser.parse_args()

    output_dir = RESULTS_DIR / "phase0"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_results(output_dir)
    elif args.model:
        if args.model not in CORE_MODELS:
            print(f"Unknown model: {args.model}. Choose from: {list(CORE_MODELS)}")
            sys.exit(1)
        run_single_model(args.model, CORE_MODELS[args.model], output_dir)
    else:
        parser.print_help()
        print(f"\nAvailable models: {list(CORE_MODELS)}")


if __name__ == "__main__":
    main()
