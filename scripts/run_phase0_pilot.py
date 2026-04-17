#!/usr/bin/env python3
"""
Phase 0: Non-determinism pilot.
5 models × 50 questions × 30 reps at temperature 0.
Measures baseline variance to determine required n per condition.
"""
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


def main():
    output_dir = RESULTS_DIR / "phase0"
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions("mmlu", n=50, seed=42)
    n_reps = 30

    results = {}
    for model_name, model_id in CORE_MODELS.items():
        print(f"\n=== {model_name} ({model_id}) ===")
        agent = BedrockAgent(
            agent_id=model_name,
            model_id=model_id,
            temperature=0.0,
        )

        model_results = []
        for q in tqdm(questions, desc=model_name):
            answers = []
            for rep in range(n_reps):
                try:
                    resp = agent.invoke(q["question"])
                    ans = extract_answer(resp.text, task_type="mcq")
                    answers.append(ans)
                except Exception as e:
                    print(f"  Error on {q['question_id']} rep {rep}: {e}")
                    answers.append("ERROR")

            unique = len(set(a for a in answers if a != "ERROR"))
            most_common = Counter(answers).most_common(1)[0]

            model_results.append({
                "question_id": q["question_id"],
                "correct_answer": q["correct_answer"],
                "answers": answers,
                "n_unique": unique,
                "most_common": most_common[0],
                "most_common_count": most_common[1],
                "deterministic": unique == 1,
            })

        results[model_name] = model_results

        # Summary stats
        det_count = sum(1 for r in model_results if r["deterministic"])
        print(f"  {det_count}/{len(model_results)} questions are deterministic "
              f"({det_count/len(model_results)*100:.1f}%)")

    # Save results
    output_file = output_dir / "phase0_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Compute recommended n
    print("\n=== RECOMMENDATION ===")
    for model_name, model_results in results.items():
        det_rate = np.mean([r["deterministic"] for r in model_results])
        avg_unique = np.mean([r["n_unique"] for r in model_results])
        print(f"  {model_name}: {det_rate*100:.0f}% deterministic, "
              f"avg {avg_unique:.1f} unique answers per question")

    overall_det = np.mean([
        r["deterministic"]
        for model_results in results.values()
        for r in model_results
    ])
    if overall_det > 0.95:
        print(f"\n  → High determinism ({overall_det*100:.0f}%): n=5 runs sufficient")
    elif overall_det > 0.80:
        print(f"\n  → Moderate determinism ({overall_det*100:.0f}%): n=10-15 runs recommended")
    else:
        print(f"\n  → Low determinism ({overall_det*100:.0f}%): n=20+ runs needed")


if __name__ == "__main__":
    main()
