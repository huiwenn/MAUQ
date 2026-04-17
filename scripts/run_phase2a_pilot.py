#!/usr/bin/env python3
"""
Phase 2a: LLM Topology Pilot.
100 MMLU questions × 3 topologies (independent, star, complete) × 20 runs.
Each run uses 5 heterogeneous agents (one per model).

Usage:
  uv run python scripts/run_phase2a_pilot.py --topology independent
  uv run python scripts/run_phase2a_pilot.py --topology star
  uv run python scripts/run_phase2a_pilot.py --topology complete
  uv run python scripts/run_phase2a_pilot.py --merge
"""
import argparse
import json
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topology_tax.config import CORE_MODELS, RESULTS_DIR
from topology_tax.bedrock_client import BedrockAgent
from topology_tax.datasets import load_questions, extract_answer
from topology_tax.topologies import build_topology
from topology_tax.orchestrator import DebateOrchestrator
from topology_tax.storage import ResultStore

N_QUESTIONS = 100
N_RUNS = 20
N_ROUNDS = 2
PILOT_TOPOLOGIES = ["independent", "star", "complete"]


def make_agents():
    agents = []
    for i, (name, model_id) in enumerate(CORE_MODELS.items()):
        agents.append(BedrockAgent(
            agent_id=f"{name}_{i}",
            model_id=model_id,
            temperature=0.0,
        ))
    return agents


def run_single_topology(topo_name, output_dir):
    questions = load_questions("mmlu", n=N_QUESTIONS, seed=42)
    agents = make_agents()
    orch = DebateOrchestrator(agents)

    db_path = output_dir / f"phase2a_{topo_name}.db"
    store = ResultStore(db_path)

    results_file = output_dir / f"phase2a_{topo_name}.json"
    all_results = []

    for run_id in range(N_RUNS):
        G = build_topology(topo_name, n_agents=len(agents), seed=run_id)
        run_results = []

        for qi, q in enumerate(questions):
            # Check if already completed (resume support)
            first_agent_id = f"{list(CORE_MODELS.keys())[0]}_0"
            if store.is_completed("mmlu", q["question_id"], topo_name, run_id, first_agent_id):
                continue

            t0 = time.monotonic()
            try:
                result = orch.run_debate(
                    G, question=q["question"],
                    task_type="mcq", n_rounds=N_ROUNDS,
                )
            except Exception as e:
                print(f"  ERROR run={run_id} q={qi}: {e}")
                continue
            wall_ms = int((time.monotonic() - t0) * 1000)

            # Majority vote
            answers = list(result["final_answers"].values())
            if answers:
                vote_counts = Counter(answers)
                majority_answer = vote_counts.most_common(1)[0][0]
            else:
                majority_answer = ""

            correct = (majority_answer == q["correct_answer"])

            # Save per-agent results to SQLite with full traces
            for agent_idx, agent in enumerate(agents):
                agent_answer = result["final_answers"].get(agent_idx, "")
                agent_response = result["responses"].get(agent_idx, "")
                agent_traces = [t for t in result.get("traces", []) if t["agent_idx"] == agent_idx]

                store.save_result(
                    dataset="mmlu",
                    question_id=q["question_id"],
                    topology=topo_name,
                    run_id=run_id,
                    agent_id=agent.agent_id,
                    agent_position=agent_idx,
                    prompt=q["question"],
                    response=agent_response,
                    answer_extracted=agent_answer,
                    correct=(agent_answer == q["correct_answer"]),
                    model_id=agent.model_id,
                    temperature=0.0,
                    latency_ms=agent_traces[-1]["latency_ms"] if agent_traces else 0,
                    input_tokens=sum(t["input_tokens"] for t in agent_traces),
                    output_tokens=sum(t["output_tokens"] for t in agent_traces),
                    metadata={
                        "majority_answer": majority_answer,
                        "majority_correct": correct,
                        "wall_ms": wall_ms,
                        "n_rounds": N_ROUNDS,
                        "traces": agent_traces,
                    },
                )

            run_results.append({
                "question_id": q["question_id"],
                "correct_answer": q["correct_answer"],
                "majority_answer": majority_answer,
                "majority_correct": correct,
                "per_agent_answers": {str(k): v for k, v in result["final_answers"].items()},
                "wall_ms": wall_ms,
            })

            progress_total = qi + 1 + run_id * N_QUESTIONS
            total_calls = N_RUNS * N_QUESTIONS
            print(f"  [{topo_name}] run={run_id} q={qi+1}/{N_QUESTIONS} "
                  f"majority={'Y' if correct else 'N'} "
                  f"({progress_total}/{total_calls} overall, {progress_total/total_calls*100:.1f}%)")

        all_results.extend(run_results)

        # Incremental save after each run
        with open(results_file, "w") as f:
            json.dump({
                "topology": topo_name,
                "n_questions": N_QUESTIONS,
                "n_runs_completed": run_id + 1,
                "n_rounds": N_ROUNDS,
                "results": all_results,
            }, f, indent=2)

    # Summary
    if all_results:
        acc = np.mean([r["majority_correct"] for r in all_results])
        print(f"\n  {topo_name}: {len(all_results)} results, majority accuracy = {acc:.3f}")
    store.close()


def merge_results(output_dir):
    merged = {}
    for topo in PILOT_TOPOLOGIES:
        f = output_dir / f"phase2a_{topo}.json"
        if f.exists():
            data = json.loads(f.read_text())
            n_runs = data.get("n_runs_completed", "?")
            n_results = len(data.get("results", []))
            acc = np.mean([r["majority_correct"] for r in data["results"]]) if data["results"] else 0
            merged[topo] = {"n_runs": n_runs, "n_results": n_results, "accuracy": acc}
            print(f"  {topo}: {n_runs} runs, {n_results} results, accuracy={acc:.3f}")
        else:
            print(f"  {topo}: NOT FOUND")

    if not merged:
        print("No results to merge.")
        return

    print("\n=== TOPOLOGY COMPARISON ===")
    for topo, stats in merged.items():
        print(f"  {topo}: accuracy={stats['accuracy']:.3f}")

    if "independent" in merged and len(merged) > 1:
        baseline = merged["independent"]["accuracy"]
        for topo, stats in merged.items():
            if topo != "independent":
                benefit = stats["accuracy"] - baseline
                print(f"  B({topo}) = {benefit:+.3f} (vs independent)")


def main():
    parser = argparse.ArgumentParser(description="Phase 2a LLM topology pilot")
    parser.add_argument("--topology", type=str, help="Single topology to run")
    parser.add_argument("--merge", action="store_true", help="Merge and compare results")
    args = parser.parse_args()

    output_dir = RESULTS_DIR / "phase2a"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_results(output_dir)
    elif args.topology:
        if args.topology not in PILOT_TOPOLOGIES:
            print(f"Unknown topology: {args.topology}. Choose from: {PILOT_TOPOLOGIES}")
            sys.exit(1)
        run_single_topology(args.topology, output_dir)
    else:
        parser.print_help()
        print(f"\nAvailable topologies: {PILOT_TOPOLOGIES}")


if __name__ == "__main__":
    main()
