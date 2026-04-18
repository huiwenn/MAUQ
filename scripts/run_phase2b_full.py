#!/usr/bin/env python3
"""
Phase 2b: Full LLM topology experiment.
500 MMLU questions × 10 topologies × 10 runs × 5 agents × 2 rounds ≈ 500K Bedrock calls.

Usage:
  uv run python scripts/run_phase2b_full.py --topology ring
  uv run python scripts/run_phase2b_full.py --topology all
  uv run python scripts/run_phase2b_full.py --merge
  uv run python scripts/run_phase2b_full.py --status
"""
import argparse
import json
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topology_tax.config import CORE_MODELS, RESULTS_DIR, TOPOLOGY_NAMES
from topology_tax.bedrock_client import BedrockAgent
from topology_tax.datasets import load_questions, extract_answer
from topology_tax.topologies import build_topology
from topology_tax.orchestrator import DebateOrchestrator
from topology_tax.storage import ResultStore

N_QUESTIONS = 500
N_RUNS = 10
N_ROUNDS = 2
ALL_TOPOLOGIES = TOPOLOGY_NAMES

_shutdown = False


def _handle_signal(sig, frame):
    global _shutdown
    _shutdown = True
    print("\nGraceful shutdown requested — finishing current question...")


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def make_agents():
    agents = []
    for i, (name, model_id) in enumerate(CORE_MODELS.items()):
        agents.append(BedrockAgent(
            agent_id=f"{name}_{i}",
            model_id=model_id,
            temperature=0.0,
        ))
    return agents


def run_single_topology(topo_name, output_dir, questions):
    global _shutdown
    agents = make_agents()
    orch = DebateOrchestrator(agents)

    db_path = output_dir / f"phase2b_{topo_name}.db"
    store = ResultStore(db_path)

    results_file = output_dir / f"phase2b_{topo_name}.json"

    all_results = []
    if results_file.exists():
        prev = json.loads(results_file.read_text())
        all_results = prev.get("results", [])

    first_agent_id = f"{list(CORE_MODELS.keys())[0]}_0"
    errors = []
    t_start = time.monotonic()
    completed_this_session = 0

    for run_id in range(N_RUNS):
        G = build_topology(topo_name, n_agents=len(agents), seed=run_id)
        run_results = []

        for qi, q in enumerate(questions):
            if _shutdown:
                print(f"  Shutting down — saving {len(all_results)} results")
                break

            if store.is_completed("mmlu", q["question_id"], topo_name, run_id, first_agent_id):
                continue

            t0 = time.monotonic()
            try:
                result = orch.run_debate(
                    G, question=q["question"],
                    task_type="mcq", n_rounds=N_ROUNDS,
                )
            except Exception as e:
                errors.append({"run_id": run_id, "qi": qi, "error": str(e)})
                print(f"  ERROR run={run_id} q={qi}: {e}")
                continue
            wall_ms = int((time.monotonic() - t0) * 1000)

            answers = list(result["final_answers"].values())
            if answers:
                vote_counts = Counter(answers)
                majority_answer = vote_counts.most_common(1)[0][0]
            else:
                majority_answer = ""

            correct = (majority_answer == q["correct_answer"])

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
                "run_id": run_id,
                "correct_answer": q["correct_answer"],
                "majority_answer": majority_answer,
                "majority_correct": correct,
                "per_agent_answers": {str(k): v for k, v in result["final_answers"].items()},
                "wall_ms": wall_ms,
                "subject": q.get("subject"),
            })

            completed_this_session += 1
            progress = qi + 1 + run_id * N_QUESTIONS
            total = N_RUNS * N_QUESTIONS
            elapsed = time.monotonic() - t_start
            rate = completed_this_session / elapsed if elapsed > 0 else 0
            eta_s = (total - progress) / rate if rate > 0 else 0
            print(f"  [{topo_name}] run={run_id} q={qi+1}/{N_QUESTIONS} "
                  f"majority={'Y' if correct else 'N'} "
                  f"({progress}/{total}, {progress/total*100:.1f}%) "
                  f"[{rate:.1f} q/s, ETA {eta_s/3600:.1f}h]")

        if _shutdown:
            break

        all_results.extend(run_results)

        actual_runs = len({r.get("run_id", -1) for r in all_results})
        with open(results_file, "w") as f:
            json.dump({
                "topology": topo_name,
                "dataset": "mmlu",
                "n_questions": N_QUESTIONS,
                "n_runs_completed": actual_runs,
                "n_rounds": N_ROUNDS,
                "n_models": len(CORE_MODELS),
                "models": list(CORE_MODELS.keys()),
                "results": all_results,
                "errors": errors,
            }, f, indent=2)

    if all_results:
        acc = np.mean([r["majority_correct"] for r in all_results])
        print(f"\n  {topo_name}: {len(all_results)} results, majority accuracy = {acc:.3f}")
        if errors:
            print(f"  ({len(errors)} errors encountered)")
    store.close()


def show_status(output_dir):
    print("\n=== Phase 2b Status ===\n")
    total_expected = N_QUESTIONS * N_RUNS
    grand_total = 0
    grand_complete = 0

    for topo in ALL_TOPOLOGIES:
        f = output_dir / f"phase2b_{topo}.json"
        if not f.exists():
            print(f"  {topo}: NOT STARTED")
            grand_total += total_expected
            continue
        data = json.loads(f.read_text())
        results = data.get("results", [])
        n_runs = data.get("n_runs_completed", 0)
        n_errors = len(data.get("errors", []))
        acc = np.mean([r["majority_correct"] for r in results]) if results else 0

        pct = len(results) / total_expected * 100
        status = "COMPLETE" if n_runs >= N_RUNS else f"{pct:.0f}%"
        err_str = f" ({n_errors} errors)" if n_errors else ""
        print(f"  {topo}: {status} — {n_runs}/{N_RUNS} runs, "
              f"{len(results)} results, acc={acc:.3f}{err_str}")
        grand_total += total_expected
        grand_complete += len(results)

    print(f"\n  Overall: {grand_complete}/{grand_total} "
          f"({grand_complete/grand_total*100:.1f}%)")
    remaining_calls = (grand_total - grand_complete) * len(CORE_MODELS) * N_ROUNDS
    print(f"  Estimated remaining Bedrock calls: ~{remaining_calls:,}")


def merge_results(output_dir):
    print("\n=== Phase 2b: Full MMLU Topology Comparison ===\n")
    all_data = {}

    for topo in ALL_TOPOLOGIES:
        f = output_dir / f"phase2b_{topo}.json"
        if not f.exists():
            print(f"  {topo}: NOT FOUND")
            continue
        data = json.loads(f.read_text())
        results = data.get("results", [])
        if not results:
            continue

        all_data[topo] = results
        n_runs = data.get("n_runs_completed", "?")
        acc = np.mean([r["majority_correct"] for r in results])
        ci = 1.96 * np.std([r["majority_correct"] for r in results]) / np.sqrt(len(results))
        print(f"  {topo} ({n_runs} runs): accuracy = {acc:.3f} ± {ci:.3f} ({len(results)} results)")

    if not all_data:
        print("No results to merge.")
        return

    # Per-subject breakdown
    subjects = sorted({r.get("subject") for results in all_data.values()
                       for r in results if r.get("subject")})
    if subjects:
        print("\n=== ACCURACY BY SUBJECT ===\n")
        header = f"{'topology':<20}" + "".join(f"{s[:12]:>14}" for s in subjects)
        print(header)
        print("-" * len(header))
        for topo, results in all_data.items():
            row = f"{topo:<20}"
            for subj in subjects:
                sr = [r for r in results if r.get("subject") == subj]
                if sr:
                    row += f"{np.mean([r['majority_correct'] for r in sr]):.3f}".rjust(14)
                else:
                    row += "—".rjust(14)
            print(row)

    # Topology benefit vs independent
    if "independent" in all_data:
        print("\n=== TOPOLOGY BENEFIT (vs independent) ===\n")
        indep_acc = np.mean([r["majority_correct"] for r in all_data["independent"]])
        for topo, results in sorted(all_data.items()):
            if topo == "independent":
                continue
            acc = np.mean([r["majority_correct"] for r in results])
            benefit = acc - indep_acc
            print(f"  B({topo}) = {benefit:+.3f}  (acc={acc:.3f} vs indep={indep_acc:.3f})")

    # Save merged summary
    summary = {}
    for topo, results in all_data.items():
        summary[topo] = {
            "accuracy": float(np.mean([r["majority_correct"] for r in results])),
            "n_results": len(results),
            "by_subject": {
                subj: float(np.mean([r["majority_correct"]
                                     for r in results if r.get("subject") == subj]))
                for subj in subjects
                if any(r.get("subject") == subj for r in results)
            },
        }
    summary_file = output_dir / "phase2b_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2b full MMLU topology experiment")
    parser.add_argument("--topology", type=str, help="Topology to run (or 'all')")
    parser.add_argument("--merge", action="store_true", help="Merge and compare results")
    parser.add_argument("--status", action="store_true", help="Show progress status")
    parser.add_argument("--n-questions", type=int, default=N_QUESTIONS)
    args = parser.parse_args()

    output_dir = RESULTS_DIR / "phase2b"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.status:
        show_status(output_dir)
        return

    if args.merge:
        merge_results(output_dir)
        return

    questions = load_questions("mmlu", n=args.n_questions, seed=42)
    print(f"Loaded {len(questions)} MMLU questions")
    subject_dist = Counter(q.get("subject") for q in questions)
    for subj in sorted(subject_dist):
        print(f"  {subj}: {subject_dist[subj]} questions")
    print(f"\nExperiment: {len(questions)} questions × {N_RUNS} runs × "
          f"{len(CORE_MODELS)} agents × {N_ROUNDS} rounds")
    print(f"Estimated Bedrock calls per topology: ~{len(questions) * N_RUNS * len(CORE_MODELS) * N_ROUNDS:,}")

    if args.topology == "all":
        for topo in ALL_TOPOLOGIES:
            results_file = output_dir / f"phase2b_{topo}.json"
            if results_file.exists():
                data = json.loads(results_file.read_text())
                if data.get("n_runs_completed", 0) >= N_RUNS:
                    print(f"  Skipping {topo} — already complete ({data['n_runs_completed']} runs)")
                    continue
            print(f"\n=== Running topology: {topo} ===")
            run_single_topology(topo, output_dir, questions)
            if _shutdown:
                print("Shutdown — stopping topology loop")
                break
    elif args.topology:
        if args.topology not in ALL_TOPOLOGIES:
            print(f"Unknown topology: {args.topology}. Choose from: {ALL_TOPOLOGIES}")
            sys.exit(1)
        run_single_topology(args.topology, output_dir, questions)
    else:
        parser.print_help()
        print(f"\nAvailable topologies: {ALL_TOPOLOGIES}")


if __name__ == "__main__":
    main()
