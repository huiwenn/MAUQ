#!/usr/bin/env python
"""
Fast UQ computation for all debate traces without visualization.
"""

import sys
sys.path.insert(0, 'src')

import os
import json
from debate_graph_viz import DebateGraphBuilder, GraphBasedUQ, DebateDAGUncertainty

def compute_all_debate_uq(traces_dir='results/debate', output_file='results/debate_uq_metrics.json'):
    """Compute UQ for all debates and save to JSON."""

    # Find all traces
    trace_files = sorted([
        f for f in os.listdir(traces_dir)
        if f.startswith('traces_') and f.endswith('.json')
    ])

    print(f"Computing UQ for {len(trace_files)} debate traces...")

    builder = DebateGraphBuilder()
    all_metrics = []

    for i, trace_file in enumerate(trace_files):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(trace_files)} traces...")

        # Load trace
        trace_path = os.path.join(traces_dir, trace_file)
        with open(trace_path, 'r') as f:
            trace_data = json.load(f)

        # Build graph
        graph = builder.build_graph_from_trace(trace_data)

        # Calculate basic metrics
        metrics = builder.calculate_graph_metrics()
        metrics['trace_file'] = trace_file

        # Graph-based UQ
        uq_calculator = GraphBasedUQ(graph)
        uq_results = uq_calculator.calculate_total_uncertainty()
        agent_profiles = uq_calculator.calculate_agent_uncertainty_profile()

        # DAG-based UQ
        dag_uq_calculator = DebateDAGUncertainty(graph)
        dag_uq_results = dag_uq_calculator.compute_dag_uncertainty(
            question=f"Debate on answer: {trace_data.get('answer', 'unknown')}",
            ground_truth_answer=str(trace_data.get('answer', 'unknown'))
        )

        # Merge results
        combined_uq = {**uq_results, **dag_uq_results}
        metrics['uq_results'] = combined_uq
        metrics['agent_profiles'] = agent_profiles

        all_metrics.append(metrics)

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n✓ Computed UQ for {len(all_metrics)} debates")
    print(f"✓ Saved to {output_file}")

    return all_metrics

if __name__ == "__main__":
    compute_all_debate_uq()
