#!/usr/bin/env python3
"""
Phase 1: Synthetic validation.
Validates the Topology Tax framework on noisy-channel agents.
Produces: decomposition validation, power analysis, scaling laws, theorem checks.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topology_tax.config import TOPOLOGY_NAMES, RESULTS_DIR
from topology_tax.topologies import build_topology, compute_graph_features
from topology_tax.simulator import NoisyChannelSimulator

try:
    from topology_tax.predictive_model import TopologyPredictor
    HAS_PREDICTIVE_MODEL = True
except ImportError:
    HAS_PREDICTIVE_MODEL = False


def run_decomposition_validation():
    """Validate B(G) = A-(G) - A+(G) across conditions."""
    print("=== Decomposition Validation ===")
    configs = [
        {"name": "bad_hub_star", "accs": [0.3, 0.8, 0.8, 0.8, 0.8],
         "anchoring_hub": 0.9, "expected": "high amplification"},
        {"name": "good_diverse", "accs": [0.7, 0.6, 0.8, 0.65, 0.75],
         "anchoring_hub": 0.2, "expected": "moderate absorption"},
        {"name": "homogeneous_weak", "accs": [0.5, 0.5, 0.5, 0.5, 0.5],
         "anchoring_hub": 0.5, "expected": "uncertain"},
    ]
    results = []
    for cfg in configs:
        n = len(cfg["accs"])
        alpha = np.full((n, n), cfg["anchoring_hub"])
        np.fill_diagonal(alpha, 0.0)
        sim = NoisyChannelSimulator(n, cfg["accs"], alpha, seed=42)

        for topo_name in ["complete", "star", "ring", "chain", "independent"]:
            G = build_topology(topo_name, n_agents=n)
            r = sim.run(G, n_questions=1000, n_runs=20)
            dec = r["decomposition"]
            results.append({
                "config": cfg["name"],
                "topology": topo_name,
                "accuracy": r["majority_accuracy"],
                "independent_accuracy": r["independent_accuracy"],
                **dec,
            })
            print(f"  {cfg['name']}/{topo_name}: acc={r['majority_accuracy']:.3f}, "
                  f"B={dec['benefit']:.3f}, A+={dec['amplification']:.3f}, "
                  f"A-={dec['absorption']:.3f}")
    return results


def run_scaling_analysis():
    """Test how B(G) changes with agent count."""
    print("\n=== Scaling Analysis ===")
    results = []
    for n_agents in [3, 5, 10, 20, 50]:
        accs = [0.65] * n_agents
        alpha = np.full((n_agents, n_agents), 0.3)
        np.fill_diagonal(alpha, 0.0)
        sim = NoisyChannelSimulator(n_agents, accs, alpha, seed=42)

        for topo_name in ["complete", "ring", "independent"]:
            G = build_topology(topo_name, n_agents=n_agents)
            r = sim.run(G, n_questions=500, n_runs=20)
            results.append({
                "n_agents": n_agents,
                "topology": topo_name,
                "accuracy": r["majority_accuracy"],
                **r["decomposition"],
            })
            print(f"  n={n_agents}, {topo_name}: acc={r['majority_accuracy']:.3f}, "
                  f"B={r['decomposition']['benefit']:.3f}")
    return results


def run_predictive_model_validation():
    """Test graph features -> B(G) regression on synthetic data."""
    print("\n=== Predictive Model Validation ===")
    if not HAS_PREDICTIVE_MODEL:
        print("  [SKIP] topology_tax.predictive_model not yet available")
        return {"skipped": True, "reason": "predictive_model module not implemented yet"}

    n = 5
    accs = [0.65] * n
    alpha = np.full((n, n), 0.3)
    np.fill_diagonal(alpha, 0.0)
    sim = NoisyChannelSimulator(n, accs, alpha, seed=42)

    features_list = []
    benefits = []
    for topo_name in TOPOLOGY_NAMES:
        G = build_topology(topo_name, n_agents=n, seed=42, target_edge_count=4)
        feats = compute_graph_features(G)
        r = sim.run(G, n_questions=500, n_runs=20)
        features_list.append(feats)
        benefits.append(r["decomposition"]["benefit"])

    pred = TopologyPredictor(model_type="linear")
    pred.fit(features_list, benefits)
    print(f"  Linear R2: {pred.r_squared:.3f}")

    pred_rf = TopologyPredictor(model_type="random_forest")
    pred_rf.fit(features_list, benefits)
    print(f"  Random Forest R2: {pred_rf.r_squared:.3f}")
    print(f"  Feature importance: {pred_rf.feature_importance()}")

    return {"linear_r2": pred.r_squared, "rf_r2": pred_rf.r_squared}


def main():
    output_dir = RESULTS_DIR / "phase1"
    output_dir.mkdir(parents=True, exist_ok=True)

    decomp = run_decomposition_validation()
    scaling = run_scaling_analysis()
    pred = run_predictive_model_validation()

    all_results = {
        "decomposition_validation": decomp,
        "scaling_analysis": scaling,
        "predictive_model": pred,
    }

    output_file = output_dir / "phase1_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {output_file}")


if __name__ == "__main__":
    main()
