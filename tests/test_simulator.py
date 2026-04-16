import numpy as np
from topology_tax.simulator import NoisyChannelSimulator
from topology_tax.topologies import build_topology


def test_independent_accuracy():
    sim = NoisyChannelSimulator(
        n_agents=5,
        agent_accuracies=[0.8] * 5,
        anchoring_matrix=np.zeros((5, 5)),
        seed=42,
    )
    G = build_topology("independent", n_agents=5)
    results = sim.run(G, n_questions=1000)
    assert 0.85 < results["majority_accuracy"] < 0.99


def test_star_amplification():
    anchoring = np.zeros((5, 5))
    for j in range(1, 5):
        anchoring[j, 0] = 0.9
    sim = NoisyChannelSimulator(
        n_agents=5,
        agent_accuracies=[0.3, 0.8, 0.8, 0.8, 0.8],
        anchoring_matrix=anchoring,
        seed=42,
    )
    G_star = build_topology("star", n_agents=5)
    G_indep = build_topology("independent", n_agents=5)
    r_star = sim.run(G_star, n_questions=1000)
    r_indep = sim.run(G_indep, n_questions=1000)
    assert r_star["majority_accuracy"] < r_indep["majority_accuracy"]
    assert r_star["decomposition"]["amplification"] > 0


def test_complete_homogeneous_absorption():
    n = 5
    anchoring = np.full((n, n), 0.3)
    np.fill_diagonal(anchoring, 0.0)
    sim = NoisyChannelSimulator(n, [0.6] * n, anchoring, seed=42)
    G_comp = build_topology("complete", n_agents=n)
    G_indep = build_topology("independent", n_agents=n)
    r_comp = sim.run(G_comp, n_questions=1000)
    r_indep = sim.run(G_indep, n_questions=1000)
    assert r_comp["decomposition"]["absorption"] >= 0


def test_simulator_returns_per_question_data():
    sim = NoisyChannelSimulator(
        n_agents=3,
        agent_accuracies=[0.7] * 3,
        anchoring_matrix=np.full((3, 3), 0.2) - np.diag([0.2] * 3),
        seed=0,
    )
    G = build_topology("ring", n_agents=3)
    results = sim.run(G, n_questions=50, n_runs=10)
    assert len(results["per_question_correct"]) == 50
    assert results["per_question_correct"].shape == (50, 10)


def test_scaling_agents():
    for n in [3, 5, 10, 20]:
        accs = [0.7] * n
        alpha = np.full((n, n), 0.3) - np.diag([0.3] * n)
        sim = NoisyChannelSimulator(n, accs, alpha, seed=42)
        G = build_topology("complete", n_agents=n)
        results = sim.run(G, n_questions=100)
        assert "majority_accuracy" in results
