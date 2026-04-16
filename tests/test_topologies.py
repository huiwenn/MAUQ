import networkx as nx
import numpy as np
from topology_tax.topologies import (
    build_topology,
    compute_graph_features,
    TOPOLOGY_REGISTRY,
)


def test_complete_topology():
    G = build_topology("complete", n_agents=5)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 10
    assert nx.is_connected(G.to_undirected())


def test_star_topology():
    G = build_topology("star", n_agents=5)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 4
    degrees = dict(G.degree())
    hub = max(degrees, key=degrees.get)
    assert degrees[hub] == 4


def test_ring_topology():
    G = build_topology("ring", n_agents=5)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 5
    for _, d in G.degree():
        assert d == 2


def test_chain_topology():
    G = build_topology("chain", n_agents=5)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 4
    assert nx.is_directed(G)


def test_binary_tree_topology():
    G = build_topology("binary_tree", n_agents=5)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 4


def test_erdos_renyi_topology():
    G = build_topology("erdos_renyi", n_agents=5, seed=42)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() >= 1


def test_small_world_topology():
    G = build_topology("small_world", n_agents=5, seed=42)
    assert G.number_of_nodes() == 5


def test_independent_topology():
    G = build_topology("independent", n_agents=5)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 0


def test_context_matched_topology():
    G = build_topology("context_matched", n_agents=5, seed=42,
                       target_edge_count=4)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 4


def test_sparse_random_topology():
    G = build_topology("sparse_random", n_agents=5, seed=42)
    assert G.number_of_nodes() == 5


def test_graph_features():
    G = build_topology("complete", n_agents=5)
    features = compute_graph_features(G)
    assert features["n_nodes"] == 5
    assert features["n_edges"] == 10
    assert features["diameter"] == 1
    assert features["edge_density"] == 1.0
    assert 0.0 <= features["clustering_coefficient"] <= 1.0
    assert features["algebraic_connectivity"] > 0
    assert features["spectral_gap"] > 0
    assert features["avg_path_length"] == 1.0
    assert features["degree_entropy"] >= 0.0
    assert features["max_betweenness"] >= 0.0


def test_graph_features_independent():
    G = build_topology("independent", n_agents=5)
    features = compute_graph_features(G)
    assert features["edge_density"] == 0.0
    assert features["diameter"] == float("inf")
    assert features["algebraic_connectivity"] == 0.0


def test_all_topologies_registered():
    from topology_tax.config import TOPOLOGY_NAMES
    for name in TOPOLOGY_NAMES:
        G = build_topology(name, n_agents=5, seed=42, target_edge_count=4)
        assert G.number_of_nodes() == 5


def test_communication_order():
    G = build_topology("chain", n_agents=5)
    from topology_tax.topologies import get_communication_order
    order = get_communication_order(G)
    assert len(order) == 5
    assert order[0] == 0
