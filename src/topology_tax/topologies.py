import networkx as nx
import numpy as np
from typing import Optional


def _complete(n: int, **kw) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j)
    return G


def _star(n: int, **kw) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    hub = 0
    for i in range(1, n):
        G.add_edge(hub, i)
    return G


def _ring(n: int, **kw) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        j = (i + 1) % n
        G.add_edge(i, j)
    return G


def _chain(n: int, **kw) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    return G


def _binary_tree(n: int, **kw) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n:
            G.add_edge(i, left)
        if right < n:
            G.add_edge(i, right)
    return G


def _erdos_renyi(n: int, seed: Optional[int] = None, **kw) -> nx.DiGraph:
    rng = np.random.RandomState(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < 0.4:
                G.add_edge(i, j)
    if G.number_of_edges() == 0:
        G.add_edge(0, 1)
    return G


def _small_world(n: int, seed: Optional[int] = None, **kw) -> nx.DiGraph:
    if n < 4:
        return _ring(n)
    k = min(2, n - 1)
    Gu = nx.watts_strogatz_graph(n, k, 0.3, seed=seed)
    G = Gu.to_directed()
    return G


def _independent(n: int, **kw) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    return G


def _context_matched(n: int, seed: Optional[int] = None,
                     target_edge_count: int = 4, **kw) -> nx.DiGraph:
    rng = np.random.RandomState(seed)
    all_edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    rng.shuffle(all_edges)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u, v in all_edges[:target_edge_count]:
        G.add_edge(u, v)
    return G


def _sparse_random(n: int, seed: Optional[int] = None, **kw) -> nx.DiGraph:
    rng = np.random.RandomState(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < 0.2:
                G.add_edge(i, j)
    if G.number_of_edges() == 0:
        G.add_edge(0, 1)
    return G


TOPOLOGY_REGISTRY = {
    "complete": _complete,
    "star": _star,
    "ring": _ring,
    "chain": _chain,
    "binary_tree": _binary_tree,
    "erdos_renyi": _erdos_renyi,
    "small_world": _small_world,
    "independent": _independent,
    "context_matched": _context_matched,
    "sparse_random": _sparse_random,
}


def build_topology(name: str, n_agents: int = 5, **kwargs) -> nx.DiGraph:
    if name not in TOPOLOGY_REGISTRY:
        raise ValueError(f"Unknown topology: {name}. Choose from {list(TOPOLOGY_REGISTRY)}")
    return TOPOLOGY_REGISTRY[name](n_agents, **kwargs)


def get_communication_order(G: nx.DiGraph) -> list[int]:
    if G.number_of_edges() == 0:
        return list(G.nodes())
    try:
        return list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        start = max(G.nodes(), key=lambda n: G.out_degree(n))
        visited = list(nx.bfs_tree(G, start).nodes())
        remaining = [n for n in G.nodes() if n not in visited]
        return visited + remaining


def compute_graph_features(G: nx.DiGraph) -> dict[str, float]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    Gu = G.to_undirected()
    m_undirected = Gu.number_of_edges()
    max_undirected_edges = n * (n - 1) // 2

    if nx.is_connected(Gu):
        diameter = nx.diameter(Gu)
        avg_path = nx.average_shortest_path_length(Gu)
    else:
        diameter = float("inf")
        avg_path = float("inf")

    clustering = nx.average_clustering(Gu) if m > 0 else 0.0

    if m > 0 and n > 2:
        L = nx.laplacian_matrix(Gu).astype(float)
        try:
            eigenvalues = np.sort(np.linalg.eigvalsh(L.toarray()))
            algebraic_connectivity = float(eigenvalues[1]) if n > 1 else 0.0
            spectral_gap = float(eigenvalues[-1]) if n > 1 else 0.0
        except Exception:
            algebraic_connectivity = 0.0
            spectral_gap = 0.0
    else:
        algebraic_connectivity = 0.0
        spectral_gap = 0.0

    degrees = [d for _, d in Gu.degree()]
    if sum(degrees) > 0:
        probs = np.array(degrees, dtype=float) / sum(degrees)
        degree_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        degree_entropy = 0.0

    betweenness = nx.betweenness_centrality(Gu)
    max_betweenness = max(betweenness.values()) if betweenness else 0.0

    return {
        "n_nodes": n,
        "n_edges": m,
        "edge_density": m_undirected / max_undirected_edges if max_undirected_edges > 0 else 0.0,
        "diameter": diameter,
        "clustering_coefficient": clustering,
        "algebraic_connectivity": algebraic_connectivity,
        "spectral_gap": spectral_gap,
        "avg_path_length": avg_path,
        "degree_entropy": degree_entropy,
        "max_betweenness": max_betweenness,
    }
