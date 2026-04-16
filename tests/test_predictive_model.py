import numpy as np
from topology_tax.predictive_model import TopologyPredictor
from topology_tax.topologies import build_topology, compute_graph_features

def test_predictor_fit_and_predict():
    features = []
    targets = []
    for topo_name in ["complete", "star", "ring", "chain", "independent"]:
        G = build_topology(topo_name, n_agents=5)
        feat = compute_graph_features(G)
        features.append(feat)
        targets.append(np.random.RandomState(42).uniform(-0.5, 0.5))
    pred = TopologyPredictor()
    pred.fit(features, targets)
    preds = pred.predict(features)
    assert len(preds) == 5
    assert pred.r_squared is not None

def test_predictor_feature_importance():
    features = []
    targets = []
    rng = np.random.RandomState(0)
    for topo_name in ["complete", "star", "ring", "chain", "independent", "erdos_renyi", "small_world", "sparse_random"]:
        G = build_topology(topo_name, n_agents=5, seed=0, target_edge_count=4)
        feat = compute_graph_features(G)
        features.append(feat)
        targets.append(feat["edge_density"] * 0.5 + rng.normal(0, 0.1))
    pred = TopologyPredictor(model_type="random_forest")
    pred.fit(features, targets)
    importance = pred.feature_importance()
    assert len(importance) > 0
    assert "edge_density" in importance

def test_predictor_cross_validate():
    features = []
    targets = []
    rng = np.random.RandomState(0)
    for i in range(20):
        G = build_topology("erdos_renyi", n_agents=5, seed=i)
        feat = compute_graph_features(G)
        features.append(feat)
        targets.append(feat["edge_density"] + rng.normal(0, 0.1))
    pred = TopologyPredictor()
    scores = pred.cross_validate(features, targets, cv=3)
    assert len(scores) == 3
