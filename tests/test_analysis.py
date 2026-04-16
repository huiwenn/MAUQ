import numpy as np
from topology_tax.analysis import (
    permutation_test_topology_effect,
    compute_partial_eta_squared,
    benjamini_hochberg,
)

def test_permutation_test_significant():
    rng = np.random.RandomState(42)
    correct_by_topology = {
        "A": rng.binomial(1, 0.8, size=100),
        "B": rng.binomial(1, 0.4, size=100),
    }
    stat, p_value = permutation_test_topology_effect(correct_by_topology, n_permutations=500, seed=42)
    assert p_value < 0.05

def test_permutation_test_not_significant():
    rng = np.random.RandomState(42)
    correct_by_topology = {
        "A": rng.binomial(1, 0.6, size=100),
        "B": rng.binomial(1, 0.6, size=100),
    }
    stat, p_value = permutation_test_topology_effect(correct_by_topology, n_permutations=500, seed=42)
    assert p_value > 0.05

def test_partial_eta_squared_range():
    rng = np.random.RandomState(42)
    correct_by_topology = {
        "A": rng.binomial(1, 0.9, size=50),
        "B": rng.binomial(1, 0.3, size=50),
        "C": rng.binomial(1, 0.6, size=50),
    }
    eta_sq = compute_partial_eta_squared(correct_by_topology)
    assert 0.0 <= eta_sq <= 1.0

def test_benjamini_hochberg():
    p_values = [0.01, 0.04, 0.03, 0.20, 0.50]
    rejected = benjamini_hochberg(p_values, alpha=0.05)
    assert rejected[0] is True
    assert rejected[3] is False
    assert rejected[4] is False
