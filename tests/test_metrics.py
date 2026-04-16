import numpy as np
from topology_tax.metrics import (
    topology_benefit,
    topology_sensitivity_mi,
    amplification,
    absorption,
    decompose_topology_tax,
    jsd_between_topologies,
)


def test_topology_benefit_positive():
    correct_G = np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 0])
    correct_indep = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0])
    b = topology_benefit(correct_G, correct_indep)
    assert b > 0
    assert abs(b - 0.2) < 1e-10


def test_topology_benefit_negative():
    correct_G = np.array([1, 0, 0, 0, 0])
    correct_indep = np.array([1, 1, 1, 0, 0])
    b = topology_benefit(correct_G, correct_indep)
    assert b < 0


def test_topology_benefit_zero():
    correct = np.array([1, 1, 0, 0])
    b = topology_benefit(correct, correct)
    assert b == 0.0


def test_amplification_detects_wrong_consensus():
    answers_G = [["A", "A", "A", "A", "B"]] * 10
    answers_indep = [["A", "B", "C", "A", "B"]] * 10
    correct_answer = "C"
    a_plus = amplification(answers_G, answers_indep, correct_answer,
                           consensus_threshold=0.8)
    assert a_plus > 0


def test_absorption_detects_correction():
    answers_G = [["C", "C", "C", "C", "C"]] * 10
    answers_indep = [["A", "B", "C", "A", "B"]] * 10
    correct_answer = "C"
    a_minus = absorption(answers_G, answers_indep, correct_answer)
    assert a_minus > 0


def test_decompose_returns_all_components():
    correct_G = np.array([1, 1, 1, 0, 0])
    correct_indep = np.array([1, 0, 0, 0, 0])
    answers_G = [["C", "C", "C", "A", "B"]] * 5
    answers_indep = [["C", "A", "B", "A", "B"]] * 5
    correct_answer = "C"
    result = decompose_topology_tax(
        correct_G, correct_indep,
        answers_G, answers_indep,
        correct_answer
    )
    assert "benefit" in result
    assert "amplification" in result
    assert "absorption" in result
    assert "residual" in result
    assert "sensitivity" in result
    assert "tax" in result
    assert result["tax"] == -result["benefit"]


def test_sensitivity_mi_nonnegative():
    correctness_by_topology = {
        "complete": np.array([1, 1, 0, 0, 1]),
        "star": np.array([0, 1, 0, 1, 1]),
        "ring": np.array([1, 0, 0, 0, 0]),
    }
    s = topology_sensitivity_mi(correctness_by_topology)
    assert s >= 0.0


def test_jsd_identical_distributions():
    dist_a = np.array([0.5, 0.3, 0.2])
    dist_b = np.array([0.5, 0.3, 0.2])
    assert jsd_between_topologies(dist_a, dist_b) < 1e-10


def test_jsd_different_distributions():
    dist_a = np.array([1.0, 0.0, 0.0])
    dist_b = np.array([0.0, 0.0, 1.0])
    jsd = jsd_between_topologies(dist_a, dist_b)
    assert jsd > 0.5
