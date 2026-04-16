import numpy as np
from collections import Counter
from typing import Optional


def topology_benefit(
    correct_G: np.ndarray,
    correct_indep: np.ndarray,
) -> float:
    return float(correct_G.mean() - correct_indep.mean())


def amplification(
    answers_G: list[list[str]],
    answers_indep: list[list[str]],
    correct_answer: str,
    consensus_threshold: float = 0.8,
) -> float:
    def wrong_consensus_rate(runs: list[list[str]]) -> float:
        count = 0
        for run_answers in runs:
            counter = Counter(run_answers)
            most_common, freq = counter.most_common(1)[0]
            if freq / len(run_answers) >= consensus_threshold and most_common != correct_answer:
                count += 1
        return count / len(runs) if runs else 0.0

    wc_G = wrong_consensus_rate(answers_G)
    wc_indep = wrong_consensus_rate(answers_indep)
    return max(0.0, wc_G - wc_indep)


def absorption(
    answers_G: list[list[str]],
    answers_indep: list[list[str]],
    correct_answer: str,
) -> float:
    def correct_rate(runs: list[list[str]]) -> float:
        correct_counts = []
        for run_answers in runs:
            correct_counts.append(
                sum(1 for a in run_answers if a == correct_answer) / len(run_answers)
            )
        return np.mean(correct_counts) if correct_counts else 0.0

    cr_G = correct_rate(answers_G)
    cr_indep = correct_rate(answers_indep)
    return max(0.0, cr_G - cr_indep)


def topology_sensitivity_mi(
    correctness_by_topology: dict[str, np.ndarray],
) -> float:
    all_correct = np.concatenate(list(correctness_by_topology.values()))
    n_total = len(all_correct)
    p_correct = all_correct.mean()
    p_wrong = 1.0 - p_correct

    if p_correct == 0 or p_correct == 1:
        return 0.0

    H_Y = -p_correct * np.log2(p_correct) - p_wrong * np.log2(p_wrong)

    H_Y_given_G = 0.0
    for topo_name, correct_arr in correctness_by_topology.items():
        p_g = len(correct_arr) / n_total
        p_c_given_g = correct_arr.mean()
        p_w_given_g = 1.0 - p_c_given_g
        if 0 < p_c_given_g < 1:
            h = -p_c_given_g * np.log2(p_c_given_g) - p_w_given_g * np.log2(p_w_given_g)
        else:
            h = 0.0
        H_Y_given_G += p_g * h

    return float(max(0.0, H_Y - H_Y_given_G))


def jsd_between_topologies(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
) -> float:
    dist_a = np.asarray(dist_a, dtype=float)
    dist_b = np.asarray(dist_b, dtype=float)
    dist_a = dist_a / dist_a.sum()
    dist_b = dist_b / dist_b.sum()
    m = 0.5 * (dist_a + dist_b)

    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    return 0.5 * kl(dist_a, m) + 0.5 * kl(dist_b, m)


def decompose_topology_tax(
    correct_G: np.ndarray,
    correct_indep: np.ndarray,
    answers_G: list[list[str]],
    answers_indep: list[list[str]],
    correct_answer: str,
    consensus_threshold: float = 0.8,
) -> dict[str, float]:
    b = topology_benefit(correct_G, correct_indep)
    a_plus = amplification(answers_G, answers_indep, correct_answer, consensus_threshold)
    a_minus = absorption(answers_G, answers_indep, correct_answer)
    residual = b - a_minus + a_plus

    correctness_by_topology = {
        "G": correct_G,
        "independent": correct_indep,
    }
    s = topology_sensitivity_mi(correctness_by_topology)

    return {
        "benefit": b,
        "tax": -b,
        "amplification": a_plus,
        "absorption": a_minus,
        "residual": residual,
        "sensitivity": s,
    }
