import numpy as np


def permutation_test_topology_effect(correct_by_topology, n_permutations=1000, seed=None):
    rng = np.random.RandomState(seed)
    groups = list(correct_by_topology.values())
    all_data = np.concatenate(groups)
    group_sizes = [len(g) for g in groups]

    def test_statistic(data, sizes):
        offset = 0
        means = []
        for s in sizes:
            means.append(data[offset:offset + s].mean())
            offset += s
        grand_mean = data.mean()
        return sum(s * (m - grand_mean) ** 2 for s, m in zip(sizes, means))

    observed = test_statistic(all_data, group_sizes)
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(all_data)
        if test_statistic(perm, group_sizes) >= observed:
            count += 1
    return float(observed), float((count + 1) / (n_permutations + 1))


def compute_partial_eta_squared(correct_by_topology):
    groups = list(correct_by_topology.values())
    all_data = np.concatenate(groups)
    grand_mean = all_data.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_data - grand_mean) ** 2)
    if ss_total == 0:
        return 0.0
    return float(ss_between / ss_total)


def benjamini_hochberg(p_values, alpha=0.05):
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * n
    max_rejected = -1
    for rank, (orig_idx, p) in enumerate(indexed, 1):
        if p <= alpha * rank / n:
            max_rejected = rank
    if max_rejected > 0:
        for rank, (orig_idx, p) in enumerate(indexed, 1):
            if rank <= max_rejected:
                rejected[orig_idx] = True
    return rejected
