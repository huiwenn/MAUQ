# Information-Geometric Uncertainty: Reasoning Trajectories as Curves on Statistical Manifolds

## Description
We propose recasting multi-agent reasoning trajectories as curves on a statistical manifold and using the geometry of this manifold to define a new class of uncertainty measures. Each reasoning step of an agent induces a probability distribution over next tokens (or over answer candidates); a full trajectory is thus a sequence of distributions, which traces a curve on the simplex. Using the Fisher information metric, we compute three geometric quantities: (1) trajectory curvature -- high curvature indicates that the agent is rapidly revising its beliefs, signaling instability; (2) trajectory length -- long trajectories on the manifold indicate the agent explored many distributional states, which can signal either thorough reasoning or confused wandering; (3) inter-trajectory geodesic distance -- the minimum-length path between two agents' trajectory endpoints on the Fisher manifold, which is a more principled measure of disagreement than KL divergence (it is symmetric, satisfies the triangle inequality, and accounts for the manifold's curvature). We define a composite "geometric uncertainty" score as a function of these three quantities. The crucial mechanism: standard divergence measures (KL, JS, total variation) treat each reasoning step independently, while the geometric approach captures the *shape* of how beliefs evolve, enabling detection of pathological trajectories (e.g., circular reasoning, where the trajectory loops back on itself on the manifold).

## Gap Filled
UProp uses NLL and PMI -- both are pointwise measures that do not capture trajectory shape. DPIMPR accumulates evidence along DAG paths but does not model the distributional geometry of the trajectory. MATU decomposes via tensors but in a discrete algebraic framework, not a continuous geometric one. CoE uses entropy, another pointwise measure. Information geometry has been applied to single neural networks (natural gradient descent) but never to multi-agent reasoning trajectories. This is a genuinely new lens on the problem that could reveal uncertainty patterns invisible to existing methods.

## Experiments
1. Instrument agents to log their token-level probability distributions at each reasoning step, building trajectory curves on the probability simplex.
2. Compute Fisher information matrices, geodesic distances, and trajectory curvatures for 1000+ problem instances with 5 agents each.
3. Train a classifier on geometric features alone (curvature, length, geodesic distance) to predict answer correctness; compare AUROC to DPIMPR.
4. Analyze "confident but wrong" cases: do they exhibit distinct geometric signatures (e.g., low curvature indicating premature convergence, or short trajectories indicating insufficient exploration)?
5. Combine geometric features with DPIMPR confidence for a hybrid predictor.
6. Compare geodesic distance to standard disagreement measures (JS divergence, CoE) for detecting agent disagreement.

## Risks
- Fisher information matrices for LLM outputs are enormous (vocabulary-sized), requiring dimensionality reduction that may lose important structure.
- The computational cost of geodesic distance on high-dimensional simplices is non-trivial; approximations (e.g., projecting to a lower-dimensional submanifold) may be required.
- Token-level probability distributions may not be available for closed-source models, limiting applicability to open-source models only.
- The geometric features may be highly correlated with simpler statistics (entropy of the distribution at each step, which is the trace of the Fisher matrix), making the full geometric machinery unnecessary overhead.
- The information geometry framework is elegant but may not provide practical improvements over simpler trajectory statistics, leading reviewers to view it as "math for math's sake."
