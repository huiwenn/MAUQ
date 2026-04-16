# Schur Complement Meets Multi-Agent DAGs: Hierarchical Uncertainty Decomposition via Exact Elimination

## Description
We propose applying the Schur complement -- the key mathematical tool from CLUES (which decomposes single-model uncertainty into ambiguity and instability) -- to the multi-agent setting by defining a joint precision matrix over all agent-node pairs in the DPIMPR reasoning DAG and using Schur complements to perform exact marginalization over subsets of agents or reasoning steps. The key mechanism: represent the full system's uncertainty as a multivariate Gaussian over the latent "belief states" at each node of the reasoning DAG (justified by the central limit theorem for aggregated token probabilities). The precision matrix Q encodes the conditional independence structure: Q_{ij} = 0 if agents/nodes i and j are conditionally independent given the rest. The Schur complement Q/Q_{AA} for a subset A of nodes gives the marginal precision of the remaining nodes after exactly integrating out A. This enables three novel operations: (1) Agent elimination -- compute the uncertainty of the system with a specific agent "integrated out," revealing that agent's marginal contribution to confidence. (2) Step elimination -- compute the uncertainty of the final answer with intermediate reasoning steps integrated out, revealing which steps are load-bearing. (3) Hierarchical decomposition -- recursively eliminate nodes in topological order, producing a multi-resolution uncertainty pyramid from fine-grained (per-node) to coarse (whole-system). This extends CLUES' two-component decomposition to an arbitrary-depth hierarchy over a DAG structure.

## Gap Filled
CLUES decomposes into ambiguity vs. instability using the Schur complement, but only for single models and only into two components. MATU uses tensor decomposition but loses the graph structure. DPIMPR accumulates confidence along the DAG but does not provide a principled way to ask "what happens to uncertainty if we remove this agent or this reasoning step?" The Schur complement approach fills this gap with a mathematically exact tool for marginalization that respects the DAG's conditional independence structure. It also connects to the broader Gaussian graphical model literature, which is well-understood and provides efficient algorithms.

## Experiments
1. Fit a Gaussian graphical model to the belief states at each node of the DPIMPR reasoning DAG, estimating the precision matrix Q using graphical lasso or neighborhood selection.
2. Compute Schur complements for: (a) each individual agent (agent elimination), (b) each reasoning step (step elimination), (c) hierarchical elimination in topological order.
3. Define "marginal agent importance" as the change in system uncertainty when an agent is eliminated; use this to rank agents by their contribution.
4. Define "step criticality" as the change in system uncertainty when a reasoning step is eliminated; use this to identify load-bearing reasoning steps.
5. Evaluate the hierarchical uncertainty pyramid as a UQ measure: compare AUROC at each level of the hierarchy to DPIMPR's single-level confidence.
6. Test on "confident but wrong" cases: do load-bearing steps in wrong answers have high step criticality but low local confidence, indicating a single point of failure?
7. Benchmark on TriviaQA, StrategyQA, and HotpotQA (where the multi-step structure is most meaningful).

## Risks
- The Gaussian assumption for belief states is a strong modeling choice; LLM output distributions are highly non-Gaussian (multimodal, heavy-tailed), and the CLT justification only holds for large sample sizes that may not be available.
- Fitting a Gaussian graphical model requires many observations per graph structure, but each question produces a single DAG; the model must be fit across questions, assuming that the precision structure is shared, which may not hold.
- The Schur complement is exact for Gaussians but only approximate for non-Gaussian distributions; the approximation error may dominate the signal.
- The method produces many uncertainty quantities (one per elimination) but it is unclear which ones to use for practical decisions; the hierarchy may be informative but difficult to operationalize.
- Reviewers familiar with CLUES may view this as an incremental extension (same math, bigger graph) rather than a conceptual advance.
