# Self-Healing Uncertainty: Closed-Loop UQ with Adaptive Graph Rewiring

## Description
We propose a closed-loop system where uncertainty estimates actively modify the agent communication topology in real time, creating a feedback loop between UQ and agent orchestration. The key insight comes from combining two findings: (a) Self-Healing Router showed that 93% of control-plane LLM calls can be replaced by graph routing, and (b) AgentConductor showed dynamic topologies outperform static ones. Our system, "Self-Healing Uncertainty" (SHU), works as follows: after each round of agent communication, DPIMPR-style confidence is computed at every node in the reasoning DAG. Nodes with confidence below a threshold trigger one of three graph rewiring operations: (i) EXPAND -- add a new agent connection to bring in diverse evidence (when uncertainty is high and disagreement is low, suggesting the group lacks information), (ii) PRUNE -- remove an agent connection that is propagating correlated errors (when uncertainty is high and a specific edge has high mutual information with the error pattern), (iii) CHALLENGE -- insert an adversarial agent specifically tasked with attacking the low-confidence node's reasoning (when confidence is moderate but the reasoning chain has structural weaknesses). The rewiring rules are implemented as a lightweight graph neural network trained on (DAG state, rewiring action, outcome) triples, replacing the expensive LLM-based controller of AgentConductor. The system converges when all nodes exceed the confidence threshold or a compute budget is exhausted.

## Gap Filled
Current multi-agent UQ is open-loop: you compute uncertainty and report it, but the system does not adapt. AgentConductor adapts topology but not based on uncertainty -- it uses task performance. Self-Healing Router replaces control-plane LLMs but does not use UQ signals. Gas Town's bead system provides version-controlled state machines but no UQ-driven adaptation. SHU closes the loop: uncertainty drives topology, which drives better reasoning, which reduces uncertainty. This is the missing feedback mechanism that connects UQ measurement to UQ reduction. It also provides a principled way to allocate compute: spend more on uncertain reasoning paths, less on confident ones.

## Experiments
1. Implement SHU on top of DPIMPR + a 5-agent debate system.
2. Define the three rewiring operations (EXPAND, PRUNE, CHALLENGE) with concrete graph modification rules.
3. Train the GNN controller on a dataset of 5000 (DAG state, action, outcome) triples collected from a random rewiring policy.
4. Compare to baselines: (a) static topology + DPIMPR, (b) AgentConductor + DPIMPR, (c) random rewiring + DPIMPR, (d) SHU without the GNN (using hand-crafted threshold rules).
5. Measure: (a) final answer accuracy, (b) calibration (ECE), (c) AUROC, (d) total compute cost (number of agent calls), (e) convergence speed.
6. Ablation: which rewiring operation contributes most? Is CHALLENGE (adversarial insertion) sufficient on its own?

## Risks
- The feedback loop could be unstable: rewiring based on noisy confidence estimates may cause oscillation (add agent, remove agent, add agent...) without convergence.
- Training the GNN controller requires a large dataset of rewiring outcomes, which is expensive to collect from LLM agents.
- The rewiring operations are discrete and non-differentiable, making end-to-end training difficult; the GNN must be trained via reinforcement learning or bandit methods, which are sample-inefficient.
- The system complexity is high: DPIMPR + GNN controller + dynamic topology = many moving parts that are hard to debug and ablate cleanly.
- Reviewers may question whether the GNN controller is doing anything beyond a simple heuristic (e.g., "add an agent when confidence is low"), and ablation experiments may confirm this concern.
