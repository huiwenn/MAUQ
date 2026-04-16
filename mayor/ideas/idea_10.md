# Uncertainty-Aware DAG Compilation: Optimizing Multi-Agent Workflows via Uncertainty-Guided Query Planning

## Description
We propose a "DAG compiler" that takes a multi-agent reasoning workflow (expressed as a DAG of agent calls) and produces an optimized execution plan that minimizes total uncertainty subject to a compute budget, analogous to how a database query optimizer minimizes I/O cost subject to resource constraints. The key mechanism draws on Halo's insight (3.6x speedup from database-style DAG optimization) but extends it from latency optimization to uncertainty optimization. The compiler maintains a cost model that estimates, for each edge in the DAG, the expected uncertainty contribution (from DPIMPR) and the expected compute cost. It then applies three optimization passes: (1) Redundancy elimination -- identify DAG subgraphs where multiple agents perform equivalent reasoning (detected via semantic similarity of intermediate outputs) and prune redundant paths, replacing them with a single representative and a confidence penalty. (2) Uncertainty-guided parallelism -- when the DAG has branching points, prioritize executing the branch whose expected uncertainty reduction per compute unit is highest (a multi-armed bandit formulation). (3) Early termination -- when a DAG subpath achieves confidence above a threshold, skip remaining nodes on that path and reallocate compute to uncertain paths. The compiler outputs an execution plan that is a partial ordering of agent calls with conditional branches, essentially a "query plan for reasoning."

## Gap Filled
Halo optimizes agent DAGs for latency but ignores uncertainty. DPIMPR computes uncertainty post-hoc but does not use it to guide execution. No existing system uses uncertainty estimates to dynamically allocate compute within a multi-agent DAG. This is the natural intersection of the "agent workflows as databases" paradigm (Halo) and the "UQ for multi-agent systems" paradigm (MAUQ). The Self-Healing Router's finding (93% of control-plane calls replaceable) suggests that most of the computation in agent workflows is wasted; the DAG compiler uses uncertainty to identify and eliminate this waste.

## Experiments
1. Implement the DAG compiler on top of DPIMPR, using its confidence estimates as the cost model.
2. Define the three optimization passes (redundancy elimination, uncertainty-guided parallelism, early termination) with concrete algorithms.
3. Benchmark on multi-agent reasoning tasks of varying complexity: (a) single-hop QA (TriviaQA), (b) multi-hop QA (HotpotQA), (c) mathematical reasoning (GSM8K), (d) code generation (HumanEval+).
4. Measure: (a) accuracy at iso-compute (same total LLM calls), (b) compute savings at iso-accuracy (same final accuracy), (c) calibration (ECE) under the optimized execution plan, (d) comparison to Halo's latency optimizer and to random pruning.
5. Pareto frontier: plot accuracy vs. compute cost for the compiled vs. uncompiled DAG across different compute budgets.
6. Ablation: which optimization pass contributes most? Is early termination sufficient on its own?

## Risks
- The cost model (estimated uncertainty contribution per edge) must be accurate for the compiler to make good decisions; if DPIMPR's confidence estimates are noisy at intermediate nodes, the compiler may prune valuable reasoning paths.
- The multi-armed bandit formulation for branch prioritization requires an exploration-exploitation tradeoff that may be hard to tune, and wrong exploration decisions are irreversible (you cannot un-prune a path).
- Redundancy elimination via semantic similarity may be too aggressive (removing genuinely diverse reasoning that happens to use similar words) or too conservative (failing to detect redundancy expressed in different terms).
- The comparison to Halo may be unfair since Halo optimizes for latency and this method optimizes for uncertainty; they solve different problems and a "hybrid" that optimizes both may be the real contribution, but that adds complexity.
- The paper requires both a system contribution (the compiler) and an empirical contribution (it actually helps), and papers that try to do both often do neither convincingly.
