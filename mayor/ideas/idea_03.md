# Conformal DAG Prediction Sets: Distribution-Free Coverage Guarantees for Multi-Agent Reasoning

## Description
We propose extending conformal prediction from single-model UQ (ConU) to DAG-structured multi-agent reasoning by defining a novel nonconformity score that respects the DAG topology. The key challenge ConU solved for single models -- exchangeability of calibration examples -- breaks down in multi-agent DAGs because the output of downstream agents depends on upstream agents, introducing complex dependencies. Our solution: we define a "DAG-conformal" score that factorizes along the topological ordering of the DAG. For each node v in the DAG, we compute a local nonconformity score s_v using the node's input-output pair, then aggregate scores along all source-to-sink paths, taking the maximum path score as the global nonconformity measure. This "worst-path" aggregation is conservative but preserves the marginal coverage guarantee: P(Y in C(X)) >= 1 - alpha. The factorization along topological order restores a form of conditional exchangeability (each node's residual is exchangeable given its parents' outputs), enabling valid conformal inference. We integrate this with DPIMPR by using the DAG's merged reasoning structure as the backbone and the composite confidence as a base score that we conformalize. Expected outcome: prediction sets with guaranteed coverage (e.g., 90%) that are tighter than naive Bonferroni corrections over agents.

## Gap Filled
ConU provides conformal guarantees for single models but does not handle multi-agent DAGs. DPIMPR provides strong empirical UQ but no formal coverage guarantees. No existing work provides distribution-free coverage guarantees for multi-agent reasoning systems. The gap is both theoretical (how to define valid conformal scores on DAGs) and practical (how to produce useful prediction sets for multi-step reasoning). MATU decomposes uncertainty but offers no guarantees; CoE quantifies entropy but without coverage. This would be the first work to bridge conformal prediction and multi-agent DAG reasoning.

## Experiments
1. Implement DAG-conformal prediction on top of DPIMPR: use the merged reasoning DAG as the structure, compute local nonconformity scores at each node, aggregate via worst-path.
2. Calibration experiment: split data into calibration (70%) and test (30%) sets, verify that marginal coverage holds at alpha = 0.05, 0.10, 0.20.
3. Compare prediction set sizes to: (a) naive conformal on the final answer only (ignoring DAG structure), (b) Bonferroni-corrected conformal over individual agents, (c) no conformal correction (just DPIMPR thresholding).
4. Conditional coverage analysis: does coverage hold across different difficulty levels, question types, and agent configurations?
5. Adaptive prediction sets: use DPIMPR confidence to modulate the local nonconformity scores, producing tighter sets for "easy" questions and wider sets for "hard" ones.
6. Benchmarks: TriviaQA, StrategyQA, MMLU, and a multi-hop reasoning dataset (HotpotQA) where the DAG structure is most meaningful.

## Risks
- The conditional exchangeability assumption (node residuals are exchangeable given parents) may not hold in practice, since LLM agents have complex, non-stationary behavior.
- Worst-path aggregation may be too conservative, producing prediction sets that are so large they are useless (e.g., "the answer is one of 50 options").
- The factorization along topological order assumes a clean DAG; in practice, DPIMPR's merged DAGs may have cycles or ambiguous ordering.
- Conformal prediction is inherently a post-hoc calibration method -- it cannot improve the underlying model's accuracy, only its calibration. If DPIMPR is already well-calibrated, the marginal benefit is small.
- Reviewers may view this as a straightforward extension of ConU rather than a conceptual advance, since the main novelty is the aggregation scheme.
