# Causal Uncertainty Decomposition via Do-Calculus on Agent Communication Graphs

## Description
We propose a causal framework for decomposing uncertainty in multi-agent LLM systems by applying Pearl's do-calculus to the agent communication graph. The core mechanism: instead of merely observing that agents agree or disagree, we perform interventional analysis by "clamping" specific agents' outputs and measuring the downstream effect on other agents' confidence and answer distributions. Formally, for agent A_i communicating with agent A_j, we compute the Average Causal Effect ACE(A_i -> A_j) = E[Y_j | do(X_i = x)] - E[Y_j | do(X_i = x')] where Y_j is A_j's answer distribution and X_i is A_i's message. This decomposes total system uncertainty into: (a) causal uncertainty -- uncertainty that propagates through genuine evidential dependencies, and (b) spurious uncertainty -- uncertainty from confounded correlations (e.g., agents agreeing because they share the same training data biases, not because they independently verified each other's reasoning). By constructing a structural causal model over the communication DAG and identifying causal vs. non-causal paths, we can discount the "echo chamber" effect where agents reinforce each other's errors. We expect this to directly address the 89% consensus / 42% accuracy problem by revealing that high consensus often reflects low causal uncertainty but high spurious correlation.

## Gap Filled
Existing multi-agent UQ methods (MATU, CoE, DiscoUQ, DPIMPR) all operate on observational data -- they measure what agents said, not what would change if agents had said something different. No work has applied causal inference to multi-agent LLM uncertainty. This is the fundamental missing piece: the field conflates correlation (agents agree) with causation (agents have independent evidence for the same answer). CLUES decomposes into ambiguity vs. instability, but this is not a causal decomposition -- it doesn't tell you which agent caused which uncertainty.

## Experiments
1. Implement interventional analysis: for each agent in a multi-agent debate, replace its output with a random/adversarial alternative and measure downstream effects on all other agents.
2. Construct the empirical causal graph by testing conditional independencies (PC algorithm or FCI) over agent outputs across many problem instances.
3. Compute ACE for each edge in the communication graph; define "causal confidence" as the fraction of total confidence attributable to causal (non-confounded) paths.
4. Evaluate on the DPIMPR benchmark suite: does causal confidence outperform composite confidence in predicting answer correctness?
5. Specific test: on the "confident but wrong" subset, does causal confidence correctly assign low scores (indicating that consensus was driven by shared bias rather than independent reasoning)?
6. Compare computational cost: interventional analysis requires O(n) additional forward passes per agent per problem, making it expensive. Benchmark wall-clock time vs. DPIMPR.

## Risks
- Interventional analysis is expensive: replacing each agent's output and re-running downstream agents multiplies compute by the number of agents.
- LLM agents are not truly independent -- they share training data, architecture, and often the same base model. The "independence" assumption underlying causal decomposition may be violated so thoroughly that causal analysis is uninformative.
- Defining meaningful interventions is hard: what does it mean to "clamp" an LLM agent's output? Random replacement may not be a valid do-operation in any principled sense.
- The causal graph may be trivially determined by the communication protocol (agents talk in a fixed order), making causal discovery unnecessary and the ACE computation trivially equivalent to influence functions.
