# Spectral Uncertainty Signatures: Eigendecomposition of Agent Interaction Graphs for Failure Prediction

## Description
We propose treating the DPIMPR reasoning DAG not as a static structure but as a weighted graph whose Laplacian encodes the spectral geometry of multi-agent reasoning. The key insight is that the eigenvalues of the DAG's Laplacian matrix -- specifically the Fiedler value (second-smallest eigenvalue) and the spectral gap -- characterize how "tightly coupled" the agents' reasoning is. When agents converge confidently on a wrong answer (the 89% consensus / 42% accuracy phenomenon), we hypothesize that the spectral gap collapses: the reasoning graph becomes nearly disconnected in its evidence structure even though the surface-level agreement is high. Concretely, we construct a weighted adjacency matrix where edge weights are semantic similarity between reasoning nodes (from DPIMPR's merging step), compute the normalized Laplacian, and extract the first k eigenvalues as a "spectral uncertainty signature." This signature vector is then used as input to a lightweight classifier (or directly as a calibration feature) to predict whether the consensus answer is correct. The expected outcome is that spectral features capture structural pathologies in collective reasoning -- echo chambers, shallow convergence, reasoning bottlenecks -- that scalar confidence scores miss.

## Gap Filled
DPIMPR already achieves strong AUROC (0.9372), but its confidence score is a single scalar that aggregates over the DAG. This discards rich structural information. No existing work (MATU, CoE, DiscoUQ) analyzes the spectral properties of the reasoning graph itself. Spectral methods are well-understood in graph theory and community detection, but have not been applied to multi-agent LLM reasoning structures. This fills the gap between "we have a DAG" and "we exploit the DAG's global geometry for UQ."

## Experiments
1. Extract reasoning DAGs from DPIMPR on TriviaQA, StrategyQA, and MMLU benchmarks.
2. Compute the normalized Laplacian and extract eigenvalue spectra (first 10-20 eigenvalues).
3. Train a logistic regression on spectral features alone to predict answer correctness; compare AUROC to DPIMPR's composite confidence.
4. Combine spectral features with DPIMPR confidence as a hybrid predictor.
5. Ablation: which eigenvalues carry the most signal? Is it the Fiedler value (graph connectivity) or higher-order eigenvalues (clustering structure)?
6. Analyze the "confident but wrong" cases specifically: do they exhibit distinctive spectral signatures (e.g., near-zero Fiedler value indicating a bottleneck)?

## Risks
- The reasoning DAGs from DPIMPR may be too small (few nodes) for spectral methods to be meaningful -- eigenvalue spectra of small graphs are noisy and degenerate.
- Semantic similarity weights may not accurately reflect the true evidential relationships between reasoning steps, making the Laplacian uninformative.
- The spectral features may be redundant with simpler graph statistics (degree distribution, diameter) that DPIMPR already implicitly captures.
- Computational cost of eigendecomposition is negligible, but the requirement for multiple trajectories to build a rich DAG may limit practical applicability.
