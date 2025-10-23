# MAUQ: Multi-Agent Uncertainty Quantification

A research implementation exploring **three distinct variants** of uncertainty quantification for multi-step reasoning and multi-agent systems.

## Overview

MAUQ implements uncertainty quantification algorithms for:
- **Sequential multi-step reasoning** (HotpotQA, GSM8K, MATH)
- **Multi-agent debate systems**
- **DAG-based reasoning graph analysis**

> **⚠️ Important**: This project contains **three different variants** of uncertainty quantification, each with distinct mathematical formulations. See [UQ_VARIANTS.md](UQ_VARIANTS.md) for detailed comparison.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys
export OPENAI_API_KEY="your-key-here"

# Run UProp on HotpotQA
python src/hotpot_qa_uprop.py

# Run UProp on GSM8K
python src/gsm8k_uprop.py

# Run UProp on MATH dataset
python src/math_uprop.py

# Visualize DAG from reasoning traces
python src/dag.py
```

## Uncertainty Quantification Variants

### 1. UProp: Uncertainty Propagation
- **Implementation**: `src/uprop.py`
- **Applications**: `src/hotpot_qa_uprop.py`, `src/gsm8k_uprop.py`, `src/math_uprop.py`
- **Use Case**: Sequential reasoning with log probabilities
- **Output**: Trajectory-level uncertainty scores

### 2. Graph-Based UQ
- **Implementation**: `src/debate_graph_viz.py`
- **Use Case**: Multi-agent debate analysis
- **Output**: Round-level uncertainty + convergence metrics

### 3. DAG-Based UQ (DPIMPR)
- **Implementation**: `src/dag.py`
- **Use Case**: Universal (sequential + debate)
- **Output**: Confidence scores + graph structure metrics

**📖 Read [UQ_VARIANTS.md](UQ_VARIANTS.md) for detailed comparison and usage guidelines.**

## Project Structure

```
MAUQ/
├── src/
│   ├── uprop.py                  # Core UProp algorithm
│   ├── hotpot_qa_uprop.py        # HotpotQA experiments
│   ├── gsm8k_uprop.py            # GSM8K experiments
│   ├── math_uprop.py             # MATH dataset experiments
│   ├── dag.py                    # DPIMPR algorithm + DAG UQ
│   ├── debate_graph_viz.py       # Debate graph UQ + visualization
│   └── agents.py                 # ReAct agent implementations
├── results/                      # Experiment results
├── playground.ipynb              # Interactive experiments
├── UQ_VARIANTS.md               # ⭐ Variant comparison guide
└── requirements.txt              # Dependencies
```

## Core Algorithms

### UProp: Uncertainty = IU + EU

**Intrinsic Uncertainty** (per-step diversity):
```
IU_t = (1/N) Σ_n (-log p(y_t^(n)) / |y_t^(n)|)
```

**Extrinsic Uncertainty** (cross-step dependencies):
```
EU_t = Σ_{i<t} PMI(y_t; y_i)
PMI = -log(Σ_n K_N(d(y_i^(n), y_i)))
```

**Step Normalization**:
```
U_total = (Σ_t (IU_t + EU_t)) / (Σ_t σ_t)
where σ_t = 1 + EU_t/IU_t
```

### DPIMPR: DAG Construction

1. **Semantic Merging**: Merge similar reasoning nodes while preserving acyclicity
2. **Evidence Accumulation**: Reduce uncertainty via `u -= 0.1*log(evidence + 1)`
3. **Confidence Computation**: Weighted by topology, evidence, and base uncertainty

## Datasets & Benchmarks

- **HotpotQA**: Multi-hop question answering with Wikipedia search
- **GSM8K**: Grade school math word problems
- **MATH**: Competition mathematics (Hendrycks benchmark)
- **Debate**: Multi-agent debate traces

## Key Results

Results are stored in `results/` directory with AUROC evaluations comparing:
- UProp uncertainty
- Semantic Entropy baseline
- Predictive Entropy baseline
- DAG confidence metrics

See `AUROC_EVALUATION_SUMMARY.md` for detailed results.

## Development

### Running Experiments

```bash
# HotpotQA with UProp
python src/hotpot_qa_uprop.py

# GSM8K with UProp
python src/gsm8k_uprop.py

# MATH dataset with UProp
python src/math_uprop.py

# DAG visualization
python src/dag.py

# Debate graph analysis
python src/debate_graph_viz.py --traces_dir results/debate --output_dir results/debate_vis
```

### Interactive Development

```bash
jupyter notebook playground.ipynb
```

### Evaluation

```python
from dag import evaluate_dag_uncertainty_auroc

# Evaluate DAG uncertainty on HotpotQA
results = evaluate_dag_uncertainty_auroc(
    trace_files=glob.glob("results/hotpotqa_*/paths_*.json"),
    similarity_threshold=0.7,
    k_trajectories=3
)

print(f"Best AUROC: {results['auroc_scores']}")
```

## Important Notes

### Mathematical Inconsistencies

The three variants use **different uncertainty formulations**:

| Variant | IU Formula | PMI Kernel | Output |
|---------|------------|------------|--------|
| UProp | `-log(p)/length` | `exp(-d²/2)/N` | Uncertainty ↑ |
| Graph-Based | `H(answers)/log(n)` | `exp(-d²/2σ²)` | Uncertainty ↑ |
| DAG-Based | `abs(log(p))/10` | N/A | Confidence ↑ |

These are **not comparable** without appropriate normalization. See [UQ_VARIANTS.md](UQ_VARIANTS.md) for details.

### For AUROC Evaluation

When comparing variants:
1. Use **only one variant per evaluation**
2. Invert DAG confidence: `uncertainty = 1 - confidence`
3. Document which variant was used
4. Don't average AUROC across variants

## Citation

If you use this code, please cite:

```bibtex
@software{mauq2025,
  title={MAUQ: Multi-Agent Uncertainty Quantification},
  author={[Your Name]},
  year={2025},
  note={Three variants: UProp, Graph-Based UQ, and DPIMPR}
}
```

## References

- **UProp Framework**: Duan et al. (2025) - Uncertainty Propagation in Multi-Step LLM Reasoning
- **Semantic Entropy**: Kuhn et al. (2023) - Semantic Uncertainty Quantification
- **DPIMPR**: Internal development - DAG-Preserving Incremental Multi-Path Reasoning

## License

[Add your license here]

## Documentation

- [UQ_VARIANTS.md](UQ_VARIANTS.md) - Detailed comparison of uncertainty quantification variants
- [CLAUDE.md](CLAUDE.md) - Development guide for Claude Code
- [AUROC_EVALUATION_SUMMARY.md](AUROC_EVALUATION_SUMMARY.md) - Evaluation results
- [GRAPH_UQ_SUMMARY.md](GRAPH_UQ_SUMMARY.md) - Graph-based UQ details
- [DAG_UQ_INTEGRATION_SUMMARY.md](DAG_UQ_INTEGRATION_SUMMARY.md) - DAG integration guide
