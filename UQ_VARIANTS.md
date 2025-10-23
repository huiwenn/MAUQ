# Uncertainty Quantification Variants in MAUQ

This document clarifies that MAUQ implements **two distinct variants** of uncertainty quantification, each designed for different multi-agent reasoning scenarios.

---

## Overview

The MAUQ project contains two fundamentally different approaches to uncertainty quantification:

1. **UProp**: Trajectory-based uncertainty propagation for sequential LLM reasoning
2. **Graph-Based UQ**: Debate-centric uncertainty quantification for multi-agent consensus

**Important**: These are **not** different implementations of the same algorithm. They are **different variants** with distinct mathematical formulations, designed for different use cases.

---

## Variant 1: UProp (Uncertainty Propagation)

**File**: `src/uprop.py`
**Use Cases**: HotpotQA, GSM8K, MATH dataset - sequential multi-step reasoning tasks
**Data Structure**: Trajectory-Dependent Decision Process (TDP)

### Core Algorithm

UProp quantifies uncertainty in sequential decision-making by sampling multiple trajectories and analyzing decision divergence.

#### Intrinsic Uncertainty (IU)
```
IU_t = (1/N) Σ_n (-log p(y_t^(n)) / |y_t^(n)|)
```
- Uses **length-normalized negative log-probability**
- Measures uncertainty at each individual decision step
- Accounts for varying decision complexity via length normalization

#### Extrinsic Uncertainty (EU) via PMI
```
PMI(y_t; y_i) = -log(Σ_n K_N(d(y_i^(n), y_i)))
K_N(d) = (1/√(2π)) * exp(-d²/2) / N
```
- Uses **sample-count normalized Gaussian kernel**
- Measures cross-step dependencies in reasoning paths
- Kernel sharpness controlled by number of samples N

#### Step Normalization
```
σ_t = 1 + EU_t/IU_t
λ_z = Σ_t σ_t
U_total = (1/λ_z) Σ_t (IU_t + EU_t)
```
- Normalizes by reasoning complexity at each step
- Accounts for varying trajectory lengths

### Key Characteristics
- **Input**: Single-agent, multiple sampled trajectories (Z TDPs)
- **Granularity**: Step-level uncertainty decomposition
- **Distance Metric**: Fuzzy string matching on actions
- **Output**: Normalized uncertainty score per trajectory

---

## Variant 2: Graph-Based UQ (Multi-Agent Debate)

**File**: `src/debate_graph_viz.py`
**Use Cases**: Multi-agent debate, consensus-building scenarios
**Data Structure**: Debate graph G = (V, E)

### Core Algorithm

Graph-Based UQ quantifies uncertainty through agent consensus evolution across debate rounds.

#### Intrinsic Uncertainty (IU)
```
IU(r) = H(p_r) / log|A_r|
H(p_r) = -Σ_ans p(ans) log p(ans)
```
- Uses **normalized discrete entropy** over answer distribution
- Measures answer diversity among agents at each round
- High entropy = high disagreement = high uncertainty

#### Extrinsic Uncertainty (EU) via PMI
```
PMI(y_r^a; y_i^a') = -log(exp(-d²(y_r^a, y_i^a') / 2σ²) + ε)
EU(r) = (1/N²(r-1)) Σ_a Σ_i<r Σ_a' PMI(y_r^a; y_i^a')
```
- Uses **fixed-bandwidth Gaussian kernel** (σ=0.5)
- Measures cross-agent, cross-round dependencies
- Averages over all agent pairs and round combinations

#### Step Normalization
```
σ_r = 1 + EU_r/IU_r
U_total = (Σ_r (IU_r + EU_r)) / (Σ_r σ_r)
```
- Same normalization strategy as UProp
- Applied to debate rounds instead of decision steps

#### Convergence Score
```
convergence = 1 - (IU_final / IU_initial)
```
- **Unique to Graph-Based UQ**
- Measures how much uncertainty decreased through debate

### Key Characteristics
- **Input**: Multi-agent, multiple debate rounds
- **Granularity**: Round-level uncertainty across all agents
- **Distance Metric**: Fuzzy string matching on answers
- **Output**: Uncertainty + convergence score

---

## Variant 3: DAG-Based UQ (DPIMPR)

**File**: `src/dag.py`
**Use Cases**: Both sequential reasoning (HotpotQA, GSM8K) and debate
**Data Structure**: Directed Acyclic Graph with merged reasoning nodes

### Core Algorithm

DPIMPR constructs a DAG by semantically merging similar reasoning steps across trajectories, then computes uncertainty from graph structure.

#### Node Uncertainty
```
u_t = |log p_t| / 10.0
u_merged = (u_old * w_old + u_new * w_new) / (w_old + w_new) - 0.1*log(k+1)
```
- Uses **scaled log-probability** (not length-normalized)
- Evidence accumulation **reduces uncertainty** via log(k+1) term
- Unique convergence benefit from semantic merging

#### Semantic Merging
```
sim(s_i, s_j) = max(SequenceMatcher(s_i, s_j), Jaccard(words_i, words_j) * 0.8)
merge if: sim > θ_sim AND |level_i - level_j| ≤ δ_topo
```
- Merges semantically similar nodes while preserving DAG property
- Accumulates evidence at merged nodes

#### DAG Confidence
```
confidence = (Σ_conclusion p(conclusion) * w_topo * w_evidence * w_uncertainty) / Σ_weights
w_topo = (level + 1) / max_level
w_evidence = evidence_count
w_uncertainty = 1 / (uncertainty + 0.1)
```
- **Inverted from uncertainty** (higher confidence = lower uncertainty)
- Weighted by topological depth, evidence, and base uncertainty

### Key Characteristics
- **Input**: Either format (sequential trajectories or debate graphs)
- **Granularity**: Node-level with graph structure
- **Distance Metric**: Sequence matching + word overlap
- **Output**: Confidence score + graph metrics (evidence strength, path diversity, topological depth)

### Unique Features
- Evidence accumulation across trajectories
- Topological constraint enforcement
- Multiple uncertainty metrics from graph structure

---

## Comparison Table

| Feature | UProp | Graph-Based UQ | DAG-Based UQ |
|---------|-------|----------------|--------------|
| **Primary Use Case** | Sequential reasoning | Multi-agent debate | Universal (both) |
| **IU Formula** | `-log(p)/length` | `H(answers)/log(n)` | `abs(log(p))/10` |
| **EU Method** | PMI with K_N kernel | PMI with fixed σ | Graph structure metrics |
| **Normalization** | λ_z = Σ(1 + EU/IU) | λ_z = Σ(1 + EU/IU) | Evidence-weighted |
| **Sample Aggregation** | N samples per step | All agents per round | Semantic node merging |
| **Kernel** | `exp(-d²/2)/N/√(2π)` | `exp(-d²/2σ²)` | N/A |
| **Output Type** | Uncertainty score | Uncertainty + convergence | Confidence + graph metrics |
| **Evidence Accumulation** | No | No | Yes (log benefit) |
| **Trajectory Length** | Variable | Fixed (R rounds) | Variable (DAG depth) |

---

## When to Use Each Variant

### Use UProp when:
- ✅ Working with sequential reasoning tasks (HotpotQA, GSM8K, MATH)
- ✅ You have log probabilities from LLM API
- ✅ You want step-by-step uncertainty decomposition
- ✅ Trajectories have varying lengths
- ✅ You need **trajectory-level** uncertainty for ranking/filtering

### Use Graph-Based UQ when:
- ✅ Working with multi-agent debate scenarios
- ✅ You want to measure consensus convergence
- ✅ Log probabilities are unavailable (using answer distributions)
- ✅ You need **round-level** analysis of debate dynamics
- ✅ You want both uncertainty and convergence metrics

### Use DAG-Based UQ when:
- ✅ You want to leverage semantic similarity across trajectories
- ✅ You need confidence scores (not just uncertainty)
- ✅ Evidence accumulation is important
- ✅ You want **graph structure insights** (convergence points, path diversity)
- ✅ You need to handle both debate and sequential reasoning uniformly

---

## Hybridization in the Codebase

The `DebateDAGUncertainty` class (in `debate_graph_viz.py`) creates a **hybrid approach**:

```python
class DebateDAGUncertainty:
    """Integrates DAG-based uncertainty into debate analysis"""

    def convert_debate_to_trajectories(self):
        # Converts debate graph → TDP format
        # Each agent trajectory becomes a TDP

    def compute_dag_uncertainty(self):
        # Calls DPIMPR algorithm
        # Returns DAG confidence metrics
```

This allows debate analysis to benefit from:
- Semantic merging of identical answers across agents
- Evidence accumulation from agreement
- Graph-based confidence scores

However, note that this is **DAG-Based UQ applied to debate**, not Graph-Based UQ. The uncertainty formulations are different.

---

## Mathematical Inconsistencies to Be Aware Of

### 1. IU Formulation Mismatch
- UProp: Length-normalized log-prob → continuous per token
- Graph: Discrete entropy → depends on answer clustering
- DAG: Raw scaled log-prob → no length adjustment

These produce **different scales** and are not directly comparable.

### 2. PMI Kernel Normalization
- UProp kernel: Normalized by N (number of samples)
- Graph kernel: Fixed σ=0.5 bandwidth
- Results differ when N varies

### 3. Uncertainty vs Confidence
- UProp & Graph: Output uncertainty (higher = worse)
- DAG: Outputs confidence (higher = better)
- Need inversion for AUROC comparison: `uncertainty = 1 - confidence`

### 4. Step vs Round Semantics
- UProp steps: Variable length trajectories
- Graph rounds: Fixed debate rounds
- Not directly comparable without alignment

---

## Recommendations for Future Work

### For Consistency
1. **Align IU formulations**: Choose canonical definition (recommend length-normalized log-prob)
2. **Standardize PMI kernel**: Use consistent normalization across variants
3. **Unified output**: Always output both uncertainty and confidence
4. **Document scales**: Clearly specify uncertainty ranges for each variant

### For Evaluation
1. **Separate AUROC evaluations** for each variant on appropriate tasks
2. **Document which variant** was used for each result
3. **Compare only on shared tasks** (e.g., both on HotpotQA)
4. **Report confidence intervals** to assess statistical significance

### For Extensions
1. **Create unified interface**: Abstract base class for all UQ variants
2. **Hybrid approach**: Combine UProp step-level + DAG evidence accumulation
3. **Adaptive selection**: Automatically choose variant based on task type
4. **Meta-uncertainty**: Quantify disagreement between variants

---

## Code Examples

### Using UProp (Sequential Reasoning)
```python
from uprop import UProp, MATHUProp

# For MATH dataset
uprop = MATHUProp(api_key=API_KEY, model="gpt-4", Z=3, N=3)
results = uprop.estimate_uncertainty(prompt, context, target_answer)

print(f"Total Uncertainty: {results['total_uncertainty']:.4f}")
print(f"Target Uncertainty: {results['target_uncertainty']:.4f}")
```

### Using Graph-Based UQ (Debate)
```python
from debate_graph_viz import DebateGraphBuilder, GraphBasedUQ

# Build debate graph
builder = DebateGraphBuilder()
graph = builder.build_graph_from_trace(trace_data)

# Calculate uncertainty
uq = GraphBasedUQ(graph)
results = uq.calculate_total_uncertainty()

print(f"Intrinsic Uncertainty: {results['intrinsic_uncertainty']:.4f}")
print(f"Extrinsic Uncertainty: {results['extrinsic_uncertainty']:.4f}")
print(f"Convergence Score: {results['convergence_score']:.2%}")
```

### Using DAG-Based UQ (Universal)
```python
from dag import DPIMPR

# Initialize DPIMPR
dpimpr = DPIMPR(similarity_threshold=0.7, question=question)

# Add trajectories (auto-detects format)
results = dpimpr.add_reasoning_traces(trace_data)

# Extract uncertainty metrics
metrics = dpimpr.extract_dag_uncertainty_metrics()

print(f"DAG Confidence: {metrics['dag_confidence']:.4f}")
print(f"Evidence Strength: {metrics['evidence_strength']:.2f}")
print(f"Path Diversity: {metrics['path_diversity']:.0f}")
```

---

## References

- **UProp**: Duan et al. (2025) - Uncertainty Propagation in Multi-Step LLM Reasoning
- **Semantic Entropy**: Kuhn et al. (2023) - Semantic Uncertainty via Clustering
- **DPIMPR**: Internal development - DAG-Preserving Incremental Multi-Path Reasoning
- **Graph-Based Debate UQ**: Internal development - Adapted UProp framework for debates

---

## Version History

- **v1.0** (2025-01-23): Initial documentation of variant differences
- Documented by: Claude Code Analysis
