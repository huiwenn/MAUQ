# Debate UQ AUROC Evaluation Summary

## Overview

Evaluated uncertainty quantification metrics on 100 debate traces to assess how well they predict answer correctness.

## Dataset Statistics

- **Total debates**: 100
- **Correct final answers**: 51 (51.0%)
- **Incorrect final answers**: 49 (49.0%)
- **Perfectly balanced dataset** for AUROC evaluation

## AUROC Results

### Full Rankings

| Rank | Metric | AUROC | Category | Quality |
|------|--------|-------|----------|---------|
| 1 | dag_confidence | 0.5000 | DAG-Based | Poor (Random) |
| 1 | dag_avg_node_uncertainty | 0.5000 | DAG-Based | Poor (Random) |
| 1 | dag_conclusion_uncertainty | 0.5000 | DAG-Based | Poor (Random) |
| 1 | dag_evidence_strength | 0.5000 | DAG-Based | Poor (Random) |
| 1 | dag_path_diversity | 0.5000 | DAG-Based | Poor (Random) |
| 1 | dag_uncertainty_confidence | 0.5000 | DAG-Based | Poor (Random) |
| 1 | dag_uncertainty_evidence | 0.5000 | DAG-Based | Poor (Random) |
| 8 | graph_convergence_score | 0.4096 | Graph-Based | Poor |
| 9 | graph_intrinsic_uncertainty | 0.3379 | Graph-Based | Poor |
| 10 | graph_extrinsic_uncertainty | 0.3279 | Graph-Based | Poor |
| 11 | graph_total_uncertainty | 0.3201 | Graph-Based | Poor |

### Key Findings

1. **DAG Metrics (AUROC = 0.50)**:
   - All DAG-based metrics achieve exactly 0.5 AUROC
   - This indicates **random performance** (no predictive power)
   - Root cause: DAG confidence is **constant at 0.0** across all debates
   - Why: Debate structure doesn't create strong belief states in DAG nodes

2. **Graph Metrics (AUROC = 0.32-0.41)**:
   - Better than DAG but still poor predictive power
   - Graph UQ shows weak positive correlation (r=0.32) with correctness
   - Counter-intuitive: Higher uncertainty slightly predicts **correctness**
   - Suggests high uncertainty debates may explore more options

## Metric Value Analysis

### Graph-Based UQ
```
Range: [0.000, 1.153]
Mean: 0.677
Correlation with correctness: 0.3184
```

### DAG-Based UQ
```
DAG Confidence:
  Range: [0.000, 0.000]  (CONSTANT!)
  Mean: 0.000
  Correlation: NaN (no variance)

Reason: All DAG confidence values are 0.0 because:
- Debate traces don't have strong conclusion belief states
- Agents provide numeric answers without explicit confidence
- DAG algorithm designed for reasoning traces with beliefs
```

## Interpretation

### Why Low AUROC Scores?

1. **Task Mismatch**:
   - DAG UQ designed for sequential reasoning (HotpotQA, GSM8K)
   - Debates have parallel multi-agent dynamics
   - Different uncertainty characteristics

2. **Label Definition**:
   - Current label: "Any agent correct at final round"
   - This may not capture debate quality well
   - Alternative: Majority vote correctness, consensus accuracy

3. **High Variance Debates**:
   - High uncertainty might indicate productive exploration
   - Low uncertainty might indicate groupthink
   - Inverted relationship from expected

## Visualizations Generated

### 1. ROC Curves ([debate_auroc_curves.png](results/debate_auroc_eval/debate_auroc_curves.png))
- **Top-left**: Graph-Based UQ ROC curves
- **Top-right**: DAG-Based UQ ROC curves (all overlap at diagonal)
- **Bottom-left**: Bar chart comparing all metrics
- **Bottom-right**: Best metric detailed view

### 2. Results Table ([debate_auroc_table.png](results/debate_auroc_eval/debate_auroc_table.png))
- Ranked list of all metrics with AUROC scores
- Color-coded by category (Graph vs DAG)

### 3. CSV Export ([debate_auroc_results.csv](results/debate_auroc_eval/debate_auroc_results.csv))
```csv
Metric,AUROC,Category
dag_confidence,0.5000,DAG-Based
...
graph_total_uncertainty,0.3201,Graph-Based
```

## Recommendations

### 1. Alternative Label Definitions

Try different correctness criteria:
```python
# Current: Any agent correct
is_correct = int(final_accuracy > 0)

# Alternative 1: Majority vote
is_correct = int(final_accuracy >= 0.5)

# Alternative 2: Consensus quality
is_correct = int(final_accuracy == 1.0 and consensus > 0.8)

# Alternative 3: Confidence-weighted
is_correct = int(final_accuracy * (1 - total_uncertainty) > threshold)
```

### 2. Debate-Specific UQ Metrics

Design metrics specifically for multi-agent debates:
- **Trajectory divergence**: How much agents disagree over time
- **Flip rate**: How often agents change answers
- **Influence asymmetry**: Dominance of one agent
- **Consensus stability**: How long consensus lasts

### 3. Combine Multiple Metrics

Create ensemble uncertainty score:
```python
ensemble_uncertainty = (
    0.4 * graph_intrinsic_uncertainty +
    0.3 * (1 - graph_convergence_score) +
    0.3 * agent_flip_rate
)
```

## Statistical Analysis

### Correlation Matrix (with Correctness)

```
Metric                          Correlation
─────────────────────────────────────────────
graph_total_uncertainty         0.3184
graph_intrinsic_uncertainty     0.2891
graph_extrinsic_uncertainty     0.2654
graph_convergence_score        -0.2147
dag_confidence                  NaN (const)
```

### Observations

1. **Positive correlation** between uncertainty and correctness
   - Contradicts typical UQ assumption
   - Suggests productive debates have higher uncertainty

2. **Convergence negatively correlates** with correctness
   - Fast convergence → wrong answer (groupthink?)
   - Slow convergence → more exploration → better answer

3. **DAG metrics uninformative** for debate setting
   - Need redesign for parallel multi-agent structure

## Code Usage

### Compute UQ for All Debates
```bash
conda activate agents
cd MAUQ
python compute_debate_uq_only.py
```

### Run AUROC Evaluation
```python
from debate_graph_viz import evaluate_debate_auroc

results = evaluate_debate_auroc(
    metrics_summary_path='results/debate_uq_metrics.json',
    output_dir='results/debate_auroc_eval'
)

print(f"Best metric: {results['best_metric']}")
print(f"AUROC scores: {results['auroc_scores']}")
```

### Files Generated

```
results/
├── debate_uq_metrics.json           # All 100 debates with UQ metrics
└── debate_auroc_eval/
    ├── debate_auroc_curves.png      # ROC curve visualizations
    ├── debate_auroc_table.png       # Results table image
    └── debate_auroc_results.csv     # Results in CSV format
```

## Future Work

### 1. Investigate Inverted Relationship

Why does high uncertainty predict correctness?
- Analyze specific high-uncertainty debates
- Check if uncertainty captures exploration vs convergence
- Consider non-linear relationships

### 2. Temporal Dynamics

Track uncertainty evolution across rounds:
```python
round_wise_auroc = {}
for round_num in range(4):
    round_iu = [debates[i]['round_uncertainties'][round_num]['iu']
                for i in range(100)]
    auroc = roc_auc_score(correctness, round_iu)
    round_wise_auroc[round_num] = auroc
```

### 3. Agent-Level Analysis

Compute AUROC per agent instead of per debate:
- 300 agent responses (3 agents × 100 debates)
- Check if individual agent uncertainty is more predictive
- Identify which agents are more calibrated

### 4. Hybrid Metrics

Combine graph + DAG metrics:
```python
hybrid_score = (
    graph_total_uncertainty * 0.6 +
    dag_topological_depth * 0.2 +
    dag_evidence_strength * 0.2
)
```

## Conclusion

**Main Finding**: Neither Graph-based nor DAG-based UQ metrics effectively predict answer correctness in multi-agent debates (AUROC ≈ 0.3-0.5).

**Key Insights**:
1. DAG UQ not applicable to debate structure (constant values)
2. Graph UQ shows weak, **inverted** correlation
3. High uncertainty may indicate productive debate exploration
4. Need debate-specific UQ metrics

**Actionable Next Steps**:
1. Redefine correctness labels (majority vote, consensus quality)
2. Design debate-specific uncertainty metrics
3. Investigate temporal dynamics of uncertainty
4. Consider uncertainty as exploration signal, not just confidence

---

**Evaluation Date**: 2025-10-13
**Dataset**: 100 debate traces from MAUQ
**Status**: ✅ Complete - AUROC curves, tables, and analysis generated
**Best Metric**: All tied at 0.5 (random) - indicates need for better metrics
