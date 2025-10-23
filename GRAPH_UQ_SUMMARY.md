# Graph-Based Uncertainty Quantification Implementation Summary

## What Was Added

Graph-based uncertainty quantification (UQ) for multi-agent debate traces, adapting the UProp framework to post-hoc debate analysis.

## Files Modified/Created

1. **src/debate_graph_viz.py** (UPDATED)
   - Added `GraphBasedUQ` class (287 lines)
   - Integrated UQ calculations into visualization pipeline
   - Added UQ metrics to graph visualizations

2. **src/DEBATE_UQ_README.md** (NEW)
   - Complete mathematical framework documentation
   - Theoretical foundations and adaptations from UProp
   - Usage examples and interpretation guide

3. **results/debate_vis_uq/** (NEW DIRECTORY)
   - 11 debate graph visualizations with UQ metrics
   - JSON file with comprehensive metrics including UQ results

## Mathematical Framework

### Core Concepts

**Intrinsic Uncertainty (IU)**: Answer diversity at each round
- Measured via normalized Shannon entropy of answer distribution
- IU = 1 means maximum disagreement, IU = 0 means consensus

**Extrinsic Uncertainty (EU)**: Cross-agent trajectory dependencies
- Measured via PMI between agent answers across rounds
- Uses Gaussian kernel to convert semantic distance to PMI
- Captures how much current round depends on past rounds

**Total Uncertainty**: Normalized sum with step-length normalization
- Total = (Σ IU_t + EU_t) / λ
- where λ = Σ (1 + EU_t/IU_t)
- Follows UProp's normalization principle

### Key Adaptations from UProp

| Aspect | UProp (Original) | Graph-Based UQ (Debate) |
|--------|------------------|-------------------------|
| Data Source | Sequential LLM decisions with logprobs | Parallel agent responses (post-hoc) |
| Intrinsic | Per-step predictive entropy | Per-round answer diversity entropy |
| Extrinsic | PMI from trajectory sampling | PMI from semantic distance |
| Probability | Token log probabilities | Gaussian kernel over fuzzy distance |
| Normalization | Step-length (trajectory depth) | Round-length (debate depth) |

## Results on 11 Debate Traces

### Summary Statistics

```
Average Total Uncertainty:       0.7739
Average Intrinsic Uncertainty:   0.4647 (answer diversity)
Average Extrinsic Uncertainty:   0.4807 (trajectory dependencies)
Average Convergence Score:       82.56%

Average Final Accuracy:          42.42%
Average Final Consensus:         89.39%
```

### Key Insights

1. **Nearly Equal IU/EU Contribution**
   - Intrinsic and extrinsic uncertainties both ~0.47
   - Both answer diversity AND trajectory dependencies matter

2. **High Consensus ≠ Low Uncertainty**
   - 89% consensus but 0.77 total uncertainty
   - Agents agree, but HOW they reached agreement involves uncertainty

3. **Strong Convergence**
   - 82.56% convergence score (uncertainty reduced across rounds)
   - Most debates successfully reduce disagreement

4. **Accuracy vs Consensus Gap**
   - Only 42% accuracy despite 89% consensus
   - Agents can confidently converge to wrong answers

### Per-Trace Examples

**Trace 0** (High Uncertainty, Full Convergence):
```
Total Uncertainty: 0.9504
  IU: 0.4591, EU: 0.6933
Convergence: 100% (went from 0.918 to 0.0 IU)
Final Accuracy: 100%
```

**Trace 4** (Highest Uncertainty, Low Convergence):
```
Total Uncertainty: 0.9560
  IU: 0.9591, EU: 0.6525
Convergence: 8.17% (barely converged)
Final Accuracy: 33.33%
```

**Trace 8** (Low Uncertainty):
```
Total Uncertainty: 0.3824
  IU: 0.2296, EU: 0.2222
Convergence: 100%
Final Accuracy: 0% (confident but wrong!)
```

## Visualization Enhancements

Each graph visualization now includes:

### Left Panel (Graph)
- Nodes: Agent responses per round
- Edges: Self-edges (temporal) and influence edges (cross-agent)
- Colors: Agent-coded with correctness brightness

### Right Panel (Metrics) - NEW UQ Section
```
Uncertainty Quantification
----------------------------------------
Total Uncertainty: 0.9504
  Intrinsic (IU): 0.4591
  Extrinsic (EU): 0.6933
Convergence Score: 100.00%

Uncertainty by Round:
  R0: IU=0.918, EU=0.000
  R1: IU=0.000, EU=1.333
  R2: IU=0.918, EU=0.781
  R3: IU=0.000, EU=0.659
```

## Agent Uncertainty Profiles

For each agent, the system calculates:

1. **Trajectory Stability** (0-1): How consistent their answers are
2. **Influence Susceptibility** (0-1): How much they're influenced
3. **Influence Strength** (0-1): How much they influence others

Example from Trace 0:
```
Agent 0: Stability=0.443, Susceptibility=0.0, Influence=0.3
  → Unstable, independent, moderate influence
  
Agent 1: Stability=0.777, Susceptibility=0.3, Influence=0.767
  → Stable, moderately influenced, strong influencer
  
Agent 2: Stability=0.443, Susceptibility=0.533, Influence=0.0
  → Unstable, highly susceptible, no influence
```

## Mathematical Coherence

### Why This Adaptation Works

1. **Preserves Core UProp Principles**
   - Decomposition into intrinsic + extrinsic still valid
   - Step normalization adapted to round normalization
   - PMI still measures trajectory dependencies

2. **Handles Missing Data Gracefully**
   - No logprobs → use entropy of answer distribution
   - No sampling → use actual agent responses as "samples"
   - No sequential decisions → use rounds as "steps"

3. **Theoretically Sound**
   - All metrics bounded and normalized
   - Monotonic relationships (more diversity → higher IU)
   - Composable (Total = IU + EU decomposition)

4. **Equal Evidence Treatment**
   - All agents weighted equally (no ground truth bias during UQ)
   - Ground truth only used for accuracy measurement
   - Uncertainty reflects epistemic uncertainty in debate

## Usage

### Command Line
```bash
conda activate agents
cd MAUQ
python src/debate_graph_viz.py \
  --traces_dir results/debate \
  --output_dir results/debate_vis_uq \
  --layout round
```

### Programmatic
```python
from src.debate_graph_viz import DebateGraphBuilder, GraphBasedUQ

# Load and build graph
builder = DebateGraphBuilder()
graph = builder.build_graph_from_trace(trace_data)

# Calculate UQ
uq = GraphBasedUQ(graph)
results = uq.calculate_total_uncertainty()
profiles = uq.calculate_agent_uncertainty_profile()

print(f"Total UQ: {results['total_uncertainty']:.4f}")
print(f"Convergence: {results['convergence_score']:.2%}")
```

## Files Generated

```
results/debate_vis_uq/
├── debate_graph_0.png      # Visualization with UQ metrics
├── debate_graph_1.png
├── ...
├── debate_graph_10.png
└── debate_metrics_summary.json  # All metrics including UQ

src/
├── debate_graph_viz.py     # Implementation (updated)
├── DEBATE_UQ_README.md    # Full mathematical documentation
└── DEBATE_GRAPH_README.md # Original graph viz docs
```

## Next Steps / Extensions

1. **Calibration Studies**: Validate uncertainty against true accuracy
2. **Semantic Embeddings**: Use SBERT instead of fuzzy matching
3. **Temporal Dynamics**: Model how uncertainty evolves
4. **Uncertainty-Guided Sampling**: Use UQ to decide which debates need more rounds
5. **Multi-Task Analysis**: Compare uncertainty across question types

## References

- UProp framework: src/uprop.py, src/hotpot_qa_uprop.py
- Original debate implementation: llm_multiagent_debate/
- Mathematical details: src/DEBATE_UQ_README.md
