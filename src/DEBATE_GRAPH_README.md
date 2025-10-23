# Debate Graph Visualization

This tool creates graph visualizations and analyzes multi-agent debate traces from the MAUQ debate results.

## Overview

The `debate_graph_viz.py` script processes debate trace JSON files and creates visual graph representations showing:
- **Agent interactions** across multiple debate rounds
- **Information flow** between agents (who influenced whom)
- **Answer convergence** over time
- **Accuracy metrics** per round and per agent

## Graph Structure

### Nodes
- Each node represents an agent's response at a specific round
- **Node ID format**: `A{agent_id}_R{round_num}` (e.g., `A0_R2` = Agent 0, Round 2)
- **Node attributes**:
  - Agent ID
  - Round number
  - Extracted answer
  - Response content (truncated)
  - Correctness (compared to ground truth)
  - Message length

### Edges
Two types of edges:

1. **Self-edges** (solid gray lines):
   - Connect an agent's responses across consecutive rounds
   - Show temporal progression of single agent's thinking

2. **Influence edges** (dashed blue lines):
   - Connect agents across rounds
   - Show cross-agent information flow
   - Edge weight based on answer agreement:
     - 1.0: Strong agreement (same answers)
     - 0.5: Neutral (default)
     - 0.3: Disagreement (different answers)

### Visual Encoding

- **Node colors**: Different color per agent (consistent across rounds)
  - Bright color = correct answer
  - Gray = incorrect answer
- **Node size**: Increases slightly with round number
- **Edge thickness**: Proportional to influence weight
- **Edge style**: Solid for self, dashed for influence

## Usage

### Basic Usage

```bash
# Activate the agents conda environment
conda activate agents

# Run with default settings
python src/debate_graph_viz.py

# Custom directories
python src/debate_graph_viz.py \
  --traces_dir results/debate \
  --output_dir results/debate_vis
```

### Layout Options

```bash
# Round-based layout (default) - time flows left to right
python src/debate_graph_viz.py --layout round

# Circular layout - agents arranged in expanding circles per round
python src/debate_graph_viz.py --layout circular
```

## Output Files

The script generates:

1. **Visualization PNGs** (`debate_graph_{id}.png`):
   - Left panel: Graph visualization with color-coded nodes and edges
   - Right panel: Metrics summary including accuracy, consensus, influence scores
   - Legend showing agent colors and edge types

2. **Metrics summary JSON** (`debate_metrics_summary.json`):
   - Aggregated metrics for all debates
   - Per-round accuracy and consensus
   - Influence scores per agent
   - Graph structure statistics

## Key Metrics

### Accuracy by Round
Percentage of agents with correct answers at each round:
```
Round 0: 0.00%
Round 1: 0.00%
Round 2: 66.67%
Round 3: 100.00%
```

### Consensus by Round
Agreement level among agents (1.0 = all agents agree):
```
Round 0: 0.50 (2 different answers)
Round 1: 1.00 (all agents converged)
Round 2: 0.50 (agents split again)
Round 3: 1.00 (final consensus)
```

### Influence Scores
Total influence weight each agent contributed to others:
```
Agent 0: 1.80
Agent 1: 2.30  (most influential)
Agent 2: 1.50
```

## Implementation Details

### Class: `DebateGraphBuilder`

Main methods:
- `build_graph_from_trace(trace_data)`: Constructs NetworkX DiGraph from trace JSON
- `extract_answer(response)`: Extracts numeric/text answer from agent response
- `calculate_graph_metrics()`: Computes accuracy, consensus, influence metrics

### Class: `DebateGraphVisualizer`

Main methods:
- `create_round_based_layout(graph)`: Time-based horizontal layout
- `create_circular_per_round_layout(graph)`: Radial layout with expanding circles
- `visualize_debate_graph(graph, metrics, ...)`: Renders graph with matplotlib/networkx

### Main Function: `process_debate_traces()`

Orchestrates the full pipeline:
1. Loads all trace JSON files from directory
2. Builds graph for each debate
3. Calculates metrics
4. Generates visualizations
5. Saves aggregate metrics summary

## Example Output

For a typical 3-agent, 4-round debate:

```
Found 11 trace files

Processing traces_0.json...
  Nodes: 12, Edges: 18
  Final accuracy: 100.00%

✓ Processed 11 debates
✓ Visualizations saved to results/debate_vis
✓ Metrics summary saved to results/debate_vis/debate_metrics_summary.json

Summary Statistics:
Average final accuracy: 42.42%
Average final consensus: 89.39%
```

## Integration with MAUQ

This tool follows the implementation pattern in `hotpot_qa_uprop.py`:
- Uses similar data structures and conventions
- Compatible with MAUQ's debate trace format
- Follows project structure and naming conventions
- Designed for easy extension to other debate formats

## Dependencies

Required packages (already in MAUQ requirements.txt):
- `matplotlib>=3.4.0`
- `networkx>=2.5`
- `numpy>=1.20.0`
- Standard library: `json`, `os`, `re`, `collections`, `typing`

## Future Enhancements

Potential improvements:
- Interactive visualizations with plotly
- Animated debate progression over rounds
- Semantic clustering of similar answers
- Network centrality metrics (betweenness, PageRank)
- Export to graph formats (GraphML, DOT)
- Comparative analysis across multiple debates
