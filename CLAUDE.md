# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MAUQ (Multi-Agent Uncertainty Quantification) is a research implementation for uncertainty quantification in multi-step LLM decision making. The project implements three key algorithms:

1. **UProp**: Uncertainty Propagation for LLMs in Multi-Step Agentic Decision-Making
2. **SAUP**: Self-Aware Uncertainty Propagation (placeholder implementation)
3. **DPIMPR**: DAG-Preserving Incremental Multi-Path Reasoning Algorithm

## Core Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run HotpotQA experiments with UProp
python src/hotpot_qa_uprop.py

# Run HotpotQA experiments with SAUP
python src/hotpot_qa_saup.py

# Run DAG visualization examples
python src/dag.py

# Interactive development
jupyter notebook playground.ipynb
```

## Code Architecture

### Core Modules

- **`src/uprop.py`**: UProp algorithm implementation with TDP (Trajectory-Dependent Decision Process) sampling and uncertainty calculation
- **`src/saup.py`**: SAUP algorithm (minimal implementation - needs development)
- **`src/agents.py`**: ReAct agent implementation with uncertainty-aware reasoning
- **`src/dag.py`**: DPIMPR algorithm for building reasoning DAGs from multiple trajectories
- **`src/hotpot_qa_uprop.py`**: HotpotQA evaluation using UProp
- **`src/hotpot_qa_saup.py`**: HotpotQA evaluation using SAUP

### Key Classes

**UProp Algorithm (`src/uprop.py`)**:
- `UProp`: Main algorithm class implementing trajectory sampling and uncertainty calculation
- `TDP`: Trajectory-Dependent Decision Process data structure
- `Decision`: Individual decision representation with log probabilities
- `SemanticEntropy`: Baseline comparison implementation

**DPIMPR Algorithm (`src/dag.py`)**:
- `DPIMPR`: DAG construction algorithm with semantic node merging
- `DAGReasoningNode`: Enhanced reasoning nodes with uncertainty and belief information
- Visualization capabilities with Sugiyama algorithm for layered graph drawing

**Agent Framework (`src/agents.py`)**:
- `ReActAgent`: Multi-hop QA agent with Wikipedia search capabilities
- Log probability extraction for uncertainty estimation
- Integration with HuggingFace transformers

## Algorithm Details

### UProp Uncertainty Quantification

The UProp algorithm works by:
1. Sampling Z Trajectory-Dependent Decision Processes (TDPs)
2. For each step, sampling N decisions and randomly selecting one for the trajectory
3. Calculating intrinsic uncertainty using predictive entropy
4. Computing extrinsic uncertainty via Pointwise Mutual Information (PMI)
5. Aggregating uncertainties with step length normalization

Key parameters:
- `Z`: Number of TDP samples (default: 3)
- `N`: Number of per-step samples (default: 3)
- `temperature`: Sampling temperature (default: 0.8)
- `distance_metric`: For decision distance calculation (default: 'fuzzy')

### DPIMPR DAG Construction

The DPIMPR algorithm constructs reasoning DAGs by:
1. Incrementally adding trajectories while preserving acyclic structure
2. Semantically merging similar nodes using similarity thresholds
3. Accumulating evidence across multiple trajectories
4. Maintaining topological constraints to prevent cycles

Key parameters:
- `similarity_threshold`: For node merging (default: 0.7)
- `topological_distance`: Maximum distance for safe merging (default: 2)
- `belief_propagation_iterations`: For belief updates (default: 5)

## API Integration

The project supports multiple LLM APIs:
- **OpenAI**: via `openai` client
- **OpenRouter**: For accessing various models including Llama, Claude, etc.
- **Local models**: via HuggingFace transformers

API configuration in `playground.ipynb` shows examples for:
- GPT-4, GPT-3.5
- Claude models via OpenRouter
- Llama models via OpenRouter
- Log probability extraction

## Results and Evaluation

### HotpotQA Benchmark

Results are stored in `results/` directory:
- AUROC scores comparing UProp vs baselines (Semantic Entropy, Predictive Entropy)
- Success rates and uncertainty metrics
- Visualization outputs in `results/vis/`

### Evaluation Metrics

- **AUROC**: Area Under ROC curve for uncertainty calibration
- **Success Rate**: Percentage of correct answers
- **Uncertainty Correlation**: How well uncertainty correlates with correctness

Example results structure:
```json
{
  "questions": [...],
  "ground_truth": [...],
  "predictions": [...],
  "correct": [...],
  "uprop_uncertainty": [...],
  "se_uncertainty": [...],
  "pe_uncertainty": [...]
}
```

## Development Patterns

### Adding New Algorithms

1. Create new module in `src/`
2. Implement uncertainty quantification method
3. Add evaluation script following `hotpot_qa_uprop.py` pattern
4. Update requirements if new dependencies needed

### Extending Agents

- Follow `ReActAgent` pattern in `src/agents.py`
- Implement `solve_question()` method returning answer and trajectory
- Extract log probabilities for uncertainty calculation
- Use consistent trajectory format with step-by-step reasoning

### Visualization

- Use `DPIMPR.visualize_dag()` for reasoning DAG visualization
- Results include node uncertainty, evidence counts, and belief states
- Customizable layout and color schemes

## Important Implementation Notes

- **Uncertainty Scale**: UProp uncertainties are not normalized - higher values indicate higher uncertainty
- **Log Probabilities**: Extract token-level log probabilities for accurate uncertainty estimation
- **Trajectory Format**: Consistent structure with reasoning, action, observation, and log_prob fields
- **API Limits**: Be mindful of rate limits when running large evaluations
- **Reproducibility**: Set random seeds for consistent results across runs

## Entry Points

- **Research Experiments**: `python src/hotpot_qa_uprop.py`
- **DAG Analysis**: `python src/dag.py`
- **Interactive Development**: `jupyter notebook playground.ipynb`
- **Algorithm Testing**: Direct imports from `src/` modules


# Development Guidelines

## Philosophy

### Core Beliefs

- **Incremental progress over big bangs** - Small changes that compile and pass tests
- **Learning from existing code** - Study and plan before implementing
- **Pragmatic over dogmatic** - Adapt to project reality
- **Clear intent over clever code** - Be boring and obvious

### Simplicity Means

- Single responsibility per function/class
- Avoid premature abstractions
- No clever tricks - choose the boring solution
- If you need to explain it, it's too complex

## Process

### 1. Planning & Staging

Break complex work into 3-5 stages. Document in `IMPLEMENTATION_PLAN.md`:

```markdown
## Stage N: [Name]
**Goal**: [Specific deliverable]
**Success Criteria**: [Testable outcomes]
**Tests**: [Specific test cases]
**Status**: [Not Started|In Progress|Complete]
```
- Update status as you progress
- Remove file when all stages are done

### 2. Implementation Flow

1. **Understand** - Study existing patterns in codebase
2. **Test** - Write test first (red)
3. **Implement** - Minimal code to pass (green)
4. **Refactor** - Clean up with tests passing
5. **Commit** - With clear message linking to plan

### 3. When Stuck (After 3 Attempts)

**CRITICAL**: Maximum 3 attempts per issue, then STOP.

1. **Document what failed**:
   - What you tried
   - Specific error messages
   - Why you think it failed

2. **Research alternatives**:
   - Find 2-3 similar implementations
   - Note different approaches used

3. **Question fundamentals**:
   - Is this the right abstraction level?
   - Can this be split into smaller problems?
   - Is there a simpler approach entirely?

4. **Try different angle**:
   - Different library/framework feature?
   - Different architectural pattern?
   - Remove abstraction instead of adding?

## Technical Standards

### Architecture Principles

- **Composition over inheritance** - Use dependency injection
- **Interfaces over singletons** - Enable testing and flexibility
- **Explicit over implicit** - Clear data flow and dependencies
- **Test-driven when possible** - Never disable tests, fix them

### Error Handling

- Fail fast with descriptive messages
- Include context for debugging
- Handle errors at appropriate level
- Never silently swallow exceptions

## Decision Framework

When multiple valid approaches exist, choose based on:

1. **Testability** - Can I easily test this?
2. **Readability** - Will someone understand this in 6 months?
3. **Consistency** - Does this match project patterns?
4. **Simplicity** - Is this the simplest solution that works?
5. **Reversibility** - How hard to change later?

## Project Integration

### Learning the Codebase

- Find 3 similar features/components
- Identify common patterns and conventions
- Use same libraries/utilities when possible
- Follow existing test patterns

### Tooling

- Use project's existing build system
- Use project's test framework
- Use project's formatter/linter settings
- Don't introduce new tools without strong justification

## Quality Gates

### Definition of Done

- [ ] Tests written and passing
- [ ] Code follows project conventions
- [ ] No linter/formatter warnings
- [ ] Commit messages are clear
- [ ] Implementation matches plan
- [ ] No TODOs without issue numbers

### Test Guidelines

- Test behavior, not implementation
- One assertion per test when possible
- Clear test names describing scenario
- Use existing test utilities/helpers
- Tests should be deterministic

## Important Reminders

**NEVER**:
- Use `--no-verify` to bypass commit hooks
- Disable tests instead of fixing them
- Commit code that doesn't compile
- Make assumptions - verify with existing code

**ALWAYS**:
- Commit working code incrementally
- Update plan documentation as you go
- Learn from existing implementations
- Stop after 3 failed attempts and reassess
