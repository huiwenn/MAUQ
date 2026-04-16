# Graph-Based Self-Healing Tool Routing for Cost-Efficient LLM Agents

**Authors:** Neeraj Bholani

**Venue:** arXiv:2603.01548, March 2, 2026

**Category:** DAG / Agent Workflows / Fault Tolerance

## Abstract

Introduces Self-Healing Router, which treats most agent control-flow decisions as routing rather than reasoning. Employs a cost-weighted tool graph using Dijkstra's algorithm for deterministic pathfinding. When a tool fails mid-execution, its edges are reweighted to infinity and the path is recomputed, yielding automatic recovery without invoking the LLM. Parallel health monitors assign priority scores to runtime conditions. Matches ReAct correctness while reducing LLM control-plane calls by 93% and eliminating silent failures in compound tool failure scenarios.

## Key Contribution

Separates agent control flow (graph-based routing) from reasoning (LLM calls), achieving fault-tolerant execution through graph reweighting rather than LLM re-prompting -- a 93% reduction in control-plane overhead.

## Relevance to Gas Town / MAUQ

The self-healing graph with runtime edge reweighting is directly relevant to Gas Town's patrol cycle model for error recovery. The principle of treating control flow as graph routing (not LLM reasoning) aligns with Gas Town's separation of coordination logic (patrol cycles, bead state machines) from LLM reasoning within beads.
