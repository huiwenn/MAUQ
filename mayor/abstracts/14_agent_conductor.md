# AgentConductor: Topology Evolution for Multi-Agent Competition-Level Code Generation

**Authors:** Siyu Wang, Ruotian Lu, Zhihao Yang, Yuchao Wang, Yanzhou Zhang, Lei Xu, Qimin Xu, Guojun Yin, Cailian Chen, Xinping Guan

**Venue:** arXiv:2602.17100, February 19, 2026

**Category:** DAG / Agent Workflows / Dynamic Topology

## Abstract

Current multi-agent systems use fixed topologies that do not adapt to task complexity. AgentConductor introduces an LLM-based orchestrator that constructs task-adapted, density-aware layered directed acyclic graph (DAG) topologies dynamically. Uses difficulty interval partitioning and a novel density function to control graph sparsity. Achieves up to 14.6% improvement in pass@1 accuracy, 13% density reduction, and 68% token cost savings across five code datasets.

## Key Contribution

Dynamic DAG topology generation where the graph structure itself is adapted per-task by an orchestrator agent, moving beyond fixed multi-agent communication patterns to learned, density-controlled topologies.

## Relevance to Gas Town / MAUQ

AgentConductor's dynamic topology generation is relevant to Gas Town's non-static coordination model. Where Gas Town uses patrol cycles rather than pure static DAGs, AgentConductor validates that adaptive graph structures outperform fixed ones. The density-aware pruning parallels DPIMPR's semantic merging of reasoning nodes.
