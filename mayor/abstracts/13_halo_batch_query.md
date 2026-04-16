# Halo: Batch Query Processing and Optimization for Agentic Workflows

**Authors:** Junyi Shen, Noppanat Wadlom, Yao Lu

**Venue:** arXiv:2509.02121, September 2, 2025 (v2: January 19, 2026)

**Category:** DAG / Agent Workflows / Optimization

## Abstract

Halo represents each agentic workflow as a structured query plan DAG and constructs a consolidated graph for batched queries that exposes shared computation. A cost model jointly considers heterogeneous resource constraints, prefill and decode costs, cache reuse, and GPU placement to minimize redundant operations. The Processor implements adaptive batching, KV-cache sharing and migration, and fine-grained CPU-GPU pipelining. Achieves up to 3.6x speedup for batch inference and 2.6x throughput enhancement during online serving.

## Key Contribution

Treats agentic workflows as formal query plan DAGs amenable to database-style optimization (shared subexpression elimination, cost-based planning), achieving dramatic efficiency gains through graph-level analysis.

## Relevance to Gas Town / MAUQ

Halo's treatment of agent workflows as query plan DAGs with cost-based optimization validates the idea that agent execution graphs should be first-class data structures. The "shared computation across batched queries" concept could inform optimization of Gas Town's patrol cycles when multiple beads share common subgraph patterns.
