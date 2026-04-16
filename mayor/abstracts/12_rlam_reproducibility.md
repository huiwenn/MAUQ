# R-LAM: Reproducibility-Constrained Large Action Models for Scientific Workflow Automation

**Authors:** Suriya Sureshkumar

**Venue:** arXiv:2601.09749, January 12, 2026

**Category:** DAG / Agent Workflows / Reproducibility

## Abstract

Standard LLM-based agents lack the deterministic guarantees required for scientific work -- unconstrained action generation can lead to silent state changes, non-deterministic executions, and irreproducible experimental results. R-LAM introduces structured action schemas, deterministic execution policies, and explicit provenance tracking to ensure all actions remain auditable and replayable. Supports failure-aware execution loops and controlled workflow forking. Released as open-source Python package via PyPI.

## Key Contribution

First framework to formally constrain LLM agent actions for reproducibility through structured schemas and provenance tracking, treating reproducibility as a first-class architectural concern rather than an afterthought.

## Relevance to Gas Town / MAUQ

R-LAM's "structured action schemas with provenance tracking" directly validates Gas Town's approach of using beads as discrete, auditable work units. The "controlled workflow forking" maps to Dolt's branching model. R-LAM's concern with "silent state changes" is exactly what Gas Town's state machine (open/hooked/closed) is designed to prevent.
