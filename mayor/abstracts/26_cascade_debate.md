# CascadeDebate: Multi-Agent Deliberation for Cost-Aware LLM Cascades

**Authors:** Chang et al.

**Venue:** arXiv:2604.12262, April 2026

**Category:** Multi-Agent / Uncertainty Quantification

## Summary

Introduces multi-agent deliberation at escalation boundaries with confidence-based routing. When cheaper models are uncertain, the system escalates to more capable (and expensive) models through a structured debate mechanism. Combines UQ signals with cost-aware routing to minimize inference cost while maintaining quality.

## Key Contribution

Bridges multi-agent debate with practical cost optimization by using uncertainty as the escalation trigger in model cascades.

## Relevance to Gas Town / MAUQ

Relevant to MAUQ's debate architecture -- demonstrates how UQ signals can drive practical routing decisions (when to escalate, when to accept). This cost-aware dimension is something MAUQ could incorporate: using DAG confidence to decide whether additional debate rounds are worth the compute.
