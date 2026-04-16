# Mixture-of-Agents Enhances Large Language Model Capabilities

**Authors:** Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, James Zou

**Venue:** arXiv:2406.04692, June 7, 2024

**Category:** Multi-Agent Systems

## Abstract

The paper presents a Mixture-of-Agents (MoA) layered architecture where multiple LLMs work collectively. Each agent takes all outputs from agents in the previous layer as auxiliary information in generating its response. The approach achieved leading performance on several benchmarks -- MoA using only open-source LLMs achieved a score of 65.1% on AlpacaEval 2.0 compared to 57.5% by GPT-4 Omni. Surpassed GPT-4o on AlpacaEval 2.0, MT-Bench, and FLASK.

## Key Contribution

Introduced the Mixture-of-Agents layered architecture where each layer's agents refine outputs from the previous layer, demonstrating that collaboration among open-source models can surpass frontier closed-source models.

## Relevance to Gas Town / MAUQ

The layered refinement pattern (proposers -> aggregators) maps directly onto Gas Town's Crew -> Refinery pipeline. When Crew agents produce work outputs that flow through Refinery for merge/consolidation, this is essentially a two-layer MoA. The paper validates that this sequential refinement architecture genuinely improves output quality.
