# DiscoUQ: Structured Disagreement Analysis for Uncertainty Quantification in LLM Agent Ensembles

**Authors:** Bo Jiang

**Venue:** arXiv:2603.20975, March 2026

**Category:** Uncertainty Quantification / Multi-Agent

## Abstract

Multi-agent LLM systems where multiple prompted instances independently answer questions are increasingly used for complex reasoning tasks. However, existing methods for quantifying uncertainty rely on shallow voting statistics that discard the rich semantic information in agents' reasoning. DiscoUQ extracts and leverages the structure of inter-agent disagreement -- both linguistic properties (evidence overlap) and embedding geometry. Three increasingly sophisticated variants are presented: DiscoUQ-LLM uses logistic regression on LLM-extracted structure features, DiscoUQ-Embed applies similar regression to embedding geometry, and DiscoUQ-Learn combines all features in a neural network. Tested across four benchmarks (StrategyQA, MMLU, TruthfulQA, ARC-Challenge) with five Qwen3.5-27B agents, DiscoUQ-LLM achieved an average AUROC of 0.802, outperforming the LLM Aggregator baseline (0.791) with superior calibration.

## Key Contribution

Moves beyond simple voting to analyze the structure of disagreement (evidence overlap, embedding geometry) among LLM agents for uncertainty estimation.

## Relevance to Gas Town / MAUQ

DiscoUQ's structured disagreement analysis is complementary to MAUQ's graph-based and DAG-based approaches. MAUQ's AUROC of 0.9372 significantly outperforms DiscoUQ's 0.802, suggesting MAUQ's DAG approach captures richer structure. However, DiscoUQ's linguistic feature extraction could augment MAUQ's features.
