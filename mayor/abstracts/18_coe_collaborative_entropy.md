# CoE: Collaborative Entropy for Uncertainty Quantification in Agentic Multi-LLM Systems

**Authors:** Kangkang Sun, Jun Wu, Jianhua Li, Minyi Guo, Xiuzhen Che, Jianwei Huang

**Venue:** ICLR 2026 Workshop "Agentic AI in the Wild: From Hallucinations to Reliable Autonomy". arXiv:2603.28360, March 2026

**Category:** Uncertainty Quantification / Multi-Agent

## Abstract

Uncertainty estimation in multi-LLM systems remains largely single-model-centric: existing methods quantify uncertainty within each model but do not adequately capture semantic disagreement across models. CoE proposes a unified information-theoretic metric for semantic uncertainty in multi-LLM collaboration. CoE is defined on a shared semantic cluster space and combines two components: intra-model semantic entropy and inter-model divergence to the ensemble centroid. The paper includes theoretical analysis of the metric's properties and a practical coordination strategy. Experiments using LLaMA, Qwen, and Mistral models on QA datasets demonstrate improved UQ compared to existing entropy and divergence-based methods.

## Key Contribution

An information-theoretic framework that unifies intra-model semantic entropy with inter-model divergence into a single collaborative entropy metric for multi-LLM ensembles.

## Relevance to Gas Town / MAUQ

**Very high.** CoE's decomposition into intra-model semantic entropy + inter-model divergence is directly comparable to MAUQ's normalized entropy + PMI approach in the graph-based UQ module. The shared semantic cluster space concept could improve MAUQ's debate uncertainty measurement. Compare AUROC against MAUQ's 0.9372.
