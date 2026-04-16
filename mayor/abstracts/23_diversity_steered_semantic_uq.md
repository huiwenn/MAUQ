# Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling

**Authors:** Ji Won Park, Kyunghyun Cho

**Venue:** NeurIPS 2025. arXiv:2510.21310, October 2025 (revised January 2026)

**Category:** Uncertainty Quantification / Sampling Efficiency

## Abstract

Accurately estimating semantic aleatoric and epistemic uncertainties in LLMs is particularly challenging in free-form QA, where obtaining stable estimates often requires many expensive generations. The authors introduce a diversity-steered sampler that discourages semantically redundant outputs during decoding, covers both autoregressive and masked diffusion paradigms, and employs importance reweighting and control variates to improve efficiency. Testing across four QA benchmarks shows the approach achieves comparable or superior performance while exploring more semantic variations with equivalent sample sizes. The framework operates independently of the base model's gradients, making it suitable for production deployments.

## Key Contribution

A sampling strategy that actively steers toward semantic diversity for more sample-efficient uncertainty estimation, with importance reweighting to maintain unbiased estimates. Applicable to black-box models.

## Relevance to Gas Town / MAUQ

MAUQ's multi-agent debate generates multiple responses for UQ estimation. This diversity-steered sampling approach could make each agent's contributions more informative, reducing the number of debate rounds needed for reliable uncertainty estimates. The gradient-free nature makes it compatible with MAUQ's architecture.
