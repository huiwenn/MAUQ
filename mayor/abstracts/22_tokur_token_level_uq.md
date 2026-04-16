# TokUR: Token-Level Uncertainty Estimation for Large Language Model Reasoning

**Authors:** Tunyu Zhang, Haizhou Shi, Yibin Wang, Hengyi Wang, Xiaoxiao He, Zhuowei Li, Haoxian Chen, Ligong Han, Kai Xu, Huan Zhang, Dimitris Metaxas, Hao Wang

**Venue:** ICLR 2026. arXiv:2505.11737, May 2025 (revised April 2026)

**Category:** Uncertainty Quantification / Token-Level

## Abstract

LLM output quality remains inconsistent, making it difficult to identify trustworthy responses in complex multi-step reasoning. TokUR proposes a token-level uncertainty estimation framework that enables LLMs to self-assess and self-improve their responses in mathematical reasoning. The approach uses low-rank random weight perturbation during LLM decoding to generate predictive distributions for token-level uncertainty estimation, with aggregated measurements capturing semantic uncertainty. Testing on mathematical reasoning datasets shows the method correlates strongly with answer correctness and model robustness, with uncertainty signals improving reasoning performance at test time.

## Key Contribution

Token-level uncertainty estimation via low-rank weight perturbation during decoding, providing fine-grained uncertainty signals that can be aggregated to semantic-level measures and used for test-time self-improvement.

## Relevance to Gas Town / MAUQ

TokUR provides a finer-grained uncertainty signal (token-level) than MAUQ currently uses. The low-rank perturbation approach could be used as an improved base uncertainty estimator feeding into MAUQ's propagation framework. The token-to-semantic aggregation method is complementary to MAUQ's normalized entropy approach.
