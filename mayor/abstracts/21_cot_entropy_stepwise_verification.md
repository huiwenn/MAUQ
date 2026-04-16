# Uncertainty-Aware Step-wise Verification with Generative Reward Models

**Authors:** Zihuiwen Ye, Luckeciano Carvalho Melo, Younesse Kaddar, Phil Blunsom, Sam Staton, Yarin Gal

**Venue:** arXiv:2502.11250, February 2025

**Category:** Uncertainty Quantification / Process Reward Models

## Abstract

Complex multi-step reasoning tasks remain challenging for LLMs. Process reward models (PRMs) provide intermediate rewards to verify step-wise correctness in solution traces. However, PRMs suffer from reliability issues, including susceptibility to reward hacking. This work proposes leveraging uncertainty quantification to strengthen PRMs. The authors introduce "CoT Entropy," a novel technique that outperforms existing approaches in quantifying a PRM's uncertainty in step-wise verification. Combining uncertainty estimates enhances robustness and dependability in the verification process.

## Key Contribution

CoT Entropy -- a new UQ metric specifically designed for step-wise verification of chain-of-thought reasoning, integrated with process reward models.

## Relevance to Gas Town / MAUQ

CoT Entropy provides a complementary approach to measuring uncertainty in multi-step reasoning chains. MAUQ's DAG-based DPIMPR already captures step-wise structure, but CoT Entropy's integration with process reward models could inform how MAUQ verifies individual reasoning steps within the debate. From Yarin Gal's group (Oxford), a leading UQ research lab.
