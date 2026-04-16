# Why Do Multi-Agent LLM Systems Fail?

**Authors:** Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica

**Venue:** arXiv:2503.13657, March 17, 2025 (revised October 26, 2025)

**Category:** Multi-Agent Systems

## Abstract

Despite enthusiasm for Multi-Agent LLM Systems (MAS), their performance gains on popular benchmarks are often minimal. This gap highlights a critical need for a principled understanding of why MAS fail. Addressing this question requires systematic identification and analysis of failure patterns. We introduce MAST-Data, a comprehensive dataset of 1600+ annotated traces collected across 7 popular MAS frameworks. MAST-Data is the first multi-agent system dataset to outline the failure dynamics in MAS for guiding the development of better future systems. To enable systematic classification of failures for MAST-Data, we build the first Multi-Agent System Failure Taxonomy (MAST). We develop MAST through rigorous analysis of 150 traces, guided closely by expert human annotators and validated by high inter-annotator agreement (kappa = 0.88). This process identifies 14 unique modes, clustered into 3 categories: (i) system design issues, (ii) inter-agent misalignment, and (iii) task verification. To enable scalable annotation, we develop an LLM-as-a-Judge pipeline with high agreement with human annotations. We leverage MAST and MAST-Data to analyze failure patterns across models (GPT4, Claude 3, Qwen2.5, CodeLlama) and tasks (coding, math, general agent), demonstrating improvement headrooms from better MAS design. Our analysis provides insights revealing that identified failures require more sophisticated solutions, highlighting a clear roadmap for future research.

## Key Contribution

First systematic failure taxonomy for multi-agent LLM systems, identifying 14 failure modes across 3 categories (system design, inter-agent misalignment, task verification) from 1600+ annotated traces across 7 frameworks.

## Relevance to Gas Town / MAUQ

The three failure categories map onto Gas Town concerns: system design issues relate to hook/patrol architecture; inter-agent misalignment relates to mail-based communication fidelity between Mayor/Crew/Witness; task verification relates to Refinery merge validation. The taxonomy can inform Witness monitoring patterns and Mayor coordination strategies.
