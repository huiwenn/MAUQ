# Every Response Counts: Quantifying Uncertainty of LLM-based Multi-Agent Systems through Tensor Decomposition

**Authors:** Tiejin Chen, Huaiyuan Yao, Jia Chen, Evangelos E. Papalexakis, Hua Wei

**Venue:** ACL 2026 (accepted). arXiv:2604.08708, April 2026

**Category:** Uncertainty Quantification / Multi-Agent

## Abstract

Multi-agent LLM systems consistently outperform single-agent systems on complex tasks, but their intricate interactions introduce critical reliability challenges arising from communication dynamics and role dependencies. Existing UQ methods, designed for single-turn outputs, fail to address the unique complexities of MAS. Specifically, these methods struggle with cascading uncertainty in multi-step interactions, role-dependent uncertainty from specialized agents, and topology-sensitive uncertainty from different communication patterns. The paper introduces MATU, a framework using tensor decomposition that represents complete reasoning trajectories as embedding matrices and organizes multiple execution runs into higher-order tensors. Through tensor decomposition, the method isolates and measures various uncertainty sources, delivering a comprehensive reliability assessment applicable across different agent architectures.

## Key Contribution

First principled framework for decomposing uncertainty in multi-agent LLM systems via tensor methods, separating cascading, role-dependent, and topology-sensitive uncertainty components.

## Relevance to Gas Town / MAUQ

**Extremely high -- most direct competitor.** Addresses the same core problem (UQ for multi-agent LLM systems) but uses tensor decomposition rather than graph-based entropy/PMI or DAG-based approaches. The decomposition into cascading/role/topology uncertainty sources directly parallels MAUQ's intrinsic/extrinsic decomposition in UProp. MAUQ should benchmark against this and consider whether tensor methods could complement the existing DAG-DPIMPR approach.
