# When One LLM Drools, Multi-LLM Collaboration Rules

**Authors:** Shangbin Feng, Wenxuan Ding, Alisa Liu, Zifeng Wang, Weijia Shi, Yike Wang, Zejiang Shen, Xiaochuang Han, Hunter Lang, Chen-Yu Lee, Tomas Pfister, Yejin Choi, Yulia Tsvetkov

**Venue:** arXiv:2502.04506, February 6, 2025

**Category:** Multi-Agent Systems

## Abstract

This position paper argues that in many realistic (i.e., complex, contextualized, subjective) scenarios, one LLM is not enough to produce a reliable output. We challenge the status quo of relying solely on a single general-purpose LLM and argue for multi-LLM collaboration to better represent the extensive diversity of data, skills, and people. We first posit that a single LLM underrepresents real-world data distributions, heterogeneous skills, and pluralistic populations, and that such representation gaps cannot be trivially patched by further training a single LLM. We then organize existing multi-LLM collaboration methods into a hierarchy, based on the level of access and information exchange, ranging from API-level, text-level, logit-level, to weight-level collaboration. Based on these methods, we highlight how multi-LLM collaboration addresses challenges that a single LLM struggles with, such as reliability, democratization, and pluralism. Finally, we identify the limitations of existing multi-LLM methods and motivate future work. We envision multi-LLM collaboration as an essential path toward compositional intelligence and collaborative AI development.

## Key Contribution

Provides a principled taxonomy of multi-LLM collaboration methods organized by information-exchange depth (API-level through weight-level), and argues theoretically why single-model approaches are fundamentally insufficient for complex real-world tasks.

## Relevance to Gas Town / MAUQ

Gas Town operates at the "text-level collaboration" tier in this taxonomy (agents communicate via mail messages containing text). The paper's framework helps reason about what Gas Town gains and loses at this level versus tighter integration. The argument for heterogeneous skills validates Gas Town's specialized roles (different agents for different capabilities).
