# Scaling Large Language Model-based Multi-Agent Collaboration

**Authors:** Chen Qian, Zihao Xie, YiFei Wang, Wei Liu, Kunlun Zhu, Hanchen Xia, Yufan Dang, Zhuoyun Du, Weize Chen, Cheng Yang, Zhiyuan Liu, Maosong Sun

**Venue:** arXiv:2406.07155, June 11, 2024 (accepted at ICLR 2025)

**Category:** Multi-Agent Systems

## Abstract

The research investigates whether adding more collaborative agents yields performance improvements similar to neural scaling laws. Agents are organized using directed acyclic graphs into a multi-agent collaboration network (MacNet) to manage interactive reasoning. Key findings: the system supports collaboration among over 1,000 agents; irregular topologies outperform regular ones; a "collaborative scaling law" emerges showing a logistic growth pattern as agents scale; and collaborative emergence occurs earlier than traditional neural emergence. The authors hypothesize that scaling agents encourages multidimensional considerations during interactive reflection, producing more comprehensive outputs.

## Key Contribution

First empirical demonstration of scaling laws for multi-agent LLM collaboration, showing logistic growth patterns and that irregular (non-uniform) network topologies outperform regular ones. Accepted at ICLR 2025.

## Relevance to Gas Town / MAUQ

The finding that irregular topologies outperform regular ones validates Gas Town's heterogeneous role design (Mayor, Witness, Crew, Polecats) over uniform agent pools. The logistic growth pattern suggests diminishing returns at scale, informing decisions about Polecat spawning limits and crew sizing.
