# More Agents Is All You Need

**Authors:** Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, Deheng Ye

**Venue:** arXiv:2402.05120, February 3, 2024 (revised October 11, 2024)

**Category:** Multi-Agent Systems

## Abstract

The researchers discovered that simply via a sampling-and-voting method, the performance of large language models scales with the number of agents instantiated. Their approach operates independently from other LLM enhancement techniques, with effectiveness correlating to problem complexity. They validated findings across multiple benchmarks and made their implementation publicly available.

## Key Contribution

Demonstrates the simplest possible scaling result for multi-agent systems: even naive sampling-and-voting across multiple agent instances improves performance, with gains proportional to task difficulty. A clean empirical baseline for multi-agent scaling.

## Relevance to Gas Town / MAUQ

Establishes the theoretical floor for Gas Town's value proposition. Even without Gas Town's sophisticated coordination (hooks, mail, specialized roles), simply running multiple agents and voting improves results. This means Gas Town's orchestration overhead must demonstrably outperform naive sampling-and-voting to justify its complexity. Also relevant to Polecat (ephemeral worker) design -- spawning multiple Polecats on the same task with majority voting is a valid strategy.
