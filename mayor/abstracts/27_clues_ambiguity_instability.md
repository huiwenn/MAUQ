# Disentangling Ambiguity from Instability in Large Language Models

**Authors:** Ziletti & D'Ambrosi

**Venue:** arXiv:2602.12015, February 2026

**Category:** Uncertainty Quantification / Decomposition

## Summary

Introduces the CLUES framework that decomposes semantic uncertainty into two distinct sources: ambiguity (genuine question underspecification) and instability (model inconsistency on well-defined questions). Uses a bipartite semantic graph and Schur complement to perform this decomposition analytically.

## Key Contribution

First principled decomposition of semantic uncertainty into ambiguity vs. instability components, providing actionable signals (ambiguity -> ask for clarification; instability -> don't trust the model).

## Relevance to Gas Town / MAUQ

The ambiguity/instability decomposition concept directly parallels MAUQ's intrinsic/extrinsic uncertainty separation. MAUQ's intrinsic uncertainty (per-agent) maps roughly to instability; extrinsic uncertainty (cross-agent PMI) maps roughly to ambiguity. The Schur complement technique could provide a more principled mathematical foundation for MAUQ's decomposition.
