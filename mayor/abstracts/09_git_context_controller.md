# Git Context Controller: Manage the Context of LLM-based Agents like Git

**Authors:** Junde Wu, Minhao Hu, Jiayuan Zhu, Jiazhen Pan, Yuyuan Liu, Min Xu, Yueming Jin

**Venue:** arXiv:2508.00031, July 30, 2025 (v2: March 1, 2026)

**Category:** DAG / Agent Workflows / Version Control

## Abstract

Large language model agents face a fundamental bottleneck as interaction histories grow unbounded and become costly to maintain. Git-Context-Controller (GCC) elevates agent memory from temporary token streams to persistent, navigable workspaces with explicit COMMIT, BRANCH, MERGE, and CONTEXT operations. This enables milestone-based checkpointing, isolated exploration of alternative reasoning paths, and hierarchical retrieval of historical information. On SWE-Bench Verified, GCC achieves over 80% success rate (13% relative improvement over long-context baselines).

## Key Contribution

Directly applies version-control semantics (commit, branch, merge) to LLM agent context management, treating reasoning trajectories as a versioned file system with persistent memory across sessions.

## Relevance to Gas Town / MAUQ

The closest academic analog to Gas Town's architecture. GCC's COMMIT/BRANCH/MERGE operations mirror Dolt's version-control primitives for agent state. The "versioned file system" concept parallels Gas Town's bead state machine (open/hooked/closed) where each bead is a committed unit of work. GCC validates the core thesis that agent state should be version-controlled, not ephemeral.
