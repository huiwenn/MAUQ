# Research Abstracts: Multi-Agent UQ, DAG Workflows, Agent Systems

Collected 2026-04-15. 27 papers/posts across three research threads relevant to Gas Town and MAUQ.

## Index

### Multi-Agent Systems (01-08)
| # | Paper | Year | Key Topic |
|---|-------|------|-----------|
| 01 | [Why Do Multi-Agent LLM Systems Fail?](01_multi_agent_failure_taxonomy.md) | 2025 | Failure taxonomy (14 modes, 1600+ traces) |
| 02 | [Scaling Multi-Agent Collaboration (MacNet)](02_scaling_multi_agent_collaboration.md) | 2024/ICLR 2025 | Collaborative scaling laws, DAG topologies |
| 03 | [Mixture-of-Agents](03_mixture_of_agents.md) | 2024 | Layered refinement, beats GPT-4o |
| 04 | [When One LLM Drools, Multi-LLM Collaboration Rules](04_multi_llm_collaboration.md) | 2025 | Collaboration taxonomy by access level |
| 05 | [Agent Interoperability Protocols (MCP/ACP/A2A/ANP)](05_agent_interoperability_protocols.md) | 2025 | Communication protocol survey |
| 06 | [AutoGen](06_autogen.md) | 2023 | Foundational multi-agent framework |
| 07 | [LLM Multi-Agents Survey](07_llm_multi_agents_survey.md) | 2024 | Comprehensive field survey |
| 08 | [More Agents Is All You Need](08_more_agents_is_all_you_need.md) | 2024 | Naive scaling baseline |

### DAG / Agent Workflows (09-16)
| # | Paper | Year | Key Topic |
|---|-------|------|-----------|
| 09 | [Git Context Controller](09_git_context_controller.md) | 2025-2026 | Version-control for agent context |
| 10 | [ScienceClaw + Infinite](10_scienceclaw_infinite.md) | 2026 | Plannerless DAG-based scientific discovery |
| 11 | [POLARIS](11_polaris_typed_planning.md) | 2026 | Type-checked DAG synthesis, governance |
| 12 | [R-LAM](12_rlam_reproducibility.md) | 2026 | Reproducibility-constrained agents |
| 13 | [Halo](13_halo_batch_query.md) | 2025-2026 | DAG query optimization, 3.6x speedup |
| 14 | [AgentConductor](14_agent_conductor.md) | 2026 | Dynamic DAG topology generation |
| 15 | [Self-Healing Tool Routing](15_self_healing_tool_routing.md) | 2026 | Graph-based fault tolerance, 93% less LLM calls |
| 16 | [LangGraph](16_langgraph.md) | 2024-2026 | Industry standard graph orchestration |

### Uncertainty Quantification (17-27)
| # | Paper | Year | Key Topic |
|---|-------|------|-----------|
| 17 | [MATU (Tensor Decomposition)](17_matu_tensor_decomposition.md) | ACL 2026 | **Direct competitor** -- tensor UQ for multi-agent |
| 18 | [CoE (Collaborative Entropy)](18_coe_collaborative_entropy.md) | ICLR 2026 WS | Intra-model + inter-model entropy |
| 19 | [DiscoUQ](19_discouq_structured_disagreement.md) | 2026 | Structured disagreement analysis |
| 20 | [SAUP](20_saup_situation_awareness.md) | 2024 | Situation-aware uncertainty propagation |
| 21 | [CoT Entropy](21_cot_entropy_stepwise_verification.md) | 2025 | Step-wise verification UQ (Yarin Gal) |
| 22 | [TokUR](22_tokur_token_level_uq.md) | ICLR 2026 | Token-level UQ via weight perturbation |
| 23 | [Diversity-Steered Sampling](23_diversity_steered_semantic_uq.md) | NeurIPS 2025 | Sample-efficient semantic UQ |
| 24 | [ConU (Conformal Prediction)](24_conu_conformal_prediction.md) | EMNLP 2024 | Statistical coverage guarantees |
| 25 | [UQ Survey: Passive to Active](25_uq_survey_active_signal.md) | 2026 | Field survey, positioning reference |
| 26 | [CascadeDebate](26_cascade_debate.md) | 2026 | Cost-aware debate escalation |
| 27 | [CLUES (Ambiguity vs Instability)](27_clues_ambiguity_instability.md) | 2026 | Uncertainty decomposition via Schur complement |

---

## Synthesis: Where MAUQ Stands and Where to Go

### MAUQ's Current Strengths
- **DPIMPR AUROC 0.9372** outperforms all reported baselines (DiscoUQ: 0.802, SAUP: ~20% improvement over prior)
- DAG-based semantic node merging has **no direct parallel** in the literature
- Multi-agent UQ is an emerging field (most competitor papers are 2026) -- MAUQ is early

### Direct Competitors to Watch
1. **MATU** (ACL 2026) -- tensor decomposition approach, most direct competitor
2. **CoE** (ICLR 2026 WS) -- information-theoretic collaborative entropy
3. **DiscoUQ** -- structured disagreement (MAUQ already outperforms)

### High-Impact Contribution Directions

1. **Conformal prediction wrapper** -- Add ConU-style coverage guarantees on top of DPIMPR scores. No multi-agent UQ paper currently provides formal statistical guarantees. This would be a strong differentiator.

2. **Tensor + DAG hybrid** -- Combine MATU's tensor decomposition (good at isolating uncertainty sources) with DPIMPR's DAG structure (good at capturing reasoning topology). Each captures structure the other misses.

3. **Version-controlled uncertainty provenance** -- Gas Town's Dolt backend enables something no other system has: tracking how uncertainty evolves across bead versions. This is a novel contribution combining DAG workflows with UQ.

4. **Active UQ for agent routing** -- Use DPIMPR confidence scores to drive Gas Town's coordination decisions (when to spawn Polecats, when to escalate to Mayor, when to accept Crew output). This is the "passive to active" frontier the UQ survey identifies.

5. **Token-level base signals** -- Integrate TokUR's fine-grained uncertainty as input to DPIMPR, potentially improving the raw uncertainty estimates that flow through the DAG.

### Architectural Validation from the Literature
- **Beads = versioned artifacts** validated by GCC, ScienceClaw, R-LAM
- **Patrol cycles = adaptive DAG** validated by AgentConductor, Self-Healing Router
- **Heterogeneous roles > uniform agents** validated by MacNet (ICLR 2025)
- **Asynchronous coordination** fills a gap -- most frameworks are synchronous (AutoGen, LangGraph)
