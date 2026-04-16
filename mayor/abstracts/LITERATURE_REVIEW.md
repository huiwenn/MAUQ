# Uncertainty Quantification and Workflow Formalization in Multi-Agent Large Language Model Systems: A Literature Review

## Abstract

Multi-agent systems built on large language models (LLMs) have emerged as a dominant paradigm for complex reasoning, software engineering, and scientific discovery tasks. However, these systems introduce compounding reliability challenges: cascading errors across agent interactions, false consensus among collaborating models, and the absence of formal guarantees on output correctness. This review surveys recent literature (2024--2026) across three converging research threads: (1) uncertainty quantification for multi-agent LLM systems, (2) directed acyclic graph (DAG) formalizations of agent workflows, and (3) coordination architectures for multi-agent orchestration. We synthesize 27 papers and identify key open problems at the intersection of these areas, with particular attention to how DAG-structured reasoning can serve as a substrate for both workflow coordination and uncertainty estimation.

---

## 1. Introduction

The deployment of LLMs as autonomous agents---systems capable of tool use, multi-step reasoning, and inter-agent communication---has progressed rapidly from single-agent prompting to elaborate multi-agent architectures. Wu et al. (2023) established the foundational paradigm with AutoGen, demonstrating that customizable, conversable agents with flexible interaction patterns could address tasks spanning mathematics, coding, and decision-making. Since then, the field has expanded along several axes: scaling the number of collaborating agents (Qian et al., 2024; Li et al., 2024), formalizing communication protocols (Ehtesham et al., 2025), and attempting to understand why these systems fail despite their promise (Cemri et al., 2025).

Three fundamental problems remain insufficiently addressed. First, multi-agent systems can produce *confident consensus on incorrect answers*---a failure mode empirically observed at rates as high as 89% consensus with only 42% accuracy (Duan et al., 2025). Second, the informal, ad-hoc nature of most agent workflow specifications makes it difficult to reproduce, audit, or optimize multi-agent executions. Third, existing uncertainty quantification (UQ) methods, designed primarily for single-model, single-turn settings, do not capture the complex dependency structures that emerge when multiple agents interact across multiple rounds.

This review organizes the literature around these three problems and identifies where recent advances create opportunities for novel contributions.

## 2. Multi-Agent LLM Architectures and Their Limitations

### 2.1 Foundational Frameworks

The modern landscape of multi-agent LLM systems traces to AutoGen (Wu et al., 2023), which introduced the abstraction of conversable agents operating in flexible interaction patterns combining LLMs, human inputs, and tools. AutoGen's synchronous conversation-based architecture has been widely adopted, with over 2,000 citations, but its reliance on rigid turn-taking creates bottlenecks when agents must coordinate asynchronously or when the number of agents exceeds what sequential conversation can support.

Guo et al. (2024) provided the first comprehensive survey of the field, establishing a taxonomy of agent profiling mechanisms, communication patterns (cooperative, debate, competitive), and capability enhancement methods. This taxonomy remains the standard reference for classifying multi-agent architectures and situating new systems within the broader design space.

### 2.2 Scaling Laws and Topology

A central question in multi-agent system design is whether performance scales predictably with the number of agents. Li et al. (2024) demonstrated the simplest possible affirmative result: naive sampling-and-voting across multiple agent instances improves performance proportional to task difficulty, establishing a theoretical floor that any sophisticated orchestration must exceed. However, this baseline says nothing about how to *organize* agents.

Qian et al. (2024), in work accepted at ICLR 2025, addressed this directly by organizing agents into directed acyclic graphs within a multi-agent collaboration network (MacNet). Their key finding---that *irregular* topologies outperform regular ones, and that a logistic collaborative scaling law governs performance as agents increase---has significant implications for system design. It suggests that heterogeneous role assignments (e.g., specialized coordinator, monitor, and worker agents) are not merely convenient engineering choices but reflect a deeper structural advantage validated empirically at scales exceeding 1,000 agents.

Wang et al. (2024) reinforced this finding from a different angle with Mixture-of-Agents (MoA), a layered architecture where each agent incorporates all outputs from the previous layer. MoA achieved 65.1% on AlpacaEval 2.0 using only open-source models, surpassing GPT-4o's 57.5%, demonstrating that the *topology* of collaboration can compensate for individual model capability.

### 2.3 The Case for Multi-Model Collaboration

Feng et al. (2025) offered a theoretical grounding for why multi-agent collaboration is not merely beneficial but *necessary*. Their position paper argues that single LLMs fundamentally underrepresent real-world data distributions, heterogeneous skills, and pluralistic populations---gaps that cannot be trivially patched by further training. They organize existing multi-LLM collaboration methods into a hierarchy by information-exchange depth: API-level, text-level, logit-level, and weight-level. This taxonomy is useful for situating practical systems: most deployed multi-agent frameworks (AutoGen, LangGraph, Gas Town) operate at the text level, exchanging natural language messages, which preserves modularity at the cost of information bandwidth.

### 2.4 Failure Modes

Perhaps the most sobering contribution to the multi-agent literature is MAST (Cemri et al., 2025), which introduces the first systematic failure taxonomy for multi-agent LLM systems. Through rigorous analysis of 1,600+ annotated traces across seven popular frameworks, they identify 14 unique failure modes clustered into three categories: system design issues, inter-agent misalignment, and task verification failures. The high inter-annotator agreement ($\kappa = 0.88$) lends confidence to the taxonomy's validity. Critically, many identified failures---such as cascading misinterpretation across agent boundaries and verification gaps where no agent takes responsibility for checking final output---require *architectural* solutions rather than mere prompt engineering.

### 2.5 Communication Protocols

Ehtesham et al. (2025) surveyed the four emerging agent communication protocols: MCP (JSON-RPC tool invocation), ACP (RESTful multimodal messaging), A2A (peer-to-peer task delegation via capability-based Agent Cards), and ANP (decentralized discovery via W3C DIDs). Their proposed phased adoption roadmap---from tool access through structured messaging to decentralized marketplaces---provides a useful framework for evaluating where any given multi-agent system sits on the maturity spectrum and what capabilities it could gain by adopting standardized protocols.

## 3. DAG-Based Formalizations of Agent Workflows

### 3.1 From Chains to Graphs

The evolution from linear chain-of-thought prompting to graph-structured agent execution represents a qualitative shift in how agent workflows are conceptualized. LangGraph (Chase et al., 2024--2026), the most widely adopted industry framework, explicitly supports cyclic graphs with typed state objects, conditional edges determined at runtime, and Postgres-backed persistence with time-travel debugging. Its three multi-agent patterns---collaboration via shared scratchpad, supervisor-based routing, and hierarchical teams via nested sub-graphs---have become de facto standards.

However, LangGraph's centralized StateGraph model constrains workflows to a single execution context. Recent academic work has explored more powerful formalizations.

### 3.2 Version-Controlled Agent State

Wu et al. (2025--2026) introduced Git Context Controller (GCC), which elevates agent memory from ephemeral token streams to persistent, navigable workspaces with explicit COMMIT, BRANCH, MERGE, and CONTEXT operations. By treating reasoning trajectories as a versioned file system, GCC enables milestone-based checkpointing, isolated exploration of alternative reasoning paths, and hierarchical retrieval---achieving over 80% success rate on SWE-Bench Verified (a 13% relative improvement over long-context baselines).

GCC validates a thesis that several concurrent works share: that agent state should be *version-controlled*, not ephemeral. Sureshkumar (2026) independently arrived at similar conclusions with R-LAM, which introduces structured action schemas, deterministic execution policies, and explicit provenance tracking to ensure all agent actions remain auditable and replayable. R-LAM frames reproducibility as a first-class architectural concern, supporting failure-aware execution loops and controlled workflow forking.

### 3.3 DAG-Based Provenance and Scientific Discovery

Wang et al. (2026) presented ScienceClaw, a framework for autonomous scientific investigation where independent agents produce immutable artifacts with full computational lineage tracked as a DAG. The system's key innovation is *plannerless coordination*: agents coordinate not through a central scheduler but through schema-overlap matching on a shared artifact space. An autonomous mutation layer prunes the expanding DAG to resolve conflicting workflows. This approach demonstrates that DAGs can serve simultaneously as coordination mechanisms and provenance records.

Moslemi et al. (2026) took a complementary approach with POLARIS, treating enterprise automation as typed plan synthesis over DAGs with formal governance constraints. A planner proposes structurally diverse, type-checked DAGs; a rubric-guided module selects a compliant plan; and execution is guarded by validator gates, bounded repair loops, and compiled policy guardrails. The emphasis on auditability and policy alignment distinguishes this work from research-oriented frameworks.

### 3.4 Dynamic and Adaptive Topologies

A limitation of static DAG specifications is that optimal agent topology depends on task complexity. Wang et al. (2026) addressed this with AgentConductor, which uses an LLM-based orchestrator to construct task-adapted, density-aware DAG topologies dynamically. Using difficulty interval partitioning and a density function to control graph sparsity, AgentConductor achieves up to 14.6% improvement in pass@1 accuracy with 68% token cost savings, demonstrating that adaptive structures outperform fixed ones.

Bholani (2026) took this further by separating agent control flow entirely from LLM reasoning. The Self-Healing Router treats most control-flow decisions as graph routing via Dijkstra's algorithm on a cost-weighted tool graph. When a tool fails, its edges are reweighted and the path is recomputed---automatic recovery without invoking the LLM. This approach matches ReAct correctness while reducing LLM control-plane calls by 93%, suggesting that much of what we currently delegate to LLM reasoning is better handled by classical graph algorithms.

### 3.5 Computational Optimization

Shen et al. (2025--2026) introduced Halo, which represents agentic workflows as formal query plan DAGs and applies database-style optimization: shared subexpression elimination, cost-based planning, adaptive batching, and KV-cache sharing. Halo achieves up to 3.6x speedup for batch inference, demonstrating that once agent workflows are formalized as graphs, the full toolkit of query optimization becomes applicable.

## 4. Uncertainty Quantification for Multi-Agent Systems

### 4.1 Single-Agent Foundations

Uncertainty quantification for LLMs has evolved from token-level entropy to semantic-level measures. Kuhn et al. (2023) introduced semantic entropy, which clusters generations by meaning rather than surface form, providing more meaningful uncertainty estimates for open-ended generation. Wang et al. (2024) extended this to conformal prediction with ConU, applying self-consistency-based uncertainty metrics within a conformal framework to provide finite-sample statistical coverage guarantees on answer correctness across seven LLMs.

Zhang et al. (2025--2026) proposed TokUR, which operates at the token level using low-rank random weight perturbation during decoding to generate predictive distributions. The aggregated measurements capture semantic uncertainty while providing finer granularity than generation-level methods. Accepted at ICLR 2026, TokUR demonstrates that token-level and semantic-level UQ are complementary rather than competing approaches.

Park and Cho (2025) addressed efficiency with a diversity-steered sampler that discourages semantically redundant outputs during decoding, published at NeurIPS 2025. By employing importance reweighting and control variates, their approach achieves comparable or superior uncertainty estimates with fewer samples---critical for multi-agent settings where each "sample" is an expensive agent interaction.

### 4.2 Multi-Step Uncertainty Propagation

The transition from single-step to multi-step reasoning introduces cascading uncertainty that standard methods fail to capture. Zhao et al. (2024) introduced SAUP, which propagates uncertainty through each step of an LLM agent's reasoning process with situational weights, achieving up to 20% improvement in AUROC over methods that consider only final-step outputs. However, SAUP remains focused on single-agent multi-step reasoning and does not address the cross-agent dependencies that arise in collaborative settings.

Ye et al. (2025) approached the multi-step problem through process reward models, introducing CoT Entropy as a UQ metric for step-wise verification of chain-of-thought reasoning. This work, from Yarin Gal's Oxford UQ group, integrates uncertainty signals directly into the reward model, enhancing robustness against reward hacking. The integration of UQ into verification (rather than treating it as a passive diagnostic) represents an important conceptual shift.

Duan et al. (2025) proposed UProp, which explicitly models multi-step uncertainty propagation through Trajectory-Dependent Decision Processes (TDPs). UProp decomposes uncertainty at each step into intrinsic uncertainty (length-normalized negative log-probability, $\text{IU}_t = \frac{1}{N} \sum_n \frac{-\log p(y_t^{(n)})}{|y_t^{(n)}|}$) and extrinsic uncertainty (pointwise mutual information across steps via Gaussian kernel estimation). A step-normalization factor $\sigma_t = 1 + \text{EU}_t/\text{IU}_t$ weighs the contribution of each step to the total uncertainty.

### 4.3 Multi-Agent Uncertainty Quantification

The extension of UQ to multi-agent settings is a nascent but rapidly growing subfield, with several significant contributions appearing in 2026.

Chen et al. (2026), in work accepted at ACL 2026, introduced MATU, which uses tensor decomposition to quantify uncertainty in multi-agent systems. MATU represents complete reasoning trajectories as embedding matrices and organizes multiple execution runs into higher-order tensors. Through tensor decomposition, the method isolates three uncertainty sources: cascading uncertainty from multi-step interactions, role-dependent uncertainty from specialized agents, and topology-sensitive uncertainty from different communication patterns. This decomposition is principled and general, but operates on embedding representations rather than the explicit reasoning structure.

Sun et al. (2026) proposed Collaborative Entropy (CoE) at the ICLR 2026 Workshop on Agentic AI, defining a unified information-theoretic metric on a shared semantic cluster space. CoE combines intra-model semantic entropy with inter-model divergence to the ensemble centroid, providing theoretical guarantees on the metric's properties alongside practical coordination strategies. The decomposition into intra-model and inter-model components parallels classical bias-variance decomposition.

Jiang (2026) introduced DiscoUQ, which extracts and leverages the *structure* of inter-agent disagreement---both linguistic properties (evidence overlap) and embedding geometry---rather than relying on shallow voting statistics. Three variants of increasing sophistication achieve AUROC up to 0.802 on multi-agent QA benchmarks, outperforming LLM aggregator baselines with superior calibration.

Ziletti and D'Ambrosi (2026) offered CLUES, which decomposes semantic uncertainty into ambiguity (genuine question underspecification) versus instability (model inconsistency on well-defined questions) using a bipartite semantic graph and Schur complement. This decomposition provides actionable signals: ambiguity suggests asking for clarification, while instability suggests the model should not be trusted.

### 4.4 DAG-Based Uncertainty Quantification

A distinct approach to multi-agent UQ leverages the graph structure of agent interactions directly. The DPIMPR algorithm (DAG-Preserving Incremental Multi-Path Reasoning) builds a directed acyclic graph by semantically merging similar reasoning nodes across multiple agent trajectories. Node uncertainty is computed as scaled log-probability with an evidence accumulation term: $u_{\text{merged}} = \frac{u_{\text{old}} \cdot w_{\text{old}} + u_{\text{new}} \cdot w_{\text{new}}}{w_{\text{old}} + w_{\text{new}}} - 0.1 \log(k+1)$, where $k$ is the evidence count. DAG confidence integrates topological position, evidence strength, and uncertainty into a composite score:

$$\text{confidence} = \frac{\sum_c p(c) \cdot w_{\text{topo}} \cdot w_{\text{evidence}} \cdot w_{\text{uncertainty}}}{\sum w}$$

where $w_{\text{topo}} = (\text{level}+1)/\text{max\_level}$, $w_{\text{evidence}} = \text{evidence\_count}$, and $w_{\text{uncertainty}} = 1/(\text{uncertainty} + 0.1)$.

On a benchmark of 100 multi-agent debates, DPIMPR achieves an AUROC of 0.9372 for predicting answer correctness, dramatically outperforming graph-based entropy methods (AUROC 0.32--0.41) and DiscoUQ's 0.802. This strong performance appears to stem from two properties unique to the DAG formulation: (1) semantic merging across trajectories creates a compressed representation that preserves reasoning topology while accumulating evidence, and (2) the evidence accumulation term explicitly models the epistemic benefit of corroborating reasoning paths. Notably, the benchmark reveals that agents achieve 89% consensus but only 42% accuracy---DAG confidence correctly identifies when agents confidently converge on wrong answers.

### 4.5 From Passive Measurement to Active Control

Zhang et al. (2026) surveyed the evolving role of UQ in LLMs, charting its progression from passive diagnostic to active control signal. This trajectory is particularly relevant for multi-agent systems, where uncertainty estimates can drive real-time coordination decisions: when to escalate to a more capable agent, when to spawn additional workers, and when to accept or reject collaborative output. Chang et al. (2026) demonstrated this principle with CascadeDebate, using confidence-based routing at escalation boundaries in multi-agent cascades to minimize inference cost while maintaining quality.

## 5. Synthesis and Open Problems

### 5.1 Convergent Themes

Three themes emerge across the surveyed literature:

**Agent workflows as first-class data structures.** The field is converging on the recognition that agent execution traces should be formalized as graphs (typically DAGs) amenable to analysis, optimization, and versioning. This is evident in LangGraph's state graphs, Halo's query plan DAGs, GCC's version-controlled trajectories, ScienceClaw's provenance DAGs, and DPIMPR's merged reasoning graphs. The practical benefits are clear: once workflows are graphs, classical algorithms for optimization (Halo), fault tolerance (Self-Healing Router), and dynamic adaptation (AgentConductor) become applicable.

**Separation of coordination from reasoning.** The Self-Healing Router's 93% reduction in LLM control-plane calls by treating routing as graph search rather than reasoning, Gas Town's patrol-cycle coordination, and POLARIS's type-checked plan synthesis all point toward a design principle: coordination logic should be handled by deterministic systems, reserving LLM reasoning for tasks that genuinely require it.

**Uncertainty as an active signal, not a passive metric.** The progression from single-step entropy (Kuhn et al., 2023) through multi-step propagation (SAUP, UProp) to multi-agent decomposition (MATU, CoE, DPIMPR) reflects a shift toward using uncertainty to *drive* system behavior rather than merely *report* on it.

### 5.2 Open Problems and Contribution Opportunities

Several significant gaps remain:

1. **Formal statistical guarantees for multi-agent UQ.** ConU provides conformal coverage guarantees for single-model settings, but no existing work extends conformal prediction to multi-agent systems. Wrapping DAG-based confidence scores in a conformal framework would provide provable coverage bounds---a substantial theoretical and practical contribution.

2. **Unified tensor-DAG representations.** MATU's tensor decomposition excels at isolating uncertainty sources (cascading, role-dependent, topology-sensitive) but operates on embedding representations. DPIMPR's DAG structure excels at capturing reasoning topology and accumulating evidence. These approaches are complementary: a tensor representation of a DAG-structured reasoning process could combine the strengths of both.

3. **Version-controlled uncertainty provenance.** GCC and R-LAM demonstrate the value of version-controlled agent state. Extending this to uncertainty estimates---tracking how confidence evolves across reasoning versions, debate rounds, or workflow revisions---would enable a new class of analysis: uncertainty archaeology, understanding not just what the system is uncertain about *now* but how that uncertainty developed and was resolved.

4. **Topology-adaptive UQ.** AgentConductor and MacNet show that agent topology affects performance. MATU identifies topology-sensitive uncertainty. The natural synthesis is UQ-driven topology adaptation: using real-time uncertainty signals to dynamically restructure the agent communication graph, spawning additional agents where uncertainty is high and pruning redundant interactions where confidence is established.

5. **Benchmarks for multi-agent UQ.** The field lacks standardized benchmarks. Current evaluations use diverse, incomparable setups (MATU on unspecified traces, DiscoUQ on QA, DPIMPR on debate). A shared benchmark suite---analogous to SWE-Bench for coding agents---would accelerate progress and enable meaningful comparison across methods.

---

## References

Bholani, N. (2026). Graph-based self-healing tool routing for cost-efficient LLM agents. *arXiv:2603.01548*.

Cemri, M., Pan, M. Z., Yang, S., Agrawal, L. A., Chopra, B., Tiwari, R., Keutzer, K., Parameswaran, A., Klein, D., Ramchandran, K., Zaharia, M., Gonzalez, J. E., & Stoica, I. (2025). Why do multi-agent LLM systems fail? *arXiv:2503.13657*.

Chang, et al. (2026). CascadeDebate: Multi-agent deliberation for cost-aware LLM cascades. *arXiv:2604.12262*.

Chase, H. et al. (2024--2026). LangGraph: Graph-based agent orchestration platform. LangChain. *blog.langchain.dev*.

Chen, T., Yao, H., Chen, J., Papalexakis, E. E., & Wei, H. (2026). Every response counts: Quantifying uncertainty of LLM-based multi-agent systems through tensor decomposition. *ACL 2026*. arXiv:2604.08708.

Duan, et al. (2025). Uncertainty propagation in multi-step LLM reasoning. *arXiv*.

Ehtesham, A., Singh, A., Gupta, G. K., & Kumar, S. (2025). A survey of agent interoperability protocols: MCP, ACP, A2A, and ANP. *arXiv:2505.02279*.

Feng, S., Ding, W., Liu, A., Wang, Z., Shi, W., Wang, Y., Shen, Z., Han, X., Lang, H., Lee, C.-Y., Pfister, T., Choi, Y., & Tsvetkov, Y. (2025). When one LLM drools, multi-LLM collaboration rules. *arXiv:2502.04506*.

Guo, T., Chen, X., Wang, Y., Chang, R., Pei, S., Chawla, N. V., Wiest, O., & Zhang, X. (2024). Large language model based multi-agents: A survey of progress and challenges. *arXiv:2402.01680*.

Jiang, B. (2026). DiscoUQ: Structured disagreement analysis for uncertainty quantification in LLM agent ensembles. *arXiv:2603.20975*.

Kuhn, L. et al. (2023). Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. *ICLR 2023*.

Li, J., Zhang, Q., Yu, Y., Fu, Q., & Ye, D. (2024). More agents is all you need. *arXiv:2402.05120*.

Moslemi, Z., Koneru, K., Lee, Y.-T., Kumar, S., & Radhakrishnan, R. (2026). POLARIS: Typed planning and governed execution for agentic AI in back-office automation. *arXiv:2601.11816*.

Park, J. W. & Cho, K. (2025). Efficient semantic uncertainty quantification in language models via diversity-steered sampling. *NeurIPS 2025*. arXiv:2510.21310.

Qian, C., Xie, Z., Wang, Y., Liu, W., Zhu, K., Xia, H., Dang, Y., Du, Z., Chen, W., Yang, C., Liu, Z., & Sun, M. (2024). Scaling large language model-based multi-agent collaboration. *ICLR 2025*. arXiv:2406.07155.

Shen, J., Wadlom, N., & Lu, Y. (2025--2026). Halo: Batch query processing and optimization for agentic workflows. *arXiv:2509.02121*.

Sun, K., Wu, J., Li, J., Guo, M., Che, X., & Huang, J. (2026). CoE: Collaborative entropy for uncertainty quantification in agentic multi-LLM systems. *ICLR 2026 Workshop*. arXiv:2603.28360.

Sureshkumar, S. (2026). R-LAM: Reproducibility-constrained large action models for scientific workflow automation. *arXiv:2601.09749*.

Wang, F. Y., Marom, L., Pal, S., Luu, R. K., Lu, W., Berkovich, J. A., & Buehler, M. J. (2026). ScienceClaw + Infinite: Autonomous agents coordinating distributed discovery through emergent artifact exchange. *arXiv:2603.14312*.

Wang, J., Wang, J., Athiwaratkun, B., Zhang, C., & Zou, J. (2024). Mixture-of-agents enhances large language model capabilities. *arXiv:2406.04692*.

Wang, S., Lu, R., Yang, Z., Wang, Y., Zhang, Y., Xu, L., Xu, Q., Yin, G., Chen, C., & Guan, X. (2026). AgentConductor: Topology evolution for multi-agent competition-level code generation. *arXiv:2602.17100*.

Wang, Z., Duan, J., Cheng, L., Zhang, Y., Wang, Q., Shi, X., Xu, K., Shen, H., & Zhu, X. (2024). ConU: Conformal uncertainty in large language models with correctness coverage guarantees. *EMNLP 2024 Findings*. arXiv:2407.00499.

Wu, J., Hu, M., Zhu, J., Pan, J., Liu, Y., Xu, M., & Jin, Y. (2025--2026). Git context controller: Manage the context of LLM-based agents like Git. *arXiv:2508.00031*.

Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., Jiang, L., Zhang, X., Zhang, S., Liu, J., Awadallah, A. H., White, R. W., Burger, D., & Wang, C. (2023). AutoGen: Enabling next-gen LLM applications via multi-agent conversation. *arXiv:2308.08155*.

Ye, Z., Melo, L. C., Kaddar, Y., Blunsom, P., Staton, S., & Gal, Y. (2025). Uncertainty-aware step-wise verification with generative reward models. *arXiv:2502.11250*.

Zhang, et al. (2026). From passive metric to active signal: The evolving role of uncertainty quantification in large language models. *arXiv:2601.15690*.

Zhang, T., Shi, H., Wang, Y., Wang, H., He, X., Li, Z., Chen, H., Han, L., Xu, K., Zhang, H., Metaxas, D., & Wang, H. (2025--2026). TokUR: Token-level uncertainty estimation for large language model reasoning. *ICLR 2026*. arXiv:2505.11737.

Zhao, Q., Zhao, X., Liu, Y., Cheng, W., Sun, Y., Oishi, M., Osaki, T., Matsuda, K., Yao, H., & Chen, H. (2024). SAUP: Situation awareness uncertainty propagation on LLM agent. *arXiv:2412.01033*.

Ziletti & D'Ambrosi (2026). Disentangling ambiguity from instability in large language models. *arXiv:2602.12015*.
