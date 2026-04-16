# Bead-State Uncertainty: Version-Controlled Epistemic State Tracking for Multi-Agent Workflows

## Description
We propose leveraging version-control semantics -- specifically the "bead" abstraction from Gas Town (open -> hooked -> closed state machine over versioned work units) and the database-level versioning from Dolt -- to create a novel uncertainty tracking mechanism for multi-agent workflows. The key insight: when agents reason over multiple steps with branching and merging (as in real-world agentic workflows, not just debate), the version history of the reasoning state contains rich uncertainty information that is currently discarded. Specifically, we define three new uncertainty signals from version-control metadata: (1) Branch divergence entropy -- the entropy of the distribution of answers across all active branches at any point in the workflow, computed from the bead DAG's branch structure. (2) Merge conflict rate -- when two branches of reasoning merge, the frequency and severity of "conflicts" (contradictory intermediate conclusions) directly measures epistemic uncertainty. (3) Revert frequency -- how often an agent's conclusion is later overridden by a downstream agent, tracked through Dolt's version history. We combine these into a "version-control uncertainty" (VCU) score and show that it provides a complementary signal to DPIMPR's semantic confidence: DPIMPR measures confidence in the reasoning content, while VCU measures confidence in the reasoning process. The combination should be particularly powerful for complex, multi-step agentic workflows (code generation, research synthesis) where the reasoning DAG is deep and wide.

## Gap Filled
No existing UQ method exploits version-control metadata. This is surprising because version control is ubiquitous in agent workflows (GCC, R-LAM, ScienceClaw all validate VC semantics for agent state). Halo formalized agent workflows as DAGs for optimization but did not extract uncertainty signals from them. Gas Town's bead system already tracks state transitions and versions; extracting UQ from this metadata is essentially free -- it requires no additional LLM calls, only analysis of existing workflow logs. This fills the gap between "UQ as an additional computation" and "UQ as a byproduct of the workflow's own bookkeeping."

## Experiments
1. Instrument Gas Town's bead system to log all state transitions, branch/merge events, and Dolt version metadata.
2. Run multi-agent workflows on three task types: (a) multi-hop QA (HotpotQA), (b) code generation (HumanEval+), (c) research synthesis (a custom task requiring agents to aggregate findings from multiple sources).
3. Compute the three VCU signals (branch divergence entropy, merge conflict rate, revert frequency) from the workflow logs.
4. Train a predictor using VCU signals alone; compare AUROC to DPIMPR and to a combined VCU + DPIMPR model.
5. Analyze: which VCU signal carries the most information? Does it vary by task type?
6. Computational cost analysis: VCU adds zero additional LLM calls (it is purely a metadata analysis). Quantify the cost savings compared to methods that require additional sampling.
7. Extension: use VCU in real-time to trigger early stopping (high VCU = stop and ask for human input) or compute escalation (high VCU = allocate more agent rounds).

## Risks
- The bead system and Dolt are specific to Gas Town; the VCU signals may not generalize to other multi-agent frameworks, limiting the paper's impact.
- Branch divergence and merge conflicts are meaningful only in workflows with actual branching -- for simple linear pipelines or debate protocols, these signals are trivially zero.
- The "free" argument only holds if the workflow already uses version control; adding VC to a framework that lacks it is non-trivial.
- Merge conflict rate depends heavily on how "conflict" is defined in the semantic domain (text similarity threshold? LLM-judged contradiction?), and this definition may dominate the method's performance, making VCU essentially a relabeling of semantic similarity.
- The paper may read as a systems contribution rather than a machine learning contribution, which could be a fit issue at ML venues.
