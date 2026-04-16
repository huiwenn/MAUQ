# POLARIS: Typed Planning and Governed Execution for Agentic AI in Back-Office Automation

**Authors:** Zahra Moslemi, Keerthi Koneru, Yen-Ting Lee, Sheethal Kumar, Ramesh Radhakrishnan

**Venue:** arXiv:2601.11816, January 16, 2026

**Category:** DAG / Agent Workflows / Enterprise

## Abstract

Enterprise back-office workflows require agentic systems that are auditable, policy-aligned, and operationally predictable. POLARIS treats automation as typed plan synthesis and validated execution over LLM agents. A planner proposes structurally diverse, type-checked directed acyclic graphs (DAGs), a rubric-guided reasoning module selects a single compliant plan, and execution is guarded by validator-gated checks, a bounded repair loop, and compiled policy guardrails that block or route side effects before they occur. Achieves 0.81 micro F1 on document-centric finance tasks while producing full execution traces.

## Key Contribution

Formalizes agent plan selection as type-checked DAG synthesis with formal governance constraints, producing auditable execution traces -- bridging the gap between flexible LLM agents and enterprise compliance requirements.

## Relevance to Gas Town / MAUQ

POLARIS's type-checked DAG plans with validator gates are analogous to Gas Town's bead state machine transitions (open -> hooked -> closed) with validation at each gate. The "full execution traces" parallel Dolt's version-controlled audit trail. The bounded repair loop concept maps to Gas Town's patrol cycles for error recovery.
