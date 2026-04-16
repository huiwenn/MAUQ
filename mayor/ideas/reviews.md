# Research Ideas Review

## Evaluation Criteria

Each idea is scored 1-5 on five axes:
- **Novelty** (25%): Is the core idea genuinely new, or is it a mechanical combination of known techniques? Would a knowledgeable reviewer say "I haven't seen this before" or "this is X applied to Y"?
- **Viability** (20%): Can 2-3 people build and evaluate this in 3-6 months? Are the required data, compute, and infrastructure realistic?
- **Technical Contribution** (25%): Does this advance the field's understanding of multi-agent UQ, or does it merely add another method to the pile?
- **Elegance** (20%): Is the core insight clean and memorable? Could you explain it in one sentence to a colleague and have them nod?
- **Risk** (10%): How likely is it to produce positive results? (1 = very risky / likely null, 5 = safe bet)

**Weighted Total** = 0.25*Novelty + 0.20*Viability + 0.25*Technical + 0.20*Elegance + 0.10*Risk

Context for calibration: DPIMPR already achieves AUROC 0.9372. Any method that claims to improve UQ needs to either (a) beat this number convincingly, (b) provide complementary guarantees DPIMPR lacks, or (c) offer theoretical insight that scalar metrics cannot capture. "We got AUROC 0.94 instead of 0.9372" is not a paper.

---

## Individual Reviews

### Idea 1: Spectral Uncertainty Signatures

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Novelty | 3 | Spectral graph theory is well-trodden. Applying it to LLM reasoning DAGs is new, but the intellectual leap is modest. |
| Viability | 4 | Eigendecomposition is cheap, DPIMPR DAGs already exist, logistic regression is trivial. Very buildable. |
| Technical Contribution | 2 | Likely to find that spectral features are redundant with simpler graph statistics. Even if they work, "eigenvalues of this graph predict errors" is descriptive, not explanatory. |
| Elegance | 3 | The Fiedler value story is clean. But "compute eigenvalues, train classifier" is a generic recipe, not an insight. |
| Risk | 3 | Moderate. Small DAGs will produce degenerate spectra. The DPIMPR DAGs from 5-agent debates have maybe 15-30 nodes -- this is uncomfortably close to the regime where spectral methods are meaningless. |
| **Weighted Total** | **2.95** | |

**Critique:** The fatal issue is graph size. DPIMPR's reasoning DAGs are small enough that the eigenvalue spectrum is likely dominated by noise and discretization artifacts rather than meaningful structural signal. The proposal acknowledges this risk but hand-waves it. More fundamentally, if the spectral features beat DPIMPR's scalar confidence, the right follow-up question is "which simple graph statistic captures the same signal?" -- and the answer is almost certainly degree variance or average path length. This idea has a low ceiling: even if it works, the contribution is a feature engineering exercise, not a conceptual advance.

---

### Idea 2: Causal Uncertainty Decomposition via Do-Calculus

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Novelty | 4 | Genuinely new framing. Nobody has applied interventional causal inference to multi-agent LLM uncertainty. The observational-vs-interventional distinction is a real conceptual contribution. |
| Viability | 2 | Interventional analysis requires O(n) re-runs per agent per question. For 5 agents and 2000 questions, that is 10,000+ additional LLM calls. The compute cost is prohibitive for thorough evaluation. |
| Technical Contribution | 4 | If the causal/spurious decomposition actually works, it would be the first principled explanation of the false-consensus failure mode. This would genuinely advance understanding. |
| Elegance | 4 | "Agents agree because they share biases, not because they have independent evidence" is a killer one-liner. The causal framing makes this precise. |
| Risk | 2 | High risk. LLM agents from the same model family are so deeply confounded (shared training data, shared inductive biases) that causal decomposition may return "everything is confounded" for every question. Also, defining a valid do-operation on an LLM output is philosophically fraught. |
| **Weighted Total** | **3.30** | |

**Critique:** This is the most intellectually ambitious idea in the set, but it has a fundamental identifiability problem: when all agents share the same base model (or even the same model family), the "independent evidence" assumption is violated at the root. The causal graph will be trivially confounded, and the ACE estimates will be noisy and uninformative. The proposal would be much stronger if it focused on genuinely heterogeneous agents (different model families, different training data) and acknowledged that the homogeneous case is a degenerate regime. The experiments as written also do not address the elephant in the room: what counts as a valid intervention on an LLM's output?

---

### Idea 3: Conformal DAG Prediction Sets

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Novelty | 3 | Conformal prediction + DAGs is a natural extension. ConU exists, DAG conformal scores are a modest generalization. The worst-path aggregation is reasonable but not surprising. |
| Viability | 4 | Conformal prediction is well-understood, the implementation is straightforward on top of DPIMPR, and the experiments are standard calibration experiments. |
| Technical Contribution | 3 | Formal coverage guarantees are genuinely useful and missing from all competitors. But the guarantee is marginal, not conditional, which limits practical value. The conditional coverage analysis in the experiments is the interesting part, and it will probably fail. |
| Elegance | 3 | "Conformalize along DAG paths, take worst path" is clean enough. But the factorization trick (conditional exchangeability given parents) is the real contribution and it is buried. |
| Risk | 4 | Relatively safe -- conformal prediction works by construction. The question is whether the prediction sets are usefully tight, not whether coverage holds. |
| **Weighted Total** | **3.30** | |

**Critique:** This is a solid, publishable, and boring idea. The marginal coverage guarantee will hold because conformal prediction always works; the real question is whether DAG-aware conformal produces tighter sets than naive conformal on the final answer, and the answer depends entirely on whether the conditional exchangeability assumption holds. The proposal correctly identifies that worst-path aggregation may be too conservative, but does not propose a solution. The most likely outcome is that the DAG-conformal sets are only marginally tighter than Bonferroni, which makes the contribution feel like a technical exercise rather than a scientific advance. That said, being the first to put formal guarantees on multi-agent reasoning is a defensible contribution at a workshop or a short paper.

---

### Idea 4: The Topology Tax

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Novelty | 4 | The framing is genuinely original. Quantifying topology's contribution to uncertainty (not just performance) is new. The amplification/absorption decomposition is a useful conceptual tool. |
| Viability | 2 | Estimating mutual information across topologies requires enormous compute: 8+ topologies x 1000 questions x multiple runs per (question, topology) pair = tens of thousands of LLM calls. This is a compute-heavy empirical study. |
| Technical Contribution | 4 | If the Topology Tax framework produces predictive models of topology quality, this is immediately useful. It would explain MacNet, AgentConductor, and "More Agents" in a unified framework. |
| Elegance | 5 | "How much does the wiring cost you in uncertainty?" is a perfect one-liner. The tax metaphor is sticky and the decomposition into amplification vs. absorption is intuitive. |
| Risk | 2 | High risk. The topology effect may be dwarfed by agent-level variance and question-level variance, making T(G) too noisy to estimate. The linear model of T(G) vs. graph features is also a strong assumption -- the real relationship is probably nonlinear and context-dependent. |
| **Weighted Total** | **3.40** | |

**Critique:** This is the kind of idea that could either be a best-paper candidate or produce a depressing null result, with little in between. The core bet is that topology has a measurable, systematic effect on uncertainty that is separable from agent effects and question effects. This is an empirical claim, and my prior is that it is wrong -- topology effects in multi-agent LLM systems are small and dominated by question difficulty and agent quality. The compute budget required to test this claim properly is also underestimated; you need enough runs per (question, topology) pair to get stable mutual information estimates, and LLM agents are noisy. The idea would be much stronger as a controlled study with synthetic agents where you can actually manipulate the ground truth.

---

### Idea 5: Minority Report

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Novelty | 3 | Amplifying minority opinions is well-studied in ensemble learning and social choice. The asymmetric cost argument is the novel piece, but it is a relatively thin contribution on its own. |
| Viability | 5 | Dead simple to implement. You already have DPIMPR, you just add 1-2 extra inference rounds for dissenting agents. The experiments are straightforward. |
| Technical Contribution | 3 | This is more of an engineering trick than a scientific contribution. It directly addresses the false-consensus problem, which is valuable, but the mechanism is "just give the minority more compute and see if their argument improves." |
| Elegance | 4 | The asymmetric bet framing is clean and memorable: "amplification is cheap when the minority is wrong, valuable when they're right." This is a good pitch. |
| Risk | 3 | Moderate. The base rate of minority correctness matters enormously. If the minority is right 5% of the time, you are wasting compute 95% of the time. The method needs the minority to be right often enough to justify the cost, but not so often that the original consensus mechanism is completely broken. |
| **Weighted Total** | **3.45** | |

**Critique:** The asymmetry argument is appealing but circular: the claim is that we can cheaply determine whether the minority's elaborated argument is good by measuring its DPIMPR confidence. But if DPIMPR confidence were reliable enough to do this, we would not need the minority amplification in the first place -- we would already know the majority was wrong. The proposal needs to explain why DPIMPR confidence is reliable *for evaluating the minority's argument* when it was unreliable *for evaluating the majority's argument*. The most likely outcome is a modest calibration improvement (5-10% ECE reduction) at a non-trivial compute cost, which is a perfectly fine systems paper but not an exciting research contribution.

---

### Idea 6: Information-Geometric Uncertainty

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Novelty | 4 | Information geometry applied to multi-agent reasoning trajectories is genuinely new. The trajectory-as-curve-on-manifold framing opens up a rich mathematical toolkit. |
| Viability | 2 | Requires token-level probability distributions at every reasoning step -- immediately excludes closed-source models. Fisher information matrices over vocabulary-sized distributions are enormous. Geodesic computation on high-dimensional simplices is expensive. |
| Technical Contribution | 3 | If curvature and trajectory length carry signal beyond entropy, this is a real contribution. But the risk of "this is just entropy in a fancy hat" is high -- the trace of the Fisher matrix is the entropy, and most geometric features may collapse to entropy variants. |
| Elegance | 4 | "Reasoning trajectories trace curves on a statistical manifold, and the curvature tells you about uncertainty" is beautiful. But beauty is not a substitute for utility. |
| Risk | 2 | High risk. The geometric features likely reduce to trajectory entropy statistics after dimensionality reduction, and the computational cost of showing this is large. Also, requiring open-source models limits the experimental scope. |
| **Weighted Total** | **3.05** | |

**Critique:** This idea has the highest "math for math's sake" risk in the set. The Fisher information metric on the full vocabulary simplex is computationally intractable without aggressive dimensionality reduction, and after reduction, the geometric features are likely to be monotone transforms of entropy. The proposal needs a concrete argument for when curvature carries signal that entropy does not -- the "circular reasoning" example (trajectory loops on the manifold) is suggestive but not developed. The requirement for token-level logprobs also means you cannot evaluate on the strongest closed-source models, which weakens the experimental narrative. This would be a beautiful poster paper if the curvature signal pans out, but the odds are against it.

---

### Idea 7: Self-Healing Uncertainty

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Novelty | 3 | Closed-loop UQ + topology adaptation is a natural combination. The individual pieces (DPIMPR, GNN controllers, dynamic topologies) all exist; the novelty is in wiring them together. |
| Viability | 2 | Requires training a GNN controller via RL or bandits on (DAG state, action, outcome) triples. Collecting 5000+ training triples from LLM agents is expensive. The system has many moving parts that are hard to debug. |
| Technical Contribution | 3 | Closing the loop between UQ and orchestration is a real contribution in principle. But the experiments are likely to show that the GNN controller is not much better than "add an agent when confidence is low," which deflates the contribution. |
| Elegance | 2 | This is a Rube Goldberg machine: DPIMPR + GNN + dynamic topology + three rewiring operations + convergence criteria. The number of design choices is large and each one could be questioned. |
| Risk | 2 | High risk. Feedback loops in complex systems are hard to stabilize. The GNN controller needs a lot of data to learn meaningful policies, and LLM-based data collection is expensive. |
| **Weighted Total** | **2.55** | |

**Critique:** The honest version of this idea is: "we threshold DPIMPR confidence and add agents when it is low." Everything else -- the GNN controller, the three rewiring operations, the convergence criteria -- is engineering complexity that is unlikely to justify itself in a 3-6 month research project. The ablation study will almost certainly show that a simple threshold rule captures 80%+ of the benefit, and the GNN controller adds marginal improvement at much higher complexity. The feedback loop stability concern is also real and underexplored -- what happens when the confidence estimates are systematically biased? The system could easily get stuck in degenerate rewiring cycles. Shelve this unless you have the engineering bandwidth to build it properly.

---

### Idea 8: Uncertainty Arbitrage

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Novelty | 3 | Prediction markets for belief aggregation are well-studied (Hanson, Wolfers, Zitzewitz). Applying them to LLM agents is a setting change, not a conceptual advance. |
| Viability | 4 | LMSR is well-understood and easy to implement. The experiments are straightforward comparisons. Compute cost is minimal. |
| Technical Contribution | 3 | Heterogeneous calibration correction is a real problem, but learned per-agent calibration (Platt scaling per agent) is a simpler solution. The paper needs to show the market mechanism outperforms this simple baseline, and I doubt it will. |
| Elegance | 4 | The arbitrage mechanism -- contradictory high-confidence agents trigger uncertainty -- is clean and intuitive. The market metaphor works well for exposition. |
| Risk | 3 | Moderate. The market mechanism will produce *something*, but whether it beats simpler calibration approaches is uncertain. The sequential budget mechanism breaks down for single-question evaluation. |
| **Weighted Total** | **3.30** | |

**Critique:** The killer baseline this paper must beat is embarrassingly simple: learn a per-agent, per-domain calibration function (logistic regression from agent confidence to true accuracy) and take a calibration-corrected weighted average. This achieves heterogeneous calibration correction without any market machinery. The prediction market adds conceptual complexity (LMSR, budget dynamics, equilibrium) that is only justified if it outperforms this baseline, and I expect it will not -- at least not by enough to warrant the added complexity. The "arbitrage" concept (symmetric high confidence in different answers = high uncertainty) is the genuinely useful insight and can be implemented as a simple feature without the market framework.

---

### Idea 9: Bead-State Uncertainty

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Novelty | 4 | Extracting UQ signals from version-control metadata is genuinely novel. Nobody has done this. The "UQ for free" angle is compelling. |
| Viability | 3 | Depends entirely on Gas Town infrastructure. Requires real multi-agent workflows with meaningful branching (not just debates). Instrumenting the bead system and collecting enough data for evaluation takes effort. |
| Technical Contribution | 2 | The contribution is narrow: it works for Gas Town-like systems and not for others. Branch divergence entropy is just entropy measured at a different level. Merge conflict rate is a relabeling of semantic disagreement. The "free" claim is technically true but misleading -- you need to build all the VC infrastructure first. |
| Elegance | 3 | The "free UQ from your existing bookkeeping" pitch is clean. But the three proposed signals (branch entropy, merge conflicts, revert rate) feel ad hoc rather than principled. |
| Risk | 2 | High risk. The signals may be trivially zero for most workflow patterns (linear pipelines, simple debates). The generalization beyond Gas Town is unclear. Venue fit is a real concern -- ML reviewers will see this as a systems paper. |
| **Weighted Total** | **2.85** | |

**Critique:** This idea is too tightly coupled to Gas Town's specific infrastructure to be a general ML contribution. The three VCU signals are only meaningful in workflows with genuine branching and merging, which excludes the standard multi-agent debate setups used in benchmarks. If you restrict evaluation to Gas Town workflows, reviewers will question external validity. If you implement VCU on standard debate setups, the signals will be nearly zero because debates do not have meaningful branch/merge structure. The "free" selling point is also double-edged: it implies the signals are byproducts, which makes reviewers wonder whether byproduct signals can compete with purpose-built UQ methods. This is a nice internal tool for Gas Town but not a research paper.

---

### Idea 10: Uncertainty-Aware DAG Compilation

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Novelty | 3 | The query-optimizer analogy is nice but the individual optimization passes (redundancy elimination, early termination, prioritized execution) are standard. The combination with UQ is the new part. |
| Viability | 3 | Requires a working DPIMPR with reliable intermediate-node confidence, plus a non-trivial compiler implementation. Doable in 6 months by a strong team, but tight. |
| Technical Contribution | 3 | This is primarily a systems/efficiency contribution. It uses UQ to reduce compute, which is practically valuable but does not advance UQ methodology. The contribution is "we saved 40% of LLM calls at the same accuracy" -- useful but not a UQ paper. |
| Elegance | 3 | The query-optimizer metaphor is clear. But the three optimization passes feel like a laundry list rather than a unified insight. |
| Risk | 3 | Moderate. Early termination will definitely save compute. The question is whether the full compiler outperforms simple early termination alone, and the answer is probably "not by much." |
| **Weighted Total** | **3.00** | |

**Critique:** The value proposition here is efficiency, not UQ, and the paper should be honest about that. The most useful optimization pass (early termination) is also the simplest, and the more complex passes (redundancy elimination via semantic similarity, bandit-based branch prioritization) add marginal benefit at high implementation cost. The comparison to Halo is apples-to-oranges since they optimize different objectives. The strongest version of this paper is a narrow one: "uncertainty-guided early termination in multi-agent reasoning saves X% of compute at negligible accuracy loss." Everything else is scope creep.

---

### Idea 11: Schur Complement Meets Multi-Agent DAGs

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Novelty | 3 | Extending CLUES' Schur complement from single-model to multi-agent DAGs. The extension is natural and the author's own risk section anticipates the "incremental" critique. |
| Viability | 3 | Fitting a Gaussian graphical model across questions is doable but requires careful statistical work. The precision matrix estimation needs enough data, and each question gives one DAG observation. |
| Technical Contribution | 3 | Agent elimination and step elimination are useful analytical tools. The hierarchical uncertainty pyramid is a nice conceptual contribution. But operationalizing it (which level of the hierarchy to use, when) is unresolved. |
| Elegance | 3 | The math is clean -- Schur complements are a well-understood tool. But "apply known mathematical tool to new data structure" is not the same as an insight. |
| Risk | 2 | The Gaussian assumption is the Achilles heel. LLM belief states are emphatically not Gaussian, and the CLT justification requires sample sizes that are likely unavailable. If the Gaussian model is a bad fit, everything downstream is unreliable. |
| **Weighted Total** | **2.85** | |

**Critique:** The Gaussian assumption is fatal. LLM output distributions are multimodal, heavy-tailed, and highly non-stationary across question types. Invoking the CLT for "aggregated token probabilities" requires aggregating over far more tokens than a typical reasoning step produces. If you relax the Gaussian assumption, you lose the exact Schur complement and need approximate methods, at which point you are doing generic approximate Bayesian inference on a graphical model, which is not novel. The agent-elimination and step-elimination operations are conceptually appealing but the paper cannot deliver them without the Gaussian assumption holding. This idea would be stronger if it started from a model where the Gaussian assumption is defensible (e.g., embedding-space representations instead of token probabilities).

---

## Rankings

| Rank | Idea | Weighted Score |
|------|------|---------------|
| 1 | **Idea 5: Minority Report** | 3.45 |
| 2 | **Idea 4: The Topology Tax** | 3.40 |
| 3 | **Idea 2: Causal Uncertainty Decomposition** | 3.30 |
| 4 | **Idea 3: Conformal DAG Prediction Sets** | 3.30 |
| 5 | **Idea 8: Uncertainty Arbitrage** | 3.30 |
| 6 | **Idea 6: Information-Geometric Uncertainty** | 3.05 |
| 7 | **Idea 10: Uncertainty-Aware DAG Compilation** | 3.00 |
| 8 | **Idea 1: Spectral Uncertainty Signatures** | 2.95 |
| 9 | **Idea 9: Bead-State Uncertainty** | 2.85 |
| 10 | **Idea 11: Schur Complement Meets Multi-Agent DAGs** | 2.85 |
| 11 | **Idea 7: Self-Healing Uncertainty** | 2.55 |

---

## Top 3 Recommendations

### 1. Idea 5: Minority Report (Score: 3.45)

**Why:** This is the most buildable high-impact idea in the set. It directly attacks the central empirical finding (89% consensus / 42% accuracy) with a mechanistically clear intervention. The asymmetric cost argument, while imperfect, gives the paper a clean narrative. Implementation is straightforward on top of existing DPIMPR infrastructure. The risk of null results is moderate but manageable -- even if the minority is rarely right, the calibration improvement from recognizing *when* the minority has a strong argument is valuable.

**Caveats to address before starting:** The circularity in the confidence argument (using DPIMPR to evaluate minority arguments when DPIMPR already failed to flag the majority) must be resolved. One approach: use a *different* UQ signal for minority evaluation than for initial consensus detection -- e.g., evaluate the minority's argument against retrieved external evidence rather than using DPIMPR confidence alone.

**Suggested sequencing:** Start with the simplest amplification strategy (give minority agents more tokens) and the control experiment (amplify random agents). If the minority-vs-random gap is significant, then explore fancier amplification strategies. Do not build the full three-stage pipeline until you have validated the basic signal.

### 2. Idea 4: The Topology Tax (Score: 3.40)

**Why:** This has the highest elegance score in the set and fills a genuine explanatory gap. The field has accumulated empirical observations about topology effects (MacNet, AgentConductor, "More Agents") without a unifying framework. If the Topology Tax framework produces even a coarse predictive model of topology quality, it becomes the citation target for all future topology work. The risk is high, but the upside is commensurately high -- this is a "swing for the fences" idea.

**Caveats to address before starting:** The compute cost must be reduced. Instead of estimating mutual information across topologies (which requires an impractical number of LLM calls), consider a cheaper proxy: use the *variance* of DPIMPR confidence across topologies as a measure of topology sensitivity, and compute the Topology Tax as the fraction of total confidence variance attributable to topology (ANOVA-style decomposition). This is estimable with far fewer samples.

**Suggested sequencing:** Start with a small-scale pilot (3 agents, 4 topologies, 200 questions) to check whether topology variance is detectable at all. If yes, scale up. If the topology effect is negligible at this scale, pivot or shelve.

### 3. Idea 2: Causal Uncertainty Decomposition (Score: 3.30)

**Why:** This is the most intellectually substantive idea in the set. The observational-vs-interventional distinction is the right way to think about the false-consensus problem, and the causal decomposition into genuine-evidence vs. shared-bias is exactly the conceptual tool the field needs. If done well, this reframes the conversation about multi-agent UQ from "how much do agents agree" to "why do agents agree," which is a major conceptual upgrade.

**Caveats to address before starting:** (1) Use genuinely heterogeneous agents (different model families, different sizes) to maximize the signal-to-noise ratio of the causal decomposition. Homogeneous agents will produce degenerate results. (2) Define the intervention carefully: instead of random replacement, use a structured intervention (e.g., replace agent output with the output from a different question, or with a deliberately wrong but plausible answer). (3) Budget compute aggressively -- interventional analysis is expensive, so start with a small number of questions and agents and only scale if the signal is there.

**Suggested sequencing:** Run a proof-of-concept on 100 questions with 3 heterogeneous agents (one from each major model family). Compute ACE for each edge. If the causal/spurious decomposition shows any separation between correct and incorrect consensus cases, invest in the full evaluation. If not, the confounding is too severe and the idea should be shelved.

---

## Ideas to Deprioritize

### Shelve: Idea 7 (Self-Healing Uncertainty)
Too many moving parts for a research project. The GNN controller will underperform simple heuristics, and the feedback loop stability issues are unsolved. If you want closed-loop UQ-to-orchestration, start with Idea 5 (Minority Report), which achieves a similar goal with far less machinery.

### Shelve: Idea 9 (Bead-State Uncertainty)
Too coupled to Gas Town infrastructure. The signals are meaningful only for branching workflows that most benchmarks do not exercise. This is a useful internal tool, not a research paper. Build it as a feature, not a publication.

### Shelve: Idea 11 (Schur Complement)
The Gaussian assumption is a dealbreaker for the LLM setting. The "exact elimination" selling point only holds under Gaussianity, and without it, this is generic approximate Bayesian inference on a graphical model. Revisit only if you find a defensible distributional assumption.

### Deprioritize: Idea 1 (Spectral Uncertainty Signatures)
The DAGs are too small for spectral methods to shine. Run a quick pilot (1 day of work): compute eigenvalues for 100 DPIMPR DAGs, check if the Fiedler value correlates with correctness. If r > 0.3, promote to active development. If not, drop it permanently.

### Deprioritize: Idea 6 (Information-Geometric Uncertainty)
Beautiful math, likely null result. The computational requirements (token-level logprobs, Fisher information matrices, geodesic computation) are heavy, and the most probable outcome is that geometric features reduce to entropy statistics. Pursue only if someone on the team has deep information geometry expertise and can identify a concrete setting where curvature beats entropy.

### Deprioritize: Idea 10 (DAG Compilation)
The useful part (early termination) is trivial to implement and does not need a paper. The complex parts (redundancy elimination, bandit-based prioritization) add marginal value. Build early termination as a DPIMPR feature, not as a standalone contribution.

### Conditional: Ideas 3 and 8 (Conformal Prediction, Uncertainty Arbitrage)
Both are solid but unexciting. They are safe publication targets if you need a paper, but they will not move the field. Pursue one of them as a secondary project alongside a top-3 idea, choosing based on which reviewer audience you are targeting (theory-leaning venues favor Idea 3; applied-leaning venues favor Idea 8).
