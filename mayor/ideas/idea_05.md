# Minority Report: Uncertainty Quantification via Strategic Disagreement Amplification

## Description
We propose a method that deliberately amplifies minority opinions in multi-agent systems to improve uncertainty quantification, drawing on the "wisdom of the minority" in social choice theory. The 89% consensus / 42% accuracy finding from DPIMPR reveals that majority voting is pathologically miscalibrated -- the majority is often wrong. Our method works in three stages: (1) Identification: after an initial round of multi-agent debate, identify minority-opinion agents using semantic clustering of responses. (2) Amplification: give minority agents a "soapbox" -- additional compute budget, access to retrieved evidence, or a structured adversarial role -- to elaborate their dissenting reasoning. (3) Reweighting: use the quality of the minority's elaborated argument (measured by DPIMPR-style DAG confidence over the minority reasoning subgraph) to adjust the system's overall confidence. The key mechanism is asymmetric: amplifying the minority is cheap when they are wrong (their elaborated argument will be incoherent and low-confidence) but valuable when they are right (their elaborated argument will be coherent and expose flaws in the majority's reasoning). This creates a one-sided bet: the expected cost of amplification is low but the expected information gain is high. Expected outcome: significant improvement in calibration on the "confident but wrong" subset, with modest additional compute (1-2 extra inference rounds for minority agents only).

## Gap Filled
All existing multi-agent UQ methods treat agents symmetrically -- every agent's opinion is weighted by the same type of score. DiscoUQ analyzes disagreement but does not amplify it. "More Agents Is All You Need" adds agents but weights them equally. DPIMPR merges trajectories but does not privilege dissent. This is the first method to recognize that minority opinions carry disproportionate information value in high-consensus-low-accuracy regimes. It fills the gap between "detecting uncertainty" (which DPIMPR does well) and "resolving uncertainty" (which requires actually engaging with dissenting evidence).

## Experiments
1. Implement the three-stage pipeline on top of a 5-agent debate system using GPT-4/Claude.
2. Benchmark on TriviaQA, StrategyQA, and ARC-Challenge, focusing on the subset of questions where majority consensus >= 80%.
3. Measure: (a) calibration improvement (ECE, Brier score) on the high-consensus subset, (b) overall AUROC compared to DPIMPR, (c) additional compute cost.
4. Ablation: compare amplification strategies -- (i) giving minority agents more tokens, (ii) providing retrieved documents, (iii) explicitly assigning a "devil's advocate" role prompt.
5. Control experiment: amplify random agents instead of minority agents, to verify that the minority selection is doing the work, not the additional compute.
6. Combine with DPIMPR: use minority report confidence as an additional feature in DPIMPR's composite score.

## Risks
- If the minority is almost always wrong (which is often the case by base rate), the amplification step wastes compute and the reweighting step adds noise to an otherwise good majority signal.
- The method introduces a meta-decision: how much to amplify? This is a hyperparameter that may need per-domain tuning, undermining generality.
- Minority identification via semantic clustering may fail when agents use different phrasings for the same answer, or when the minority answer is "I don't know."
- The asymmetry argument (cheap when wrong, valuable when right) assumes that confidence scores reliably distinguish coherent minority arguments from incoherent ones -- but this is exactly the calibration problem we are trying to solve.
- Reviewers may see this as "just another debate protocol variant" rather than a UQ contribution.
