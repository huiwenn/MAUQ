# Uncertainty Arbitrage: Game-Theoretic Aggregation of Heterogeneous Agent Confidence

## Description
We propose modeling multi-agent uncertainty aggregation as a prediction market, where each agent "bets" on answer candidates using its confidence as a budget, and the market equilibrium price represents the aggregated system confidence. The crucial mechanism that distinguishes this from simple weighted voting: agents have heterogeneous and unknown calibration functions. Agent A's 90% confidence might correspond to 70% true accuracy, while Agent B's 60% confidence might correspond to 80% true accuracy. A prediction market naturally corrects for this because agents who are poorly calibrated will consistently lose money (confidence budget) and be automatically downweighted, while well-calibrated agents accumulate influence. Formally, we use a logarithmic market scoring rule (LMSR) adapted for the multi-agent setting: each agent's "trade" updates the market probability according to the LMSR rule, with the trade size proportional to the agent's remaining budget. The market price after all trades is the aggregated confidence. We further introduce "uncertainty arbitrage": when two agents have high confidence in opposite answers, an arbitrage opportunity exists -- the system recognizes that at least one agent is miscalibrated and flags this as high-uncertainty, even though both individual agents are confident. This directly addresses the confident-but-wrong problem: a prediction market would assign moderate prices (moderate confidence) when agent confidences are extreme but contradictory.

## Gap Filled
All existing aggregation methods (majority voting, confidence-weighted averaging, DPIMPR's composite score, CoE's collaborative entropy) use fixed aggregation rules. None adapt to the heterogeneous calibration of different agents across different question types. MATU decomposes uncertainty by agent role but does not adaptively recalibrate. Prediction markets are a well-studied mechanism for aggregating beliefs from heterogeneous sources with unknown reliability, but have not been applied to multi-agent LLM systems. The arbitrage mechanism specifically addresses a failure mode that no existing method handles: symmetric high confidence in contradictory answers.

## Experiments
1. Implement an LMSR-based prediction market for aggregating answers from 5-7 heterogeneous LLM agents (mix of model families and sizes).
2. Run the market over 2000+ questions from TriviaQA, StrategyQA, MMLU, and ARC-Challenge.
3. Compare market equilibrium prices to: (a) majority vote, (b) confidence-weighted average, (c) DPIMPR composite confidence, (d) an oracle that knows each agent's true calibration.
4. Measure calibration (ECE, Brier score), AUROC for selective prediction, and accuracy of the market's predicted answer.
5. Track agent budgets over time: do well-calibrated agents accumulate budget? Does the market naturally identify which agents are reliable for which question types?
6. Test "uncertainty arbitrage" specifically: on questions where two agents have > 90% confidence in different answers, does the market correctly assign low confidence?
7. Extension: allow agents to update their bets after seeing the market price (iterated market), testing whether this improves convergence.

## Risks
- LMSR requires continuous-valued trades, but LLM confidence scores are noisy and may not behave like rational bets -- the market microstructure may not converge.
- The budget mechanism assumes agents are evaluated across many questions sequentially; for a single question in isolation, there is no budget history and the market reduces to simple confidence-weighted averaging.
- The prediction market framework adds conceptual complexity but may not improve over a simple learned weighting of agent confidences (which is a simpler way to handle heterogeneous calibration).
- Computational overhead is minimal, but the method requires access to per-agent confidence scores, which may not be available from all API-based models.
- Reviewers may argue that learning agent-specific calibration functions offline (a la Platt scaling per agent) achieves the same effect without the prediction market machinery.
