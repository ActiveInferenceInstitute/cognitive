---
title: Probability Weighting
description: Probability weighting is the nonlinear transformation of stated probabilities in risky choice, with overweighting of small and underweighting of large probabilities, as formalized in prospect theory.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - decision-making
  - risk
  - prospect-theory
semantic_relations:
  - type: relates
    links:
      - "[[kahneman_tversky]]"
      - "[[decision_theory]]"
      - "[[utility_theory]]"
---

# Probability Weighting

## Overview
In expected-utility theory an outcome's probability enters the evaluation of a gamble linearly. Probability weighting refers to the empirical finding that people instead treat stated probabilities as decision weights $w(p)$ that depart from $p$: very small probabilities receive too much weight, and moderate to large ones too little. It is one of the two main components of prospect theory ([[kahneman_tversky]]) and helps explain why people both buy lottery tickets and insurance.

## Formalism
Tversky and Kahneman (1992) proposed the one-parameter form
$$w(p) = \frac{p^{\gamma}}{\left(p^{\gamma} + (1-p)^{\gamma}\right)^{1/\gamma}},$$
with median estimates of about $\gamma = 0.61$ for gains and $0.69$ for losses. With $\gamma = 0.61$, $w(0.01) \approx 0.055$, $w(0.10) \approx 0.19$, $w(0.50) \approx 0.42$, and $w(0.90) \approx 0.71$, so the curve has an inverse-S shape that crosses the diagonal near $p \approx 0.34$. A value $\gamma < 1$ bends the curve more strongly; $\gamma = 1$ recovers linear weighting. The decision weight is applied in a rank-dependent (cumulative) way in the 1992 version, which avoids violating stochastic dominance.

Other families include Prelec's two-parameter function, which separates curvature (discriminability) from elevation (attractiveness). Choices from description and choices from experience differ: when probabilities are learned by sampling outcomes, rare events are often underweighted, the description-experience gap (Hertwig and colleagues, 2004). Proposed explanations for the inverse-S shape include diminished sensitivity near the boundaries 0 and 1, attention to salient rare events, and noisy or regressive probability representation ([[uncertainty_estimation]]).

## Role in Active Inference
There is no standard derivation of probability weighting from Active Inference. In the framework, probabilities enter as beliefs over outcomes, and preference is carried separately by the prior over preferred outcomes, giving risk-sensitive behaviour through the risk and ambiguity terms of [[expected_free_energy]]. One hypothesis is that apparent weighting reflects imprecise or regularized probability beliefs, so extreme stated probabilities are shrunk toward intermediate values, as Bayesian estimation under noise would predict; another is that precision on policy selection distorts choice. These ideas are conjectures that fit the qualitative pattern but have not been shown to quantitatively reproduce prospect-theory parameters.

## Related concepts
- [[kahneman_tversky]] — the research programme that introduced the weighting function
- [[decision_theory]] — normative framework with linear probability
- [[utility_theory]] — valuation of outcomes paired with weights
- [[risk_assessment]] — applied evaluation of rare events
- [[decision_making]] — behaviour in which weighting is observed
- [[probability_theory]] — mathematical probability that people deviate from

## References
- Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty. *Journal of Risk and Uncertainty*, 5, 297-323.
- Hertwig, R., Barron, G., Weber, E. U., & Erev, I. (2004). Decisions from experience and the effect of rare events in risky choice. *Psychological Science*, 15, 534-539.
- Prelec, D. (1998). The probability weighting function. *Econometrica*, 66, 497-527.
