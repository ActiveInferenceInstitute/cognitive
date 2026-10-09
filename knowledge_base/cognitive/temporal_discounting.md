---
title: Temporal Discounting
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - decision_making
  - reward
  - time_preference
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[decision_making]]"
      - "[[reinforcement_learning]]"
---

# Temporal Discounting

## Overview
Temporal (or delay) discounting is the reduction in the subjective value of a reward as the delay to receiving it increases. Most people and animals prefer a smaller reward sooner to a larger reward later over some range of delays. How steeply an individual discounts is a stable trait-like measure that correlates with impulsivity-related behavior, though the strength of these correlations varies across studies.

## Mechanism
Common functional forms for the value $V$ of an amount $A$ at delay $D$:

- **Exponential**: $V = A\,e^{-kD}$ (or $A\,\gamma^{D}$). The discount rate is constant, so preferences are time-consistent: if A-now is preferred to B-later, adding a common delay to both leaves the ordering unchanged (Samuelson, 1937).
- **Hyperbolic**: $V = A/(1+kD)$. The discount rate falls with delay, which yields preference reversals: someone who prefers a larger-later reward from a distance may switch to the smaller-sooner reward as it becomes imminent. Hyperbolic forms fit human and animal choice data better than exponential ones in many studies (Ainslie, 1975; Mazur, 1987).
- **Quasi-hyperbolic (beta-delta)**: $V = A\,\beta\,\delta^{D}$ for $D>0$, an extra immediate-bias term $\beta<1$ (Laibson, 1997).

Neurally, valuation and delay-related signals involve the ventral striatum, ventromedial prefrontal cortex, and dopaminergic projections; whether a single valuation system or competing impulsive and patient systems explains discounting is debated. Measured rates depend on the task: real versus hypothetical rewards, magnitude (larger amounts are discounted less), and domain (money, food, health). In reinforcement learning, the discount factor $\gamma$ plays the same role in $G_t=\sum_k \gamma^k r_{t+k}$, and is normally taken as exponential for mathematical convenience.

## Role in Active Inference
Standard active inference has no explicit discount parameter. Policies are evaluated by summing [[expected_free_energy]] over a finite planning horizon, with prior preferences expressed as log-probabilities over outcomes. Time preference can then arise through other mechanisms: finite horizon truncation, greater uncertainty about distant outcomes (which lowers their pragmatic value relative to near ones), or explicit preferences over time. Whether these reproduce observed hyperbolic discounting and preference reversal has not been settled, and there is no consensus that discounting emerges without an added parameter. This is an open modeling question rather than an established result, and some formulations simply introduce a discount-like term for fits to behavior. Individual differences in discounting have been related to dopaminergic function through precision hypotheses (see [[dopamine_system]]), but that link is speculative.

## Related concepts
- [[decision_making]] — intertemporal choice is a core paradigm
- [[reinforcement_learning]] — the discount factor in value estimation
- [[reward_processing]] — valuation of delayed outcomes
- [[policy_selection]] — policies are compared over a horizon
- [[pragmatic_value]] — expected preference satisfaction that a discounting analysis would modify
- [[dopamine_system]] — implicated in impulsivity and delay valuation

## References
- Samuelson, P. A. (1937). A note on measurement of utility. *Review of Economic Studies*.
- Ainslie, G. (1975). Specious reward: a behavioral theory of impulsiveness and impulse control. *Psychological Bulletin*.
- Laibson, D. (1997). Golden eggs and hyperbolic discounting. *Quarterly Journal of Economics*.
