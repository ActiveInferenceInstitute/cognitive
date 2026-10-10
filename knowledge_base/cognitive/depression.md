---
title: Depression
description: Computational accounts of major depression as altered learning, reward valuation, precision and interoceptive inference, presented as research hypotheses within computational psychiatry rather than clinical guidance.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - computational_psychiatry
  - mood
  - interoception
semantic_relations:
  - type: relates
    links:
      - "[[computational_psychiatry]]"
      - "[[emotional_processing]]"
      - "[[interoception]]"
      - "[[reward_processing]]"
---

# Depression

## Overview
Major depression is a psychiatric condition defined clinically by persistent low mood or loss of interest, together with changes in sleep, energy, appetite, concentration and self-evaluation. This page covers only computational hypotheses about the mechanisms involved; they are under active research, are not established causal explanations, and carry no implication for treatment.

## Computational hypotheses
Several accounts coexist and are not mutually exclusive.

**Reward and learning.** Reinforcement-learning models relate anhedonia to reduced sensitivity to reward, or to a lowered learning rate from positive outcomes. Decision-theoretic analyses (Huys, Daw and Dayan, 2015) describe how pessimistic estimates of controllability or of outcome value can make effortful action appear not worth its cost.

**Aberrant priors and belief updating.** Negative self-related priors that resist revision by disconfirming evidence are proposed to explain persistent negative bias in memory and interpretation. In predictive processing terms this is a prior with excess precision, or evidence that is given too little precision.

**Low confidence in action.** Reduced expected control over outcomes corresponds, in Active Inference language, to low precision over policies, so that no course of action is strongly favoured.

**Interoception and allostasis.** Barrett, Quigley and Hamilton (2016) proposed that depression involves inefficient allostatic regulation, with interoceptive predictions that fail to be updated by bodily signals, producing fatigue and altered affect.

## Role in Active Inference
Depression is a standard target of computational psychiatry because the generative-model parameters involved (prior preferences, policy precision, sensory precision, interoceptive predictions) map onto candidate symptoms. The link is an active research programme, not a settled derivation. Models are typically fit to behavioural tasks in small samples, and distinguishing the proposals above empirically remains difficult. Heterogeneity in the diagnosis also means a single mechanism is unlikely to account for all cases.

## Related concepts
- [[computational_psychiatry]] — the framework in which these hypotheses are developed
- [[emotional_processing]] — mood and affect regulation processes implicated in the condition
- [[interoception]] — bodily-signal inference central to the allostatic account
- [[reward_processing]] — anhedonia and reward valuation
- [[dopamine_system]] — proposed to encode reward prediction error and policy precision
- [[anxiety]] — frequently co-occurring condition with overlapping computational accounts
- [[neural_plasticity]] — plasticity changes are discussed in relation to the disorder

## References
- Huys, Q. J. M., Daw, N. D., & Dayan, P. (2015). Depression: a decision-theoretic analysis. Annual Review of Neuroscience.
- Barrett, L. F., Quigley, K. S., & Hamilton, P. (2016). An active inference theory of allostasis and interoception in depression. Philosophical Transactions of the Royal Society B.
