---
title: Value Computation
description: "How the brain assigns subjective value to options, including the stages of value-based choice, valuation regions, and the Active Inference substitution of expected free energy for scalar value."
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - decision-making
  - valuation
  - reward
semantic_relations:
  - type: relates
    links:
      - "[[decision_making]]"
      - "[[reward_processing]]"
      - "[[value_learning]]"
      - "[[expected_free_energy]]"
aliases:
  - value_coding
  - value_comparison
  - value_integration
---

# Value Computation

## Overview
Value computation is the estimation of how good an option is for the decision maker, in a form that allows options to be compared. The value is subjective: it depends on the agent's state, goals, and beliefs, and it combines magnitude, probability, delay, and effort. The standard framework treats it as one stage in a sequence of processes that produce choice.

## Framework
Rangel, Camerer, and Montague describe value-based decision making as five processes: representing the decision problem (states, options, internal and external constraints), valuing the actions, selecting among them, evaluating the outcome, and learning from it.
- **Valuation systems.** Behaviour is governed by Pavlovian, habitual (cached model-free), and goal-directed (model-based) systems that compute values differently ([[reinforcement_learning]]).
- **Neural correlates.** Meta-analysis of human imaging implicates ventromedial prefrontal cortex and ventral striatum as regions whose activity tracks subjective value across reward types, with further regions for value-related processes such as salience ([[ventral_striatum]], [[medial_prefrontal]]).
- **Discounting and risk.** Delay and uncertainty reduce subjective value in systematic ways ([[temporal_discounting]]); prospect-theory-like distortions of probabilities and losses are common.
- **Common currency.** A hypothesis that disparate goods are converted into a shared value scale supports comparison; the evidence is substantial but does not rule out context-dependent computations.

## Role in Active Inference
Active Inference does not posit a separate reward or value function. The quantity ranking policies is the expected free energy $G(\pi)$, and the policy posterior is a softmax of its negative, $q(\pi) \propto \exp(-\gamma G(\pi))$, where $\gamma$ is a precision. In the repository's convention, $G$ is a sum of risk (divergence between predicted and preferred outcomes) and ambiguity (expected uncertainty of the observation mapping), so utility-like preference enters through the prior over outcomes and information-seeking enters through the same term ([[expected_free_energy]], [[pragmatic_value]], [[epistemic_value]]). The correspondence to reward-based valuation holds when preferences are fixed and ambiguity is zero.

## Related concepts
- [[reward_processing]] - receipt and prediction of rewards
- [[value_learning]] - learning of values from outcomes
- [[preference_learning]] - learning the preferences themselves
- [[dopamine_system]] - proposed teaching and precision signals
- [[policy_selection]] - the choice step

## References
- Rangel, A., Camerer, C., & Montague, P. R. (2008). A framework for studying the neurobiology of value-based decision making. Nature Reviews Neuroscience.
- Bartra, O., McGuire, J. T., & Kable, J. W. (2013). The valuation system: a coordinate-based meta-analysis of BOLD fMRI experiments examining neural correlates of subjective value. NeuroImage.
