---
title: Individual Differences
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - cognitive
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/cognitive/computational_neuroscience]]"
      - "[[knowledge_base/cognitive/precision_weighting]]"
      - "[[knowledge_base/cognitive/model_complexity]]"
      - "[[knowledge_base/cognitive/clinical_psychology]]"
---

# Individual Differences

## Overview

Individual differences research asks why people differ in perception, learning, decision-making and affect, and how stable those differences are. In an Active Inference account each person is an agent with their own generative model and their own settings of its parameters, so variation among people becomes variation in identifiable quantities: the priors held, the precision assigned to different sources of evidence, the temperature of action selection, the rate of learning, and the complexity of the model. As a research programme, computational psychiatry has applied this to clinical conditions with suggestive but contested results. Applying it to normal variation, such as personality, is largely speculative.

## Parameters That Can Differ

**Precision of sensory evidence versus priors.** If sensory precision is high relative to prior precision, beliefs track the immediate input closely; if prior precision dominates, perception is pulled toward expectation ([[knowledge_base/cognitive/precision_weighting]], [[knowledge_base/cognitive/precision_mechanisms]]). Pellicano and Burr (2012) proposed that autistic perception reflects weak priors, and Lawson, Rees and Friston (2014) recast this as aberrant precision. Adams and colleagues (2013) offered a precision-based account of psychosis. These are hypotheses with mixed evidence and have been challenged and refined, so they are best read as structured candidate explanations.

**Policy precision and exploration.** The precision over policies controls how deterministically an agent follows its best option. Lower values give more exploratory or noisier behaviour; higher values give more rigid choice.

**Learning rate.** How fast beliefs and model parameters adapt to new evidence affects flexibility in changing environments and stability in stable ones ([[knowledge_base/cognitive/stability_plasticity]]).

**Preferences.** Different log-preference vectors over outcomes yield different goals from the same dynamics ([[knowledge_base/cognitive/preference_learning]]).

**Model complexity.** People may differ in how many hidden states and levels they represent, which links to expertise and development ([[knowledge_base/cognitive/model_complexity]]).

## Measurement Issues

The approach requires fitting parameters to each person's behaviour, usually with hierarchical Bayesian estimation so individual estimates are regularised by a group distribution. Two problems recur. Parameters can trade off against each other, so recovery tests on simulated data are needed before interpreting estimates. And estimated parameters often show only modest test-retest reliability, which limits their use as stable traits. A claim that a parameter marks a trait or a diagnosis should be backed by both checks.

## In This Repository

Relevant pages: [[knowledge_base/cognitive/clinical_psychology]], [[knowledge_base/free_energy_principle/applications/psychiatry]], [[knowledge_base/cognitive/precision_mechanisms]] and [[knowledge_base/cognitive/model_complexity]]. In code, `InferenceConfig` in `code/tools/src/models/active_inference/dispatcher.py` exposes the knobs that would define a simulated individual: `precision_init`, `temperature`, `learning_rate`, `discount_factor`, `exploration_weight` and `temporal_horizon`, and `DiscreteGenerativeModel` takes per-agent $A$, $B$, $C$, $D$ and $E$. The repository contains no routine for fitting these to behavioural data, so it supports simulating individuals but not estimating them.

## References

- Pellicano, E., Burr, D. (2012). When the world becomes "too real": a Bayesian explanation of autistic perception. *Trends in Cognitive Sciences*.
- Lawson, R. P., Rees, G., Friston, K. J. (2014). An aberrant precision account of autism. *Frontiers in Human Neuroscience*.
- Adams, R. A., Stephan, K. E., Brown, H. R., Frith, C. D., Friston, K. J. (2013). The computational anatomy of psychosis. *Frontiers in Psychiatry*.

## Related Resources

- [[knowledge_base/cognitive/active_inference]]
- [[knowledge_base/cognitive/computational_neuroscience]]
