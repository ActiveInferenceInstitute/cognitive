---
title: Cognitive Flexibility
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - executive_function
  - control
  - learning
semantic_relations:
  - type: relates
    links:
      - "[[executive_functions]]"
      - "[[task_switching]]"
      - "[[cognitive_control]]"
      - "[[active_inference]]"
---

# Cognitive Flexibility

## Overview
Cognitive flexibility is the ability to adapt thoughts and behaviour when goals, rules, or environmental contingencies change. It includes switching between tasks or mental sets, updating strategies after feedback, and considering alternative perspectives. It is one of the core [[executive_functions]].

## Mechanism
Several paradigms probe different aspects. In task switching, performance is slower and less accurate on trials where the task changes than where it repeats (the switch cost); see [[task_switching]]. The Wisconsin Card Sorting Test requires inferring a sorting rule from feedback and then noticing when it changes; perseverative errors indicate difficulty disengaging. In reversal learning, the rewarded option changes and the learner must switch. Flexibility is traded off against stability: a system that updates too easily is distractible, while one that updates too rarely is perseverative ([[stability_plasticity]]).

Neural work implicates [[dorsolateral_prefrontal]] cortex in maintaining and changing rules, the [[anterior_cingulate]] in signalling that the current strategy is failing, parietal cortex in reconfiguring attention, the orbitofrontal cortex and striatum in reversal learning, and noradrenergic signalling in promoting a reset of network states, a role that remains an active hypothesis. Flexibility changes across development and aging and depends on [[working_memory]] and [[inhibitory_control]].

## Role in Active Inference
In Active Inference, flexible behaviour corresponds to revising beliefs about the current context or rule. An agent with a hierarchical [[generative_model]] holds beliefs about a slowly changing "context" state that determines which lower-level mapping applies; a switch is a belief update at that level, triggered by accumulating [[prediction_error]]. How readily the update occurs depends on the agent's expectation of change (volatility) and on the precision assigned to the current rule; the hierarchical Gaussian filter of Mathys and colleagues is a well-known Bayesian model that formalises learning under such volatility. Perseveration can be described as overly precise beliefs about the old rule, or as underestimated volatility. This account is a modelling interpretation: fitting such parameters to behaviour is possible, but attributing them to specific neural mechanisms is not settled.

## Related concepts
- [[task_switching]] — the standard experimental paradigm
- [[executive_functions]] — the family of control functions
- [[inhibitory_control]] — suppressing the old response during a switch
- [[conflict_monitoring]] — detecting that the current set is failing
- [[hierarchical_inference]] — context-level belief updating
- [[meta_learning]] — adaptation of learning itself

## References
- Mathys, C., Daunizeau, J., Friston, K. J., & Stephan, K. E. (2011). A Bayesian foundation for individual learning under uncertainty. *Frontiers in Human Neuroscience*.
- Monsell, S. (2003). Task switching. *Trends in Cognitive Sciences*.
