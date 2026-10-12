---
title: Ventral Striatum
description: "The ventral striatum, including the nucleus accumbens, as a hub for reward anticipation, prediction errors, motivation, and valuation, and its dopaminergic interpretation in Active Inference."
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - neuroanatomy
  - reward
  - basal-ganglia
semantic_relations:
  - type: relates
    links:
      - "[[reward_processing]]"
      - "[[dopamine_system]]"
      - "[[decision_making]]"
      - "[[value_learning]]"
aliases:
  - nucleus_accumbens
wikidata: Q7920296
---

# Ventral Striatum

## Overview
The ventral striatum is the ventral part of the striatum in the basal ganglia. It comprises the nucleus accumbens (core and shell), the olfactory tubercle, and the adjoining ventromedial caudate and putamen. It receives glutamatergic input from prefrontal cortex, amygdala, hippocampus, and thalamus, and dopaminergic input from the ventral tegmental area. It is often described as an interface between limbic motivation and action.

## Functions
- **Reward prediction error.** Dopamine neurons projecting to this region signal the difference between obtained and expected reward, and imaging studies find striatal BOLD responses that track prediction errors in learning tasks (Schultz, Dayan, and Montague; O'Doherty and colleagues).
- **Anticipation and motivation.** Activity rises when a reward is expected, and manipulations of accumbens dopamine affect the vigour with which animals work for rewards, which is more tied to motivation ("wanting") than to hedonic enjoyment, a distinction that remains debated.
- **Value coding.** Ventral striatal activity correlates with the subjective value of options across rewards, including money and social outcomes ([[value_computation]], [[social_reward]]).
- **Learning and Pavlovian influences.** It supports Pavlovian-instrumental interactions, where cues for reward invigorate action, and it is compared with the dorsal striatum, which is more associated with habits and action selection.
- **Clinical relevance.** Altered ventral striatal responses to reward are reported in depression, addiction, and psychosis, and are studied as candidate computational markers; these are research hypotheses, not diagnostics ([[computational_psychiatry]]).

## Role in Active Inference
In the discrete-state process theory, striatum is associated with the computation of the policy posterior, and dopamine with the precision (inverse temperature) of that posterior rather than with reward prediction error alone. Friston and colleagues proposed that phasic dopamine reports changes in the confidence in policies, which reproduces reward-prediction-error-like responses in simulations ([[dopamine_system]], [[precision_weighting]]). The mapping is a hypothesis grounded in simulation, and the standard reinforcement-learning account remains the dominant interpretation of the data.

## Related concepts
- [[reward_processing]] - the functions it supports
- [[dopamine_system]] - its main neuromodulatory input
- [[reinforcement_learning]] - the computational account of its learning signals
- [[policy_selection]] - the choice process it may support
- [[anterior_cingulate]] - a connected cortical region for effort and conflict

## References
- Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. Science.
- O'Doherty, J., Dayan, P., Schultz, J., Deichmann, R., Friston, K., & Dolan, R. J. (2004). Dissociable roles of ventral and dorsal striatum in instrumental conditioning. Science.
- Friston, K., Schwartenbeck, P., FitzGerald, T., Moutoussis, M., Behrens, T., & Dolan, R. J. (2014). The anatomy of choice: dopamine and decision-making. Philosophical Transactions of the Royal Society B.
