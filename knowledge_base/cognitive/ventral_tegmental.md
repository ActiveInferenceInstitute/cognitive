---
title: Ventral Tegmental Area
description: The midbrain nucleus that supplies dopamine to limbic and cortical targets, its role in reward prediction error and motivation, and competing readings of dopamine in Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - dopamine
  - reward
  - midbrain
semantic_relations:
  - type: relates
    links:
      - "[[dopamine_system]]"
      - "[[reward_prediction_error]]"
      - "[[ventral_striatum]]"
      - "[[reward_processing]]"
      - "[[neuromodulation]]"
---

# Ventral Tegmental Area

## Overview
The ventral tegmental area is a group of neurons in the midbrain, medial to the substantia nigra. It contains dopaminergic, GABAergic and glutamatergic neurons. Dopaminergic projections form the mesolimbic pathway, reaching the nucleus accumbens and other parts of the [[ventral_striatum]], and the mesocortical pathway, reaching prefrontal and other cortical regions. Projections to the amygdala and hippocampus also exist. The region is a core part of the [[dopamine_system]].

## Functions
- **Reward prediction error.** Recordings from midbrain dopamine neurons in monkeys showed phasic firing that follows the pattern of a temporal difference prediction error: neurons respond to unexpected rewards, shift their response to reward-predicting cues as learning proceeds, and dip when an expected reward is omitted (Schultz, Dayan and Montague, 1997). In the reinforcement learning formalism the signal corresponds to $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, where $V$ is a value estimate and $\gamma$ a discount factor.
- **Motivation and salience.** Dopamine is also linked to incentive motivation and to attention to salient events. Some midbrain dopamine neurons respond to aversive or alerting stimuli, so the reward-only reading is incomplete (Bromberg-Martin, Matsumoto and Hikosaka, 2010).
- **Heterogeneity.** Neurons differ in their projection targets, inputs and response properties, so the area does not act as a single uniform signal.

Altered dopamine signalling in this area is discussed in relation to addiction and to other conditions. Such associations are descriptive and part of continuing research.

## Relation to Active Inference
Two readings coexist. The reinforcement learning reading casts phasic dopamine as a reward prediction error. Active Inference proposals instead cast dopamine as encoding the precision of beliefs about policies, so that dopaminergic release determines how confidently an agent commits to a course of action (FitzGerald et al., 2015). Under this proposal, responses that look like reward prediction errors appear as updates in policy precision, because outcomes better than expected raise confidence in the current policy. Whether one description is more accurate than the other for the full range of data remains under investigation, and the Active Inference version is a proposal supported by simulations of reward-learning tasks.

## Related concepts
- [[dopamine_system]] — the wider set of dopaminergic pathways
- [[reward_prediction_error]] — the signal attributed to the area
- [[temporal_difference_learning]] — the algorithm that formalises the error
- [[ventral_striatum]] — the principal target of the mesolimbic projection
- [[reward_processing]] — the broader function
- [[precision_weighting]] — the Active Inference reading of dopamine

## References
- Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. Science, 275, 1593-1599.
- Bromberg-Martin, E. S., Matsumoto, M., & Hikosaka, O. (2010). Dopamine in motivational control: rewarding, aversive, and alerting. Neuron, 68, 815-834.
- FitzGerald, T. H. B., Dolan, R. J., & Friston, K. (2015). Dopamine, reward learning, and active inference. Frontiers in Computational Neuroscience, 9, 136.
