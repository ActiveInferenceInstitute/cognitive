---
title: Reward Prediction Error
description: The difference between received and expected reward that drives learning in reinforcement learning, its association with phasic dopamine signalling, and how Active Inference reinterprets dopamine as encoding precision.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - reinforcement_learning
  - dopamine
  - prediction_error
semantic_relations:
  - type: relates
    links:
      - "[[temporal_difference_learning]]"
      - "[[dopamine_system]]"
      - "[[reward_processing]]"
      - "[[prediction_error]]"
      - "[[precision_weighting]]"
      - "[[reinforcement_learning]]"
---

# Reward Prediction Error

## Overview
A reward prediction error (RPE) is the discrepancy between the reward received and the reward that was expected. In temporal-difference learning it takes the form
$$\delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t),$$
where $V$ is the learned value of a state and $\gamma$ is a discount factor. A positive $\delta_t$ indicates that things went better than predicted, and the value estimate is raised. A negative $\delta_t$ lowers it. The idea originates in the Rescorla-Wagner model of conditioning, in which learning is driven by the surprise of an outcome and not by its mere occurrence.

## Neural evidence
Recordings from midbrain dopamine neurons in monkeys showed phasic responses that matched the form of the temporal-difference error. An unexpected reward elicits a burst. After a cue has come to predict the reward, the burst moves to the cue and the fully predicted reward produces little response. An expected reward that is omitted produces a dip in firing below baseline. Schultz, Dayan and Montague proposed in 1997 that phasic dopamine signals this error and thereby teaches value to downstream structures such as the striatum. Subsequent work, including optogenetic manipulation, supports a causal role in learning, while also showing that dopamine activity relates to other variables, including novelty, salience, movement and uncertainty. The pure RPE account is therefore influential but not uncontested.

## Relation to Active Inference
Active Inference offers a different reading of the same data. It has no reward function; goals are encoded as prior preferences, and policies are chosen by minimising expected free energy. Phasic dopamine is proposed to report the precision of beliefs about policies, which controls how confidently action is selected, and not a reward error as such. In one set of simulations of this framework (FitzGerald et al., 2015), simulated dopamine responses to outcomes diminish and those to cues grow over the course of learning, suggesting that the classical dopamine findings can be accommodated by a precision account. Whether the neural data discriminate between the two accounts is still being tested, and the exact mapping depends on modelling choices.

## Related concepts
- [[temporal_difference_learning]] — the algorithm that defines the error signal
- [[dopamine_system]] — the neuromodulatory system associated with the signal
- [[reward_processing]] — neural mechanisms of reward evaluation
- [[prediction_error]] — the general class of discrepancy signals
- [[precision_weighting]] — the Active Inference quantity attributed to dopamine

## References
- Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. Science, 275, 1593-1599.
- Friston, K., Schwartenbeck, P., FitzGerald, T., Moutoussis, M., Behrens, T., & Dolan, R. J. (2014). The anatomy of choice: dopamine and decision-making. Philosophical Transactions of the Royal Society B, 369, 20130481.
- FitzGerald, T. H. B., Dolan, R. J., & Friston, K. (2015). Dopamine, reward learning, and active inference. Frontiers in Computational Neuroscience, 9, 136.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
