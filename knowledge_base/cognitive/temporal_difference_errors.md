---
title: Temporal Difference Errors
description: The prediction error signal in temporal difference learning, its definition, its correspondence to phasic dopamine activity, and how Active Inference reinterprets dopamine in terms of precision.
type: concept
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
      - "[[reward_prediction_error]]"
      - "[[dopamine_system]]"
      - "[[reinforcement_learning]]"
      - "[[prediction_error]]"
---

# Temporal Difference Errors

## Overview
A temporal difference (TD) error is the discrepancy between successive estimates of expected future reward. It is the learning signal in temporal difference methods, which update a value estimate using the next estimate rather than waiting for the final outcome, a technique called bootstrapping. Sutton (1988) introduced the family of TD methods for learning to predict.

## Definition
Let $V(s)$ be the estimated value of state $s$, $r_{t+1}$ the reward received after leaving state $s_t$, and $\gamma\in[0,1]$ a discount factor. The one-step TD error is
$$\delta_t = r_{t+1}+\gamma V(s_{t+1})-V(s_t).$$
It is positive when outcomes are better than predicted and negative when worse. The value update is $V(s_t)\leftarrow V(s_t)+\alpha\,\delta_t$ with learning rate $\alpha$. When predictions are correct on average, the expected TD error is zero. Related algorithms, such as SARSA and Q-learning, apply the same error to state-action values, and TD($\lambda$) blends errors across multiple time steps using eligibility traces. Sutton and Barto (2018) give a standard treatment.

## Dopamine
Recordings from midbrain dopamine neurons in monkeys showed phasic responses to unexpected rewards that shifted to reward-predicting cues with learning and dipped when an expected reward was omitted. Schultz, Dayan and Montague (1997) argued that this pattern matches the TD error. The proposal has been highly influential and is supported by many results, including optogenetic manipulations, but it has also been challenged: dopamine neurons are heterogeneous, with some signalling other variables such as novelty, salience or movement, and the signal may carry richer distributional information than a single scalar error. See [[dopamine_system]] and [[reward_prediction_error]].

## Relation to Active Inference
In Active Inference, behaviour is driven by minimising expected free energy rather than maximising reward, and reward appears only as prior preference over outcomes. Friston and colleagues propose that phasic dopamine reports the precision of beliefs about policies, with changes in expected policy value producing responses that resemble TD errors under suitable assumptions (Friston et al., 2014). This reinterpretation reproduces some dopamine findings in simulation, but it remains a competing hypothesis, and whether the data discriminate between precision and value-error accounts is still under investigation.

## Related concepts
- [[temporal_difference_learning]] — the family of algorithms that use the error
- [[reward_prediction_error]] — the neural and behavioural reward-error signal
- [[dopamine_system]] — neuromodulatory system linked to the signal
- [[reinforcement_learning]] — framework in which TD errors are used
- [[prediction_error]] — the broader class of error signals
- [[precision_weighting]] — the Active Inference reading of dopamine

## References
- Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. Machine Learning, 3, 9-44.
- Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. Science, 275, 1593-1599.
- Friston, K., Schwartenbeck, P., FitzGerald, T., Moutoussis, M., Behrens, T., & Dolan, R. J. (2014). The anatomy of choice: Dopamine and decision-making. Philosophical Transactions of the Royal Society B, 369, 20130481.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). MIT Press.
