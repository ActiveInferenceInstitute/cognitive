---
title: Temporal Difference Learning
description: Reinforcement learning methods that update value estimates from the difference between successive predictions, with the TD error, eligibility traces, SARSA and Q-learning, and the comparison with Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - reinforcement_learning
  - value_learning
  - prediction_error
semantic_relations:
  - type: relates
    links:
      - "[[reward_prediction_error]]"
      - "[[reinforcement_learning]]"
      - "[[value_function]]"
      - "[[markov_decision_process]]"
      - "[[dynamic_programming]]"
      - "[[learning_rate]]"
---

# Temporal Difference Learning

## Overview
Temporal difference (TD) learning estimates the value of states from experience by comparing successive predictions. It combines ideas from Monte Carlo methods, which learn from sampled experience without a model, and dynamic programming, which updates estimates from other estimates. The value of a state under a policy is the expected discounted return, $V^\pi(s)=\mathbb{E}_\pi\big[\sum_{k\ge0}\gamma^k r_{t+k+1}\mid s_t=s\big]$.

## The TD(0) update
After observing a transition from $s_t$ to $s_{t+1}$ with reward $r_{t+1}$, the estimate is moved toward a bootstrapped target:
$$V(s_t)\leftarrow V(s_t)+\alpha\,\delta_t,\qquad \delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t).$$
Here $\alpha$ is a learning rate and $\delta_t$ is the TD error. Updating from a one-step target allows learning online and before an episode ends, with lower variance than waiting for the full return, at the cost of some bias from the current estimate. In the tabular case, TD(0) converges to $V^\pi$ under standard step-size conditions.

## Extensions
- **TD($\lambda$)** uses eligibility traces to blend returns over many steps, interpolating between TD(0) and Monte Carlo.
- **SARSA** learns action values for the policy being followed and is on-policy.
- **Q-learning** uses the target $r+\gamma\max_{a'}Q(s',a')$, learns the optimal action values and is off-policy.
- With function approximation, TD methods may diverge in some off-policy settings, which motivates careful algorithm design.

## Neural interpretation
The TD error has been proposed as the signal carried by phasic dopamine responses, giving the algorithm a role in models of conditioning and decision making.

## Relation to Active Inference
Active Inference replaces learned value functions and rewards with prior preferences over outcomes and policy scoring by expected free energy. Friston, Daunizeau and Kiebel argued that behaviours usually modelled by value-based reinforcement learning can be reproduced by active inference without reward or value functions. The two approaches therefore compete as accounts of learning and choice. Some formulations of Active Inference can be related to the Bellman recursion that TD methods approximate, and the question of which mechanism the brain uses remains open.

## Related concepts
- [[reward_prediction_error]] — the TD error and its neural correlate
- [[reinforcement_learning]] — the field of which TD learning is a core method
- [[value_function]] — the quantity being estimated
- [[dynamic_programming]] — the model-based method that TD bootstraps from
- [[markov_decision_process]] — the formal setting of the algorithms

## References
- Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. Machine Learning, 3, 9-44.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
- Friston, K. J., Daunizeau, J., & Kiebel, S. J. (2009). Reinforcement learning or active inference? PLoS ONE, 4, e6421.
