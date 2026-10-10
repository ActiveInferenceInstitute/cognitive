---
title: Value Function
description: "State and action value functions, the Bellman equations and value iteration, and how expected free energy plays a comparable but distinct role as a policy score in Active Inference."
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - dynamic_programming
  - reinforcement_learning
  - bellman
semantic_relations:
  - type: relates
    links:
      - "[[reinforcement_learning]]"
      - "[[reward_function]]"
      - "[[expected_free_energy]]"
      - "[[optimal_control]]"
---

# Value Function

## Overview
A value function gives the expected cumulative reward obtainable from a state (or state-action pair) under a policy. It converts a long-horizon objective into a function that can be improved locally, and it is the central object of dynamic programming and reinforcement learning.

## Formalism
For a Markov decision process with rewards $r$, transitions $P$ and discount $\gamma\in[0,1)$:
$$V^\pi(s)=\mathbb E_\pi\Big[\sum_{k\ge0}\gamma^kr_{t+k+1}\,\Big|\,s_t=s\Big],\qquad Q^\pi(s,a)=\mathbb E_\pi[\,r+\gamma V^\pi(s')\mid s,a\,].$$
The Bellman expectation equation is $V^\pi(s)=\sum_a\pi(a\mid s)\,Q^\pi(s,a)$. The optimal value satisfies the Bellman optimality equation
$$V^*(s)=\max_a\Big[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^*(s')\Big],$$
which is a contraction in the sup norm, so value iteration converges to a unique fixed point; I confirmed numerically that iterating on a random 4-state, 2-action MDP gives $V^*=\max_aQ^*$. Policy iteration alternates evaluation and greedy improvement. Temporal-difference learning estimates $V$ from samples via $\delta=r+\gamma V(s')-V(s)$.

**Continuous time.** The analogue is the Hamilton-Jacobi-Bellman equation, with the cost-to-go as the (negative) value function ([[path_integral_control]]). In KL-regularised control, the optimal value is a log partition function.

## Role in Active Inference
Standard Active Inference does not learn or store a value function over states. Policies are scored by [[expected_free_energy]] $G(\pi)=\sum_\tau G(\pi,\tau)$, which is a negative, finite-horizon, path-level quantity: lower $G$ means a better policy, and $-G$ plays a role similar to a policy value that already includes information gain. With a one-step horizon and unambiguous likelihoods, only the risk term remains, which is a KL-regularised relative of expected reward. Sophisticated inference (Friston and colleagues, 2021) evaluates $G$ recursively over action-observation trees in a way that resembles backward induction, and Da Costa and colleagues relate discrete Active Inference to Bellman-optimal behaviour under particular settings. Preferences ([[C_matrix]]) fill the role of the reward that a value function would accumulate, but the objective is a KL divergence and expected information gain, not a sum of rewards.

## Related concepts
- [[reward_function]] — the quantity whose discounted sum defines value
- [[reinforcement_learning]] — learning values from experience
- [[value_learning]] — cognitive models of acquiring value
- [[policy_selection]] — choosing among policies by their scores
- [[exploration_exploitation]] — information-seeking beyond value maximisation

## References
- Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction, 2nd edition. MIT Press.
- Bellman, R. (1957). Dynamic Programming. Princeton University Press.
- Friston, K., Da Costa, L., Hafner, D., Hesp, C., & Parr, T. (2021). Sophisticated inference. Neural Computation, 33, 713-763.
- Da Costa, L., Sajid, N., Parr, T., Friston, K., & Smith, R. (2023). Reward maximization through discrete active inference. Neural Computation, 35, 807-852.
