---
title: Policy Gradient
description: Reinforcement learning methods that optimise a parameterised policy by gradient ascent on expected return, with the policy gradient theorem, variance reduction, and contrast to policy selection by expected free energy.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - reinforcement_learning
  - optimization
  - policies
semantic_relations:
  - type: relates
    links:
      - "[[reinforcement_learning]]"
      - "[[policy_search_methods]]"
      - "[[value_function]]"
      - "[[policy_selection]]"
      - "[[reward_function]]"
      - "[[gradient_descent]]"
---

# Policy Gradient

## Overview
Policy gradient methods learn a policy directly. The policy $\pi_\theta(a\mid s)$ is a differentiable function with parameters $\theta$, and the objective is the expected return $J(\theta)=\mathbb{E}_{\pi_\theta}\big[\sum_t\gamma^t r_t\big]$. Parameters are improved by gradient ascent, $\theta\leftarrow\theta+\eta\,\nabla_\theta J(\theta)$, using estimates of the gradient from sampled experience. Because the policy is a probability distribution, the approach handles stochastic and continuous action spaces naturally.

## The gradient estimator
The policy gradient theorem expresses the gradient without differentiating the unknown state distribution or dynamics:
$$\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\Big[\sum_t\nabla_\theta\ln\pi_\theta(a_t\mid s_t)\,Q^{\pi_\theta}(s_t,a_t)\Big],$$
with the expectation over the discounted state-visitation distribution. The simplest estimator, REINFORCE, replaces $Q$ by the sampled return from time $t$. It is unbiased but has high variance. Subtracting a baseline $b(s_t)$ that does not depend on the action leaves the expectation unchanged and can reduce variance. Choosing the state value function as baseline leads to actor-critic methods, in which a learned critic estimates values and an actor updates the policy. Variants such as natural gradients and trust-region updates constrain each step so that the policy changes in a stable way.

## Properties
Policy gradient methods usually converge to a local optimum, not a global one. They are on-policy in their basic form, which makes them sample-inefficient, and they depend on a scalar reward signal that must be specified.

## Relation to Active Inference
The two frameworks answer different questions. A policy gradient learner tunes parameters of a reward-maximising policy. In Active Inference, no reward function is primitive. Policies are scored by expected free energy, which combines preference satisfaction with information gain, and a distribution over policies is obtained by a softmax over the scores, with habits entering as a learned prior. Several papers analyse formal correspondences between control as inference, reinforcement learning and Active Inference, and the exact relation depends on how preferences are encoded. The connection is therefore a comparison between alternative formulations, not a derivation of one from the other.

## Related concepts
- [[reinforcement_learning]] — the setting in which policy gradients are used
- [[policy_search_methods]] — direct search in policy space
- [[value_function]] — the critic in actor-critic methods
- [[policy_selection]] — the Active Inference counterpart for choosing policies
- [[pragmatic_value]] — the preference-based term playing the role of reward

## References
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8, 229-256.
- Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. Advances in Neural Information Processing Systems, 12, 1057-1063.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
