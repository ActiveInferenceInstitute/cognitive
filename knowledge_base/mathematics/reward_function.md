---
title: Reward Function
description: "The scalar reward signal of reinforcement learning, its role in return and Bellman equations, and how Active Inference replaces it with log-probability preferences over outcomes."
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - reinforcement_learning
  - preferences
  - decision_theory
semantic_relations:
  - type: relates
    links:
      - "[[reinforcement_learning]]"
      - "[[utility_theory]]"
      - "[[pragmatic_value]]"
      - "[[value_function]]"
---

# Reward Function

## Overview
In reinforcement learning (RL) a reward function $r(s,a,s')$ assigns a scalar to each transition, and the agent maximises expected discounted return $G_t=\sum_{k\ge0}\gamma^k r_{t+k+1}$. The reward hypothesis states that all goals can be expressed as maximising expected cumulative scalar reward. Reward is an external specification of the task, supplied by the designer.

## Properties
**Invariances.** Adding a constant to $r$ shifts all returns equally and leaves the optimal policy unchanged in continuing discounted tasks (in episodic tasks of variable length it can change the optimum); scaling by a positive factor also preserves the optimum. Potential-based shaping, $r'=r+\gamma\Phi(s')-\Phi(s)$, preserves optimal policies (Ng, Harada and Russell, 1999).

**Design difficulties.** Sparse rewards give weak learning signals; misspecified rewards are exploited in unintended ways; inverse RL tries to recover $r$ from behaviour but the solution is not unique.

**Relation to utility.** Reward is an experienced or assigned quantity; utility is a preference ordering represented numerically ([[utility_theory]]). In the von Neumann-Morgenstern setting, utility is unique up to positive affine transformation.

## Role in Active Inference
Active Inference has no reward function as a primitive. Preferences are encoded as a prior over outcomes $P(o\mid C)$, held as log-probabilities in the vector $C$ ([[C_matrix]]). The pragmatic part of [[expected_free_energy]] is then $\mathbb E_Q[\ln P(o\mid C)]$, so $\ln P(o\mid C)$ plays the role of reward, up to normalisation. Given any reward $r(o)$ one can set $\ln P(o\mid C)=r(o)/\lambda+\text{const}$, with $\lambda$ a temperature, and recover reward-maximising preferences. Two differences matter. First, the agent minimises divergence from preferred outcomes (risk), not maximises expected reward, so it behaves differently when the preferred distribution has spread. Second, an information-gain term is added, so exploration is not an extra heuristic. Friston and colleagues argued that reward is thereby a derived quantity, a position some find convincing and others treat as a re-description; Da Costa and colleagues showed that discrete Active Inference can recover reward-maximising behaviour under suitable conditions.

## Related concepts
- [[reinforcement_learning]] — where the reward hypothesis originates
- [[reward_processing]] — neural and behavioural treatments of reward
- [[pragmatic_value]] — the preference-satisfaction term in EFE
- [[preference_learning]] — learning $C$ from experience
- [[decision_theory]] — formal treatment of utility

## References
- Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction, 2nd edition. MIT Press.
- Ng, A. Y., Harada, D., Russell, S. (1999). Policy invariance under reward transformations: theory and application to reward shaping. ICML.
- Friston, K., Daunizeau, J., Kiebel, S. J. (2009). Reinforcement learning or active inference? PLoS ONE.
