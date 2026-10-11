---
title: Belief Space Planning
description: Planning under partial observability by treating the agent's belief distribution as the state, covering belief updates, the belief-MDP formulation, tractability limits, and links to expected free energy.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - planning
  - pomdp
  - belief_state
semantic_relations:
  - type: relates
    links:
      - "[[pomdp_framework]]"
      - "[[belief_state]]"
      - "[[belief_updating]]"
      - "[[dynamic_programming]]"
      - "[[expected_free_energy]]"
---

# Belief Space Planning

## Overview
In a partially observable Markov decision process (POMDP) the agent cannot observe the state $s$, only observations $o$ related to it. The belief state $b(s)$, the posterior over hidden states given the history of actions and observations, is a sufficient statistic for that history. Planning in belief space treats $b$ as the state of a fully observable decision problem, the belief MDP, and selects actions as a function of $b$. This lets the planner value actions that gather information, because they change the future belief and so the quality of later decisions.

## Formulation
After taking action $a$ and observing $o$, the belief is updated by Bayes' rule:
$$
b'(s')=\frac{O(o\mid s',a)\sum_{s}T(s'\mid s,a)\,b(s)}{\sum_{s''}O(o\mid s'',a)\sum_{s}T(s''\mid s,a)\,b(s)}.
$$
The belief space is the probability simplex over $|S|$ states, continuous even when the state space is finite. For finite horizons the optimal value function is piecewise linear and convex in $b$ (Smallwood & Sondik, 1973), and it can be computed by dynamic programming over sets of vectors. Exact solution is intractable in general, and practical methods approximate it, for example with point-based value iteration, sampling-based tree search or heuristic policies. Kaelbling, Littman and Cassandra (1998) give a standard account of the framework.

## Relation to Active Inference
Discrete-time Active Inference plans in belief space. Policies are scored by expected free energy, computed from beliefs predicted forward under each policy, and the resulting policy posterior is itself a function of current beliefs. The expected information gain term in expected free energy is the way information-seeking value enters, which in classical belief-space planning arises implicitly through the value function. The two differ in their objective: Active Inference scores policies by prior preferences over outcomes plus epistemic value, not by expected cumulative reward. Recursive tree-search variants of Active Inference, called sophisticated inference, make the belief-space structure explicit.

## Related concepts
- [[pomdp_framework]] — the decision model that induces the belief MDP
- [[belief_state]] — the sufficient statistic used as the planning state
- [[belief_updating]] — the Bayesian update between planning steps
- [[dynamic_programming]] — the method used to solve the belief MDP
- [[active_sensing]] — a typical task where belief-space values matter
- [[expected_free_energy]] — the Active Inference objective over predicted beliefs

## References
- Smallwood, R. D., & Sondik, E. J. (1973). The optimal control of partially observable Markov processes over a finite horizon. Operations Research, 21, 1071-1088.
- Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). Planning and acting in partially observable stochastic domains. Artificial Intelligence, 101, 99-134.
- Friston, K., Da Costa, L., Hafner, D., Hesp, C., & Parr, T. (2021). Sophisticated inference. Neural Computation, 33, 713-763.
