---
title: Belief State
description: The probability distribution over hidden states given the history of actions and observations, the sufficient statistic of a partially observable decision process and the object updated in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - pomdp
  - inference
  - belief_updating
semantic_relations:
  - type: relates
    links:
      - "[[pomdp_framework]]"
      - "[[belief_updating]]"
      - "[[state_space]]"
      - "[[bayes_theorem]]"
aliases:
  - belief_states
---

# Belief State

## Overview
In a partially observable Markov decision process (POMDP) the agent never sees the state $s_t$ directly. Its belief state $b_t$ is the posterior distribution over states given everything it has seen and done: $b_t(s)=P(s_t=s\mid o_{1:t},a_{1:t-1},b_0)$. It is a sufficient statistic of the history, so decisions can depend on $b_t$ alone without loss of optimality (Åström, 1965). Planning in a POMDP is therefore a fully observed decision problem on the belief simplex.

## Update rule
After taking action $a$ and receiving observation $o$,
$$b'(s')=\frac{P(o\mid s')\sum_{s}P(s'\mid s,a)\,b(s)}{P(o\mid b,a)},$$
where the denominator normalises. The sum is a prediction step using the transition model, and the multiplication is a Bayesian correction using the likelihood. As a worked check, with two states, $b=(0.5,0.5)$ and likelihoods $P(o\mid s)=(0.8,0.3)$, the unnormalised posterior is $(0.40,0.15)$ and normalising by $0.55$ gives $b'\approx(0.727,0.273)$.

For $N$ states the belief lives on an $(N-1)$-dimensional simplex. Exact value functions over this simplex are piecewise linear and convex for finite horizons, which motivates point-based solvers. The belief space is continuous even when the state space is finite, which is why exact POMDP planning is computationally hard (Kaelbling, Littman and Cassandra, 1998).

## Role in Active Inference
Discrete-time Active Inference uses the same object but calls it an approximate posterior $Q(s_\tau)$, or sufficient statistics of it. The prior $D$ initialises it, the likelihood array $A$ and transition arrays $B$ drive prediction and correction, and variational message passing computes the update, in many treatments approximated by a mean-field factorisation across state factors. Where the update is exact, as in a single small factor, it coincides with the POMDP belief state. Where it is approximate, the Active Inference belief is a variational surrogate and does not literally satisfy the sufficiency property. Active Inference then plans over beliefs by scoring policies with expected free energy, which includes information gain about states. Belief-state planning in POMDPs captures the same epistemic value implicitly, since observations change beliefs.

## Related concepts
- [[pomdp_framework]] — the decision process in which belief states are defined
- [[belief_updating]] — the prediction-correction computation in detail
- [[belief_propagation]] — message-passing algorithm for the update on structured models
- [[exact_inference]] — when the belief can be computed without approximation
- [[expected_free_energy]] — planning criterion applied to beliefs
- [[bayes_theorem]] — the correction step

## References
- Åström, K. J. (1965). Optimal control of Markov processes with incomplete state information. *Journal of Mathematical Analysis and Applications*, 10, 174-205.
- Kaelbling, L. P., Littman, M. L. and Cassandra, A. R. (1998). Planning and acting in partially observable stochastic domains. *Artificial Intelligence*, 101, 99-134.
