---
title: Exact Inference
description: Algorithms that compute marginals and posteriors of a probabilistic model without approximation, including enumeration, variable elimination and junction trees, with their complexity limits and use in small Active Inference models.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - inference_algorithms
  - graphical_models
  - complexity
semantic_relations:
  - type: relates
    links:
      - "[[inference_algorithms]]"
      - "[[belief_propagation]]"
      - "[[probabilistic_graphical_models]]"
      - "[[variational_inference]]"
---

# Exact Inference

## Overview
Exact inference computes quantities such as a marginal $p(x_i\mid e)$ or the evidence $p(e)$ exactly, up to numerical precision, rather than bounding or sampling them. For a graphical model with $n$ discrete variables of $k$ states each, the naive route sums over $k^n$ joint configurations. Exploiting the factorisation of the joint reduces the cost when the graph structure permits.

## Algorithms
- **Enumeration.** Sum out hidden variables from the joint. Exponential in $n$, adequate for tiny models.
- **Variable elimination.** Sum out variables one at a time, pushing sums inside products of factors. The cost is exponential in the largest intermediate factor, which depends on the elimination order.
- **Belief propagation (sum-product) on trees.** Two sweeps of messages yield all marginals exactly in time linear in the number of edges ([[belief_propagation]], [[message_passing]]). On graphs with loops it is an approximation.
- **Junction tree algorithm.** Moralise and triangulate the graph, build a tree of cliques, and pass messages on it (Lauritzen and Spiegelhalter, 1988). Complexity is exponential in the **treewidth** of the graph.
- **Dynamic programming for chains.** The forward-backward algorithm for hidden Markov models and the Viterbi algorithm for most probable paths.
- **Conjugate analytic updates.** In exponential-family models with conjugate priors the posterior is closed-form ([[exponential_family]]), and the Kalman filter is the exact solution for linear-Gaussian state-space models.

General exact inference in discrete Bayesian networks is NP-hard (Cooper, 1990), so exactness is limited to low-treewidth structure; approximation by variational methods, sampling or loopy propagation takes over elsewhere.

## Role in Active Inference
Discrete-time Active Inference models are usually small enough that the posterior over a single hidden-state factor can be computed exactly by Bayes' rule, and the belief update across time is the forward-backward computation on a chain. Standard implementations nevertheless use variational message passing with a mean-field factorisation across factors, so they are not exact for multi-factor models: the marginal approximation is a deliberate trade for cost, and the resulting free energy is a bound on, not equal to, surprise. An exact calculation is therefore useful as a reference when checking an implementation on a toy problem. For continuous-state models, exactness is available only in the linear-Gaussian case.

## Related concepts
- [[variational_inference]] — the approximate alternative when exact is intractable
- [[belief_propagation]] — exact on trees, approximate on loopy graphs
- [[factor_graphs]] — representation on which elimination and messages are defined
- [[bayesian_networks]] — the directed models where these algorithms are standard
- [[computational_complexity]] — hardness results that bound what exact methods can do
- [[belief_state]] — the quantity computed exactly in a small POMDP

## References
- Cooper, G. F. (1990). The computational complexity of probabilistic inference using Bayesian belief networks. *Artificial Intelligence*, 42, 393-405.
- Koller, D. and Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press.
- Lauritzen, S. L. and Spiegelhalter, D. J. (1988). Local computations with probabilities on graphical structures and their application to expert systems. *Journal of the Royal Statistical Society B*, 50, 157-224.
