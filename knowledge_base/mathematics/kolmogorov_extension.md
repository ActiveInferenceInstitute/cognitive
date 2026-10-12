---
title: Kolmogorov Extension Theorem
description: The theorem guaranteeing that a consistent family of finite-dimensional distributions defines a stochastic process on an infinite index set, with its role and limits in constructing path-space measures.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - measure_theory
  - stochastic_processes
  - probability_theory
semantic_relations:
  - type: relates
    links:
      - "[[measure_theory]]"
      - "[[measure_space]]"
      - "[[stochastic_processes]]"
      - "[[wiener_process]]"
      - "[[path_integral_theory]]"
wikidata: Q3526996
---

# Kolmogorov Extension Theorem

## Overview
The Kolmogorov extension theorem, also called the consistency theorem, states when a collection of finite-dimensional probability distributions can be realised as the finite-dimensional marginals of a single probability measure on an infinite product space. It is the basic existence result behind stochastic processes indexed by time.

## Statement
Let $T$ be an index set and let each $E_t$ be a Polish space (a separable, completely metrisable space) with its Borel $\sigma$-algebra. Suppose that for every finite subset $F=\{t_1,\dots,t_n\}\subset T$ there is a probability measure $\mu_F$ on $\prod_{t\in F}E_t$, and that these are consistent: the measures are compatible with permuting coordinates, and the marginal of $\mu_F$ onto a smaller subset $G\subset F$ equals $\mu_G$. Then there is a unique probability measure $\mu$ on the product space $\prod_{t\in T}E_t$ with the product $\sigma$-algebra whose marginals are the $\mu_F$.

## Use and limits
The theorem is used to define processes such as Gaussian processes, where specifying means and covariances for every finite set of times gives consistent marginals. It also gives a process version of the Wiener process, but an important limitation applies: the product $\sigma$-algebra contains only events that depend on countably many coordinates. The set of continuous paths is not such an event, so the theorem alone does not show that the process has continuous sample paths. A further result, the Kolmogorov continuity theorem, supplies a modification with continuous paths under a moment condition on increments (see [[wiener_process]]).

Infinite-dimensional spaces of paths have no analogue of Lebesgue measure, so integrals "over all paths" cannot be defined by a flat measure. Rigorous path measures such as the Wiener measure are built through the extension theorem, with the Gaussian weight carried by the measure itself. Feynman's oscillatory path integral does not correspond to a measure of this kind.

## Relation to Active Inference
The theorem is background mathematics, not part of Active Inference. It matters for formulations of the free energy principle and path-integral methods that treat probability over trajectories: such expressions are well defined as measures on path space only when constructed in this way, as in measure-theoretic treatments of Active Inference.

## Related concepts
- [[measure_theory]] — the framework in which the theorem is stated
- [[measure_space]] — the structure the theorem produces on path space
- [[stochastic_processes]] — the objects whose existence it secures
- [[wiener_process]] — a process constructed using it
- [[gaussian_processes]] — processes specified by consistent finite-dimensional Gaussians
- [[measure_theoretic_active_inference]] — measure-theoretic treatment of the formalism

## References
- Billingsley, P. (1995). Probability and Measure (3rd ed.). Wiley.
- Kallenberg, O. (2002). Foundations of Modern Probability (2nd ed.). Springer.
- Kolmogorov, A. N. (1933). Grundbegriffe der Wahrscheinlichkeitsrechnung. Springer.
