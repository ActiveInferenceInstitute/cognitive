---
title: Weak Convergence of Measures
description: Convergence of probability measures through integrals of bounded continuous functions, with the portmanteau theorem, Prokhorov's theorem, the central limit theorem and Donsker's invariance principle.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - probability
  - measure_theory
  - limit_theorems
semantic_relations:
  - type: relates
    links:
      - "[[measure_theory]]"
      - "[[probability_theory]]"
      - "[[wiener_process]]"
      - "[[random_walk]]"
      - "[[topological_spaces]]"
---

# Weak Convergence of Measures

## Overview
Weak convergence is the standard notion of convergence for probability measures on a metric space $S$. A sequence $\mu_n$ converges weakly to $\mu$, written $\mu_n\Rightarrow\mu$, if
$$\int f\,d\mu_n\to\int f\,d\mu\quad\text{for every bounded continuous }f:S\to\mathbb{R}.$$
For random variables this is convergence in distribution. It is weaker than convergence in total variation: the point masses $\delta_{1/n}$ converge weakly to $\delta_0$, although the total variation distance between $\delta_{1/n}$ and $\delta_0$ equals one for every $n$.

## Equivalent characterisations
The portmanteau theorem states that the following are equivalent for probability measures on a metric space:
- $\mu_n\Rightarrow\mu$;
- $\limsup_n\mu_n(F)\le\mu(F)$ for every closed set $F$;
- $\liminf_n\mu_n(G)\ge\mu(G)$ for every open set $G$;
- $\mu_n(A)\to\mu(A)$ for every Borel set $A$ whose boundary has $\mu$-measure zero.

On $\mathbb{R}^d$, Levy's continuity theorem adds that $\mu_n\Rightarrow\mu$ exactly when the characteristic functions converge pointwise to that of $\mu$, and, more generally, that pointwise convergence to a function continuous at the origin implies convergence to some probability measure. Prokhorov's theorem relates compactness to tightness: a family of measures on a Polish space is relatively compact in the weak topology if and only if it is tight, meaning that mass does not escape to infinity ([[sigma_algebras]]).

## Limit theorems
- **Central limit theorem.** Normalised sums of independent, identically distributed variables with finite variance converge weakly to a Gaussian distribution.
- **Donsker's invariance principle.** The linearly interpolated random walk, rescaled in time and space, converges weakly in $C[0,1]$ to Wiener measure, the law of Brownian motion ([[random_walk]], [[wiener_process]]). This justifies Brownian motion as the universal scaling limit of a broad class of random walks.
- **Continuous mapping.** If $\mu_n\Rightarrow\mu$ and $h$ is continuous, or discontinuous only on a $\mu$-null set, then the image measures converge weakly.

The convergence is the narrow convergence of measures, defined by integration against bounded continuous functions (it coincides with weak-star convergence when $S$ is compact), and the associated topology is metrisable on a separable metric space, for example by the Prokhorov or Levy-Prokhorov metric (Billingsley, 1999).

## Relation to Active Inference
The connection is mathematical background. Statements about sequences of approximate posteriors, particle approximations ([[particle_filtering]]) and the convergence of random-walk samplers to their target use weak convergence, as do rigorous derivations of diffusion approximations to discrete dynamics that appear in path-integral formulations ([[path_integral_theory]]). The Active Inference formalism itself does not require the concept in finite-state models.

## Related concepts
- [[measure_theory]] — the framework in which the convergence is defined
- [[probability_theory]] — convergence in distribution and limit theorems
- [[wiener_process]] — the limit in Donsker's theorem
- [[random_walk]] — the discrete process that converges to it
- [[topological_spaces]] — the topology of weak convergence
- [[particle_filtering]] — an approximation scheme analysed through this convergence

## References
- Billingsley, P. (1999). Convergence of Probability Measures (2nd ed.). Wiley.
- Billingsley, P. (1995). Probability and Measure (3rd ed.). Wiley.
