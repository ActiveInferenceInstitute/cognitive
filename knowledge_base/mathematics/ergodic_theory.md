---
title: Ergodic Theory
description: The study of long-term statistical behaviour of measure-preserving dynamical systems, including ergodicity, Birkhoff's ergodic theorem and mixing, and the role of ergodic assumptions in the free energy principle.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - dynamical_systems
  - measure_theory
  - statistical_physics
semantic_relations:
  - type: relates
    links:
      - "[[ergodicity]]"
      - "[[dynamical_systems]]"
      - "[[measure_theory]]"
      - "[[non_equilibrium_steady_state]]"
      - "[[free_energy_principle]]"
---

# Ergodic Theory

## Overview
Ergodic theory studies the long-run behaviour of a transformation $T$ on a probability space $(X,\mathcal{B},\mu)$ that preserves the measure: $\mu(T^{-1}A)=\mu(A)$ for every measurable $A$. It asks when averages along a single orbit agree with averages over the whole space, which is what justifies replacing ensemble statistics by long time series. Continuous-time flows are handled in the same way.

## Ergodicity and Birkhoff's theorem
A measure-preserving $T$ is ergodic if every invariant set, one with $T^{-1}A=A$, has measure $0$ or $1$. Birkhoff's pointwise ergodic theorem states that for $f\in L^1(\mu)$ the time averages
$$\frac1n\sum_{k=0}^{n-1}f(T^k x)$$
converge for $\mu$-almost every $x$, and when $T$ is ergodic the limit is the constant $\int f\,d\mu$. Von Neumann's mean ergodic theorem gives the corresponding convergence in $L^2$. Stronger properties form a hierarchy: mixing, meaning $\mu(A\cap T^{-n}B)\to\mu(A)\mu(B)$, implies ergodicity, but not conversely. An irrational rotation of the circle is ergodic but not mixing.

## Markov chains and sampling
For a finite Markov chain, irreducibility and positive recurrence give a unique stationary distribution and a law of large numbers for time averages, which is the property that Markov chain Monte Carlo relies on. A chain that splits into disconnected parts is not ergodic, and its time averages depend on the starting state.

## Relation to Active Inference
The free energy principle in its common formulation assumes that the system's dynamics have a steady-state density and, in many treatments, that the dynamics are ergodic so that the time a system spends in a region reflects that density. This assumption lets surprise, defined through the steady-state density, be tied to long-term averages. It is a modelling assumption that is debated: biological systems change over their lifetimes and may not be ergodic on relevant timescales, and the theory is more safely read as describing systems for which the assumption is a good approximation.

## Related concepts
- [[ergodicity]] — the property itself, treated with applications in mind
- [[dynamical_systems]] — the setting for iterating or flowing a transformation
- [[measure_theory]] — the foundation for invariant measures
- [[markov_chain_monte_carlo]] — sampling that depends on ergodic averages
- [[statistical_physics]] — origin of the ergodic hypothesis

## References
- Birkhoff, G. D. (1931). Proof of the ergodic theorem. Proceedings of the National Academy of Sciences, 17, 656-660.
- Walters, P. (1982). An Introduction to Ergodic Theory. Springer.
- Petersen, K. (1983). Ergodic Theory. Cambridge University Press.
