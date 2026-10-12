---
title: Non-Ergodic Systems
description: Systems whose long-time averages along a single trajectory differ from ensemble averages, including broken ergodicity, multiplicative growth, and what this implies for the ergodic assumptions used with the free energy principle.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - ergodic_theory
  - stochastic_processes
  - statistical_physics
semantic_relations:
  - type: relates
    links:
      - "[[ergodicity]]"
      - "[[ergodic_theory]]"
      - "[[broken_ergodicity]]"
      - "[[non_equilibrium_steady_state]]"
      - "[[free_energy_principle]]"
---

# Non-Ergodic Systems

## Overview
A stationary process is ergodic if the time average of an observable along almost every single trajectory converges to its ensemble average. For a measure-preserving transformation $T$ on a probability space $(X,\mu)$, Birkhoff's ergodic theorem states that
$$\frac1N\sum_{n=0}^{N-1}f(T^nx)\to \bigl(\mathbb{E}[\,f\mid\mathcal{I}\,]\bigr)(x)$$
for almost every $x$, where $\mathcal{I}$ is the $\sigma$-algebra of invariant sets. The limit is the constant $\int f\,d\mu$ exactly when the system is ergodic, meaning that every invariant set has measure zero or one. A system is non-ergodic when this fails.

## Forms of non-ergodicity
- **Decomposition into invariant components.** The state space splits into regions that the dynamics never leave. Each trajectory samples only its own region, so time averages depend on the initial condition. A ferromagnet below its critical temperature in the thermodynamic limit is the standard example: the magnetisation stays in one of two symmetry-broken states. Palmer (1982) called this broken ergodicity and also described its realisation in glassy systems, where the accessible regions are separated by barriers that are not crossed on the observation timescale.
- **Multiplicative dynamics.** If a quantity grows by random multiplicative factors, the ensemble mean can grow while the typical trajectory decays. A wealth process multiplied each round by 1.5 or 0.6 with equal probability has expected factor $1.05$ and a time-average growth factor $\sqrt{0.9}<1$. Peters (2019) argues that this difference between ensemble and time averages is overlooked in economic modelling.
- **Finite observation time.** Finite data can behave non-ergodically even when the infinite-time limit is ergodic, if mixing is slow.

The practical consequence is that expectations computed over a population need not describe an individual's long-run behaviour, and that sampling a single long run may not characterise a distribution.

## Role in Active Inference
The free energy principle describes systems that possess a non-equilibrium steady-state density and are often assumed to be ergodic, so that the occupancy of states over time equals the density used to define surprise and Markov blankets (Friston, 2013). The assumption is substantive. Where a system has several metastable regions, or its parameters drift, the single-density picture applies only within a region or over limited times. Discussions of the scope of the principle often turn on this point, and it remains an open question how far non-ergodic systems can be treated within the same formalism.

## Related concepts
- [[ergodicity]] — the property whose failure defines the topic
- [[ergodic_theory]] — the mathematical theory of averages for dynamics
- [[broken_ergodicity]] — confinement to part of the state space
- [[non_equilibrium_steady_state]] — the density assumed by the free energy principle
- [[phase_transitions]] — symmetry breaking as a source of non-ergodicity
- [[stochastic_processes]] — framework for time versus ensemble averages

## References
- Walters, P. (1982). An Introduction to Ergodic Theory. Springer.
- Palmer, R. G. (1982). Broken ergodicity. Advances in Physics, 31, 669-735.
- Peters, O. (2019). The ergodicity problem in economics. Nature Physics, 15, 1216-1221.
- Friston, K. (2013). Life as we know it. Journal of the Royal Society Interface, 10, 20130475.
