---
title: Ergodicity
description: The property that long-run time averages of a process equal averages over its stationary ensemble, with ergodic theorems for dynamical systems and Markov chains, and its assumed role in the free energy principle.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - dynamical_systems
  - stochastic_processes
  - statistical_physics
semantic_relations:
  - type: relates
    links:
      - "[[stochastic_processes]]"
      - "[[dynamical_systems]]"
      - "[[non_equilibrium_steady_state]]"
      - "[[markov_chain_monte_carlo]]"
      - "[[free_energy_principle]]"
wikidata: Q5426803
---

# Ergodicity

## Overview
A stationary process is ergodic when the long-run average of a function along a single trajectory converges to the expectation of that function under the stationary distribution. In symbols, for a measure-preserving transformation $T$ and integrable $f$, ergodicity means $\frac1N\sum_{n=0}^{N-1}f(T^n x)\to\int f\,d\mu$ for almost every starting point $x$. Time averages and ensemble averages then agree, so one long observation of a system is representative of the whole distribution. Birkhoff (1931) proved the pointwise ergodic theorem that establishes convergence of time averages; ergodicity is the additional property that the limit is constant, which holds when the system cannot be split into separate invariant parts of positive measure.

## Markov chains and breaking of ergodicity
For a finite Markov chain, ergodicity of the chain requires irreducibility (every state reachable from every other), and convergence of the distribution to the stationary one also needs aperiodicity. For continuous state spaces, positive recurrence or similar conditions are used. Ergodicity can fail when the state space splits into disconnected regions, as in a spontaneously symmetry-broken system on observable timescales, or when the process has no stationary distribution, as with a multiplicative growth process. Peters (2019) argues that many quantities in economics are non-ergodic, so that ensemble averages misdescribe the typical individual trajectory. The relevant timescale also matters: a process can be ergodic in principle and yet not equilibrate within the time available.

## Relation to Active Inference
Ergodicity is an explicit assumption of the free energy principle. The principle supposes that a system's dynamics possess a stationary density that characterises the states it visits, and ergodicity lets that density be read as the fraction of time the system spends in each state, so that low-probability, high-surprise states are rarely visited. The assumption is a modelling choice with limits: organisms develop, age and die, which sits uneasily with stationarity, and criticisms of the principle have questioned whether the required ergodicity and steady state hold for biological systems. Defenders restrict the claim to the timescales over which the system is approximately stationary.

## Related concepts
- [[stochastic_processes]] — the class of processes for which ergodicity is defined
- [[dynamical_systems]] — the original setting of ergodic theory
- [[non_equilibrium_steady_state]] — the stationary regime assumed in the principle
- [[markov_chain_monte_carlo]] — relies on ergodic chains to estimate expectations
- [[detailed_balance]] — a separate property often assumed alongside stationarity

## References
- Birkhoff, G. D. (1931). Proof of the ergodic theorem. Proceedings of the National Academy of Sciences, 17, 656-660.
- Walters, P. (1982). An Introduction to Ergodic Theory. Springer.
- Peters, O. (2019). The ergodicity problem in economics. Nature Physics, 15, 1216-1221.
