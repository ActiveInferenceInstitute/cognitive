---
title: Detailed Balance
description: The condition that a stationary Markov process has zero net probability flow between every pair of states, its consequences for reversibility and sampling, and why non-equilibrium steady states violate it.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - stochastic_processes
  - thermodynamics
  - reversibility
semantic_relations:
  - type: relates
    links:
      - "[[non_equilibrium_steady_state]]"
      - "[[entropy_production]]"
      - "[[metropolis_hastings]]"
      - "[[markov_chain_monte_carlo]]"
      - "[[free_energy_principle]]"
---

# Detailed Balance

## Overview
For a Markov chain with transition probabilities $P_{ij}$ and a distribution $\pi$, detailed balance requires
$$
\pi_i P_{ij}=\pi_j P_{ji}\quad\text{for all } i,j.
$$
Each pair of states exchanges probability at equal rates in both directions, so there is no net current at stationarity. Summing over $i$ shows that any $\pi$ satisfying detailed balance is stationary. A chain satisfying it is called reversible, because in the stationary regime its statistics look the same run forwards or backwards in time. For continuous-time processes the same condition applies to transition rates.

## Properties
- **Kolmogorov's criterion**: a stationary chain is reversible exactly when the product of transition probabilities around every closed cycle is the same in both directions, which can be checked without knowing $\pi$.
- **Thermodynamic meaning**: equilibrium systems coupled to a heat bath obey detailed balance with the Boltzmann distribution, and Onsager's reciprocal relations (Onsager, 1931) rest on microscopic reversibility. At steady state with detailed balance the entropy production rate is zero.
- **Spectral structure**: the transition operator of a reversible chain is self-adjoint in a weighted inner product, giving real eigenvalues and clean mixing-time bounds (Levin & Peres, 2017).
- **Sampling**: Metropolis-Hastings is constructed by choosing acceptance probabilities so that detailed balance holds with respect to the target distribution, which guarantees that the target is stationary.

Detailed balance is sufficient, not necessary, for stationarity; chains with circulating probability currents can have a stationary distribution without satisfying it.

## Relation to Active Inference
In the free energy principle, living systems are modelled as maintaining a non-equilibrium steady state (NESS), where the dynamics generally break detailed balance. The flow of a NESS density is decomposed into a dissipative, gradient-like part that ascends the log density and a solenoidal part that circulates along its level sets. Detailed balance corresponds to the case with no solenoidal flow, where the process reduces to a gradient flow on surprise. The decomposition is a standard result for Langevin dynamics with a stationary density; that real organisms are well described by such a NESS is a modelling assumption of the framework, not a derived fact.

## Related concepts
- [[non_equilibrium_steady_state]] — steady states with persistent currents, violating detailed balance
- [[entropy_production]] — vanishes when detailed balance holds
- [[metropolis_hastings]] — sampler built to satisfy the condition
- [[markov_chain_monte_carlo]] — class of methods relying on reversibility
- [[fluctuation_theorems]] — constraints on irreversibility away from balance
- [[langevin_dynamics]] — dynamics whose stationary flows are analysed with it

## References
- Onsager, L. (1931). Reciprocal relations in irreversible processes. I. Physical Review, 37, 405-426.
- Kelly, F. P. (1979). Reversibility and Stochastic Networks. Wiley.
- Levin, D. A., & Peres, Y. (2017). Markov Chains and Mixing Times (2nd ed.). American Mathematical Society.
