---
title: Conservation Laws
description: Statements that a quantity stays constant over time, linked to symmetries by Noether's theorem and to continuity equations, with discussion of how they relate to dissipative free-energy dynamics.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - physics
  - symmetry
  - dynamical_systems
semantic_relations:
  - type: relates
    links:
      - "[[hamiltonian_mechanics]]"
      - "[[lagrangian_mechanics]]"
      - "[[thermodynamics]]"
      - "[[non_equilibrium_steady_state]]"
---

# Conservation Laws

## Overview
A conservation law says that some function of the state, a conserved quantity, has zero time derivative along every trajectory. Energy, momentum, angular momentum, charge and probability are the standard examples. Conservation laws constrain a system's motion to a lower-dimensional surface in state space and often make it solvable.

## Formal statements
- **Noether's theorem.** Each continuous symmetry of an action gives a conserved quantity: invariance under time translation gives energy, under spatial translation gives momentum, under rotation gives angular momentum ([[lagrangian_mechanics]]).
- **Hamiltonian systems.** A function $F$ is conserved if its Poisson bracket with the Hamiltonian vanishes, $\{F,H\}=0$. The Hamiltonian itself is conserved when it has no explicit time dependence, and Liouville's theorem conserves phase-space volume ([[hamiltonian_mechanics]]).
- **Continuity equation.** A conserved density satisfies $\partial_t\rho+\nabla\cdot J=0$. The Fokker-Planck equation is of this form with probability current $J$, so total probability is conserved ([[fokker_planck]]).
- **Thermodynamics.** Energy is conserved (first law) while entropy production is non-negative (second law) for closed systems, so not every quantity of interest is conserved ([[thermodynamics]]).

## Role in Active Inference
Free energy is not a conserved quantity. Under the free-energy principle a system at non-equilibrium steady state has dynamics that decompose into a dissipative gradient part and a solenoidal part, $f=(Q-\Gamma)\nabla\ln p^*$ with $Q$ antisymmetric and $\Gamma$ symmetric positive semi-definite. Because $Q$ is antisymmetric, $\nabla\ln p^*\cdot Q\nabla\ln p^*=0$: the solenoidal flow moves states along level sets of surprise without changing it, and is the part that resembles conservative Hamiltonian motion. The gradient part dissipates and drives surprise down. Conservation of probability is built into the density dynamics. The more speculative step is to ask whether there are physical conserved quantities, via Noether symmetries, for the variational action of the path-integral formulation; this is a research question and not an established result. The conserved structure that is certain is probabilistic: normalisation, and in the discrete setting columns of $A$ and $B$ summing to one.

## Related concepts
- [[hamiltonian_mechanics]] — conservative dynamics with Poisson-bracket invariants
- [[lagrangian_mechanics]] — Noether's theorem for action principles
- [[non_equilibrium_steady_state]] — setting of the solenoidal and gradient decomposition
- [[entropy_production]] — the non-conserved counterpart quantifying dissipation
- [[symplectic_geometry]] — geometric structure preserved by Hamiltonian flow
- [[integrable_systems]] — systems with enough conserved quantities to be solved by quadrature
