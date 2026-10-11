---
title: Noether Theorem
description: The result that every continuous symmetry of a system's action corresponds to a conserved quantity, with its Lagrangian statement, standard examples and the analogy it offers for flows in free energy dynamics.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - symmetry
  - conservation_laws
  - mechanics
semantic_relations:
  - type: relates
    links:
      - "[[conservation_laws]]"
      - "[[lagrangian_mechanics]]"
      - "[[hamiltonian_mechanics]]"
      - "[[euler_lagrange_equations]]"
      - "[[variational_calculus]]"
      - "[[symplectic_geometry]]"
---

# Noether Theorem

## Overview
Emmy Noether's first theorem, proved in 1918, states that each continuous one-parameter symmetry of the action functional of a physical system gives rise to a quantity that is conserved along solutions of the equations of motion. It connects invariance principles to conservation laws and is a foundation of modern theoretical physics.

## Statement
Consider a Lagrangian $L(q,\dot q,t)$ with action $S=\int L\,dt$. Suppose the transformation $q_i\to q_i+\epsilon\,\delta q_i(q,\dot q,t)$ leaves $L$ invariant, or changes it only by a total time derivative. Then, on solutions of the Euler-Lagrange equations, the quantity
$$Q=\sum_i\frac{\partial L}{\partial \dot q_i}\,\delta q_i$$
is constant in time (with a correction term when $L$ changes by a total derivative). The same result holds for field theories, where the conserved quantity becomes a current.

## Standard correspondences
- Invariance under time translation implies conservation of energy.
- Invariance under spatial translation implies conservation of linear momentum.
- Invariance under rotation implies conservation of angular momentum.
- Invariance under a phase rotation of a complex field implies conservation of charge or particle number.

The theorem requires that the system be described by an action and that the symmetry be continuous. Dissipative systems that lack a Lagrangian, or that break time-translation symmetry through friction or driving, do not conserve the corresponding quantities. A second theorem treats gauge symmetries that depend on arbitrary functions.

## Relation to Active Inference
There is no direct derivation of Active Inference from Noether's theorem. The connection is an analogy that arises because formulations of the free energy principle borrow the machinery of variational mechanics, including principles of least action and Hamiltonian forms. The flow of a system at a non-equilibrium steady state can be split into a dissipative, gradient part and a solenoidal part that circulates along contours of the steady-state density without changing it. The solenoidal part is loosely comparable to a conservative flow. Whether symmetries of the free energy functional yield conserved quantities for inference dynamics is a topic of exploratory work and not established doctrine.

## Related concepts
- [[conservation_laws]] — the quantities Noether's theorem delivers
- [[lagrangian_mechanics]] — the framework in which the theorem is stated
- [[euler_lagrange_equations]] — the equations of motion whose solutions conserve the charge
- [[hamiltonian_mechanics]] — the equivalent phase-space formulation
- [[variational_calculus]] — the mathematics of stationary action

## References
- Noether, E. (1918). Invariante Variationsprobleme. Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen, Mathematisch-Physikalische Klasse, 235-257.
- Goldstein, H., Poole, C., & Safko, J. (2002). Classical Mechanics (3rd ed.). Addison-Wesley.
- Arnold, V. I. (1989). Mathematical Methods of Classical Mechanics (2nd ed.). Springer.
