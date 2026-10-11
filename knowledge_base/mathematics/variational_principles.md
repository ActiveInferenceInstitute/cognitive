---
title: Variational Principles
description: "Physical laws expressed as stationarity of a functional, from Fermat and least action to Gibbs free energy, and the status of the free energy principle as a variational principle."
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - variational_calculus
  - physics
  - least_action
semantic_relations:
  - type: relates
    links:
      - "[[variational_calculus]]"
      - "[[lagrangian_mechanics]]"
      - "[[hamiltonian_mechanics]]"
      - "[[free_energy_principle]]"
aliases:
  - variational_principle
---

# Variational Principles

## Overview
A variational principle selects the actual behaviour of a system as a stationary point (usually a minimum) of a functional over a class of admissible candidates. The mathematics is the [[variational_calculus]]; the physics is the observation that many laws can be stated this way.

## Examples
**Fermat's principle.** Light travels between two points along paths of stationary optical length, giving Snell's law.

**Hamilton's principle.** The motion $q(t)$ of a mechanical system makes the action $S[q]=\int_{t_0}^{t_1}L(q,\dot q,t)\,dt$ stationary among paths with fixed endpoints. Stationarity gives the Euler-Lagrange equations $\frac{d}{dt}\frac{\partial L}{\partial\dot q}=\frac{\partial L}{\partial q}$ ([[lagrangian_mechanics]]). "Least action" is a misnomer, since the stationary point need not be a minimum.

**Statistical mechanics.** The equilibrium distribution at temperature $T$ minimises the free energy $F[p]=\langle E\rangle-TS[p]$ and equals the Gibbs distribution; maximum entropy under constraints is the same principle seen from information theory.

**Noether.** A continuous symmetry of the action implies a conserved quantity, a link to [[lie_theory]].

**Features.** Variational formulations are coordinate independent, make constraints easy to incorporate (Lagrange multipliers), and suggest numerical methods (finite elements, variational integrators).

## Role in Active Inference
The free energy principle is presented as a variational principle for the dynamics of systems with a Markov blanket: internal and active states evolve to minimise a functional of beliefs about external states, the variational free energy ([[variational_free_energy]]). In the path-integral formulation, expected free energy plays the role of an action, with policies as paths ([[path_integral_free_energy]]). Two cautions apply. As in mechanics, the principle describes what trajectories are stationary under a chosen functional; it does not by itself assert that the system "computes" anything. And how the functional is chosen is where content enters, so many dynamics can be written as free-energy gradient flows. Whether this makes the principle a tautology-like constraint, as instrumentalists (see [[instrumentalism]]) suggest, or a substantive hypothesis is debated.

## Related concepts
- [[variational_calculus]] — the mathematics of stationary functionals
- [[hamiltonian_mechanics]] — the dual formulation via Legendre transform
- [[path_integral]] — sum over paths weighted by exp(-action)
- [[variational_methods]] — approximation methods derived from the same ideas
- [[instrumentalism]] — a reading of the free energy principle as modelling convention
