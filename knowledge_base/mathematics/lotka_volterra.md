---
title: Lotka-Volterra Equations
description: Coupled differential equations for interacting populations, including the predator-prey model, its conserved quantity and oscillations, competitive generalisations, and links to replicator dynamics and neural sequence models.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - population_dynamics
  - ecology
  - dynamical_systems
semantic_relations:
  - type: relates
    links:
      - "[[population_dynamics]]"
      - "[[ecological_dynamics]]"
      - "[[replicator_dynamics]]"
      - "[[dynamical_systems]]"
      - "[[limit_cycles]]"
---

# Lotka-Volterra Equations

## Overview
The Lotka-Volterra equations describe how the abundances of interacting species change over time. They were proposed independently by Alfred Lotka for chemical and ecological systems and by Vito Volterra to explain oscillations in fish catches. The predator-prey form is the best-known example of a nonlinear model that produces sustained oscillations.

## Predator-prey model
With prey abundance $x$ and predator abundance $y$,
$$\dot{x}=\alpha x-\beta xy,\qquad \dot{y}=\delta xy-\gamma y,$$
where all parameters are positive. Prey grow exponentially without predators, predators decline without prey, and the interaction term $xy$ transfers biomass. There is an extinction equilibrium at the origin and a coexistence equilibrium at $(x^*,y^*)=(\gamma/\delta,\;\alpha/\beta)$. The quantity
$$V=\delta x-\gamma\ln x+\beta y-\alpha\ln y$$
is conserved along trajectories, so orbits in the positive quadrant are closed curves around the coexistence point. The cycles are neutrally stable and their amplitude depends on the initial conditions, so they are not limit cycles. Adding density-dependent prey growth, a saturating predation response, or other structure typically replaces the family of cycles by a stable equilibrium or an isolated limit cycle.

## Generalisations
The competitive and multispecies form is $\dot{x}_i=x_i\big(r_i+\sum_j a_{ij}x_j\big)$ with an interaction matrix $A=(a_{ij})$. The sign pattern of $A$ encodes competition, mutualism or predation, and the dynamics can be stable, oscillatory or chaotic once there are three or more species. The replicator equation of evolutionary game theory with $n$ types is equivalent to this model with $n-1$ species after a change of variables.

## Relation to Active Inference
The link is indirect. Population models of this type are used for the ecological and evolutionary systems in this vault. In neuroscience, generalised Lotka-Volterra dynamics with competitive couplings produce winnerless competition and stable heteroclinic channels that generate neural sequences, which suits models of sequential perception and action. These uses are modelling analogies, and the equations are not part of the Active Inference formalism itself.

## Related concepts
- [[population_dynamics]] — the wider field of abundance models
- [[ecological_dynamics]] — ecological context for predator-prey interaction
- [[replicator_dynamics]] — equivalent system for evolving strategy frequencies
- [[limit_cycles]] — isolated cycles that modified models may exhibit
- [[evolutionary_game_theory]] — a framework linking payoffs to such dynamics

## References
- Lotka, A. J. (1925). Elements of Physical Biology. Williams & Wilkins.
- Volterra, V. (1926). Fluctuations in the abundance of a species considered mathematically. Nature, 118, 558-560.
- Murray, J. D. (2002). Mathematical Biology I: An Introduction (3rd ed.). Springer.
