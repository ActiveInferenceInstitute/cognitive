---
title: Nonlinearity
description: Systems whose outputs are not proportional to inputs, covering the failure of superposition, characteristic nonlinear phenomena, and the role of nonlinearity in self-organising and neural systems.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - dynamical_systems
  - superposition
  - bifurcation
semantic_relations:
  - type: relates
    links:
      - "[[dynamical_systems]]"
      - "[[chaos_theory]]"
      - "[[bifurcation_theory]]"
      - "[[complex_systems]]"
      - "[[linear_stability]]"
---

# Nonlinearity

## Overview
A map $F$ is linear if it satisfies superposition: $F(\alpha x+\beta y)=\alpha F(x)+\beta F(y)$. Nonlinearity is the failure of this property, so that doubling an input need not double the output and the effect of two causes together differs from the sum of their separate effects. Most models of living and cognitive systems are nonlinear, and a nonlinear system cannot in general be decomposed into independently solvable parts.

## Characteristic consequences
- **Multiple equilibria.** A linear autonomous system $\dot x=Ax$ has either an isolated fixed point at the origin or a continuum of them. Nonlinear systems can have several isolated fixed points, each with its own basin of attraction (see [[basin_of_attraction]]).
- **Limit cycles.** Isolated closed orbits that attract nearby trajectories occur in nonlinear flows. Linear oscillators instead have a continuous family of orbits whose amplitude depends on initial conditions.
- **Bifurcations.** A smooth change of a parameter can change the qualitative behaviour, for example from a stable fixed point to oscillation ([[bifurcation_theory]], [[hopf_bifurcation]]).
- **Chaos.** Deterministic nonlinear systems of sufficient dimension can show sensitive dependence on initial conditions, quantified by positive Lyapunov exponents ([[chaos_theory]], [[lyapunov_exponents]]).
- **Saturation and thresholds.** Sigmoidal and rectifying functions, such as the logistic function $\sigma(x)=1/(1+e^{-x})$, give bounded responses and all-or-none behaviour.

Near a fixed point a nonlinear system can often be approximated by its linearisation, with the Jacobian governing local stability ([[linear_stability]]). The Hartman-Grobman theorem justifies this when no eigenvalue has zero real part. The approximation fails at bifurcation points and away from the fixed point.

## Relation to Active Inference
Nonlinearity enters Active Inference in several places. Generative models in continuous time usually have nonlinear flow and observation functions, and the Laplace-approximate schemes used to invert them rely on local linearisation about the posterior mean ([[laplace_approximation]]). In neural network models, nonlinear [[activation_function]] choices determine what a network can represent, and without them stacked layers collapse to a single linear map. Dynamical accounts of the brain, including those based on the [[free_energy_principle]], describe itinerant dynamics near critical points, which is an active research programme rather than a settled result. Emergent collective behaviour is commonly attributed to nonlinear interaction among components ([[emergence_self_organization]]).

## Related concepts
- [[dynamical_systems]] — the general framework for nonlinear flows and maps
- [[chaos_theory]] — sensitive dependence as a nonlinear phenomenon
- [[bifurcation_theory]] — qualitative change under parameter variation
- [[linear_stability]] — local linear analysis of nonlinear systems
- [[period_doubling]] — a route to chaos in nonlinear maps
- [[activation_function]] — nonlinearity in neural network units

## References
- Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering (2nd ed.). Westview Press.
- Hirsch, M. W., Smale, S., & Devaney, R. L. (2013). Differential Equations, Dynamical Systems, and an Introduction to Chaos (3rd ed.). Academic Press.
