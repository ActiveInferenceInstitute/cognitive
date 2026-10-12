---
title: Dissipation Function
description: The dissipation function appears in mechanics as Rayleigh's function giving frictional power loss, and in statistical mechanics as the Evans-Searles quantity that governs fluctuation theorems and entropy production.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - nonequilibrium
  - classical_mechanics
  - fluctuation_theorems
semantic_relations:
  - type: relates
    links:
      - "[[fluctuation_theorems]]"
      - "[[entropy_production]]"
      - "[[lagrangian_mechanics]]"
      - "[[non_equilibrium_thermodynamics]]"
      - "[[path_integral_theory]]"
---

# Dissipation Function

## Overview
The name refers to two related but distinct objects. In analytical mechanics, Rayleigh's dissipation function describes velocity-dependent friction. In nonequilibrium statistical mechanics, the dissipation function of Evans and Searles is a path functional that measures irreversibility and appears in an exact fluctuation theorem. Both quantify how dynamics lose or waste energy as heat.

## Rayleigh's dissipation function
For generalised coordinates $q_i$ subject to linear viscous forces, the dissipation function is
$$\mathcal{R}=\tfrac12\sum_{i,j}\gamma_{ij}\,\dot q_i\dot q_j,$$
with $\gamma_{ij}$ a symmetric, positive semi-definite matrix of friction coefficients. The Lagrange equations are modified to
$$\frac{d}{dt}\frac{\partial L}{\partial \dot q_i}-\frac{\partial L}{\partial q_i}=-\frac{\partial \mathcal{R}}{\partial \dot q_i}.$$
The rate at which the system loses mechanical energy to friction is $2\mathcal{R}$. This device lets dissipative forces that derive from a quadratic form be handled alongside a Lagrangian. The Onsager-Machlup approach to stochastic paths uses a related quadratic form in the exponent of the path probability, which explains its appearance in path-integral treatments of noisy dynamics.

## Evans-Searles dissipation function
For a system started from a distribution $f(\Gamma,0)$ and evolving for time $t$ under thermostatted, possibly driven, dynamics, the dissipation function $\Omega_t$ is defined through the logarithmic ratio of the probability of a set of initial conditions to that of its time-reversed image, with an additional term for the phase-space volume change. It satisfies the fluctuation theorem
$$\frac{P(\Omega_t=A)}{P(\Omega_t=-A)}=e^{A},$$
which makes negative values exponentially less likely than positive ones without forbidding them. The average $\langle\Omega_t\rangle$ is non-negative, in the manner of the second law, and in appropriate settings $\Omega_t$ equals the entropy production, in units of $k_B$, accumulated over the interval.

## Relation to Active Inference
The link is thermodynamic and conceptual. Treatments of the free energy principle that discuss nonequilibrium steady states relate the irreversibility of the flow to entropy production, which is the same family of quantities as the second sense above. Rayleigh's function is the classical analogue of the dissipative component in the decomposition of a flow into dissipative and solenoidal parts. The correspondence is formal and depends on modelling assumptions; it does not imply that neural populations are frictional mechanical systems.

## Related concepts
- [[fluctuation_theorems]] — exact relations constraining the probability of negative dissipation
- [[entropy_production]] — the thermodynamic quantity the dissipation function can equal
- [[lagrangian_mechanics]] — the framework extended by Rayleigh's function
- [[non_equilibrium_thermodynamics]] — the setting for irreversible processes
- [[fluctuation_dissipation]] — the relation between noise and friction
- [[path_integral_theory]] — path probabilities with dissipative terms

## References
- Goldstein, H., Poole, C., & Safko, J. (2002). Classical Mechanics (3rd ed.). Addison-Wesley.
- Onsager, L., & Machlup, S. (1953). Fluctuations and irreversible processes. Physical Review, 91, 1505-1512.
- Evans, D. J., & Searles, D. J. (2002). The fluctuation theorem. Advances in Physics, 51, 1529-1585.
