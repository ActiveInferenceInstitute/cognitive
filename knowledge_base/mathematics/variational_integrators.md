---
title: Variational Integrators
description: Numerical schemes for mechanical systems obtained by discretising Hamilton's action principle, which preserve symplectic structure and momentum maps, and their relevance to discretising path-integral formulations.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - numerical_methods
  - mechanics
  - geometric_integration
semantic_relations:
  - type: relates
    links:
      - "[[lagrangian_mechanics]]"
      - "[[euler_lagrange_equations]]"
      - "[[symplectic_geometry]]"
      - "[[noether_theorem]]"
      - "[[path_integral_theory]]"
---

# Variational Integrators

## Overview
A variational integrator is a time-stepping method for a mechanical system derived by discretising the action principle rather than the equations of motion. The continuous Lagrangian is replaced by a discrete Lagrangian, and the update rule is the condition that the discrete action is stationary. Because the structure of Hamilton's principle is retained, the resulting maps share geometric properties with the exact flow, which gives good long-time behaviour.

## Construction
Choose a time step $h$ and a discrete Lagrangian $L_d(q_k, q_{k+1}) \approx \int_{kh}^{(k+1)h} L(q,\dot q)\,dt$. The discrete action is $S_d = \sum_k L_d(q_k, q_{k+1})$. Requiring $S_d$ to be stationary with respect to each interior point $q_k$, with endpoints fixed, gives the discrete Euler-Lagrange equations
$$D_2 L_d(q_{k-1}, q_k) + D_1 L_d(q_k, q_{k+1}) = 0,$$
where $D_1$ and $D_2$ denote derivatives with respect to the first and second arguments. This is an implicit rule for $q_{k+1}$ in general. A standard example uses the trapezoidal approximation $L_d = h\left[\tfrac{m}{2}\left(\tfrac{q_{k+1}-q_k}{h}\right)^2 - \tfrac12\big(V(q_k)+V(q_{k+1})\big)\right]$, which reproduces the Stormer-Verlet (leapfrog) scheme.

## Properties
- **Symplecticity**: the discrete flow preserves a discrete symplectic two-form.
- **Momentum conservation**: a discrete symmetry of $L_d$ yields a conserved discrete momentum, a discrete counterpart of Noether's theorem.
- **Energy behaviour**: energy is generally not conserved exactly, but for well-behaved systems it shows bounded oscillation over very long times instead of the steady drift typical of non-geometric schemes.
- **Accuracy**: the order is set by the quadrature used for $L_d$.

## Relation to Active Inference
The link is indirect. Path-integral formulations of the free energy principle treat trajectories as the objects of inference and work with an action functional over paths, and discretising such a path requires a rule for the time step. Variational integrators are one principled way to do this when the underlying dynamics are conservative. Standard Active Inference schemes, such as generalised filtering, do not rely on them, so this is a numerical connection and not part of the theory.

## Related concepts
- [[lagrangian_mechanics]] — the continuous theory that is discretised
- [[hamiltonian_mechanics]] — the symplectic phase-space picture of the same dynamics
- [[principle_of_least_action]] — the stationarity principle on which the construction rests
- [[numerical_methods]] — the wider family of integration schemes
- [[variational_calculus]] — the mathematics of stationary functionals
- [[path_integral_theory]] — where discretised paths appear in this vault

## References
- Marsden, J. E., & West, M. (2001). Discrete mechanics and variational integrators. Acta Numerica, 10, 357-514.
- Hairer, E., Lubich, C., & Wanner, G. (2006). Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations (2nd ed.). Springer.
