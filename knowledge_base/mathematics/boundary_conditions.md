---
title: Boundary Conditions
description: Constraints imposed on a differential equation or dynamical system at the edge of its domain, with the main types, their effect on solutions, and their use in grid-world models and variational problems.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - differential_equations
  - dynamical_systems
  - variational_calculus
semantic_relations:
  - type: relates
    links:
      - "[[differential_equations]]"
      - "[[dynamical_systems]]"
      - "[[variational_calculus]]"
      - "[[markov_blanket]]"
---

# Boundary Conditions

## Overview
A boundary condition fixes the behaviour of a solution on the boundary $\partial\Omega$ of its spatial or temporal domain $\Omega$. A differential equation alone has a family of solutions, and boundary data select one. The word also covers the edge rules of discrete systems: what happens when an agent on a grid tries to leave it.

## Main types
For a field $u$ on $\Omega$:
- **Dirichlet:** $u=g$ on $\partial\Omega$ (value fixed).
- **Neumann:** $\partial u/\partial n=g$ (normal flux fixed; zero flux is a reflecting wall).
- **Robin:** $\alpha u+\beta\,\partial u/\partial n=g$ (mixed).
- **Periodic:** $u$ and its derivatives agree on opposite edges, equivalent to a torus.
- **Absorbing and reflecting** are the probabilistic names for Dirichlet-zero and zero-flux conditions on a diffusing density ([[fokker_planck]]).

Example: steady heat flow $u''=0$ on $[0,L]$ with $u(0)=a$, $u(L)=b$ has the unique solution $u(x)=a+(b-a)x/L$. An initial-value problem specifies data at one time; a boundary-value problem specifies data at both ends or on the whole boundary, and existence and uniqueness need separate argument.

In variational problems, the Euler-Lagrange equations extremise an action among paths with fixed endpoints, or with natural boundary conditions when endpoints are free ([[variational_calculus]], [[lagrangian_mechanics]]).

## Role in Active Inference
Three uses are worth separating.
1. **Discrete models.** In a grid-world, edge handling is written into the transition arrays $B(a)$. A move into a wall is encoded as staying in place, which keeps every column summing to one. This is a Neumann-like (reflecting) choice and changes the agent's inferred dynamics and the shape of its steady-state distribution.
2. **Continuous-time and path-integral formulations.** Paths are scored between a fixed initial state and an endpoint, so the endpoint constraint is a boundary condition on the action functional ([[path_integral_free_energy]]).
3. **Markov blankets.** The blanket is a statistical boundary, not a spatial one: it makes internal and external states conditionally independent. Calling it a boundary condition is an analogy, since it constrains dependence structure and not field values at an edge.

## Related concepts
- [[differential_equations]] — the equations to which conditions are attached
- [[dynamical_systems]] — flows on domains with edges or periodic topology
- [[variational_calculus]] — fixed and natural endpoint conditions
- [[markov_blanket]] — a statistical boundary between internal and external states
- [[state_space]] — the domain whose edges must be specified
- [[fokker_planck]] — density evolution with absorbing or reflecting boundaries
