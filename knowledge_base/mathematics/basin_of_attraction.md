---
title: Basin of Attraction
description: The set of initial states whose trajectories converge to a given attractor of a dynamical system, with examples, boundaries, and its use in accounts of stability, memory and self-organisation.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - dynamical_systems
  - attractors
  - stability
semantic_relations:
  - type: relates
    links:
      - "[[attractor_dynamics]]"
      - "[[dynamical_systems]]"
      - "[[lyapunov_functions]]"
      - "[[bifurcation_theory]]"
      - "[[non_equilibrium_steady_state]]"
---

# Basin of Attraction

## Overview
For a dynamical system with flow $\phi_t$ and an attractor $A$, the basin of attraction is the set
$$B(A)=\{x : \phi_t(x)\to A \text{ as } t\to\infty\}.$$
Every initial condition inside the basin eventually approaches the attractor, and conditions outside it do not. The state space is therefore partitioned into basins, one for each attractor, together with boundaries that separate them.

## Example and structure
Consider the one-dimensional system $\dot x = x - x^3$. It has stable fixed points at $x=\pm 1$ and an unstable one at $x=0$. The basin of $x=1$ is $(0,\infty)$ and the basin of $x=-1$ is $(-\infty,0)$, and the point $0$ is the boundary. In higher dimensions the boundary is typically formed by the stable manifolds of saddle points, and it can have complicated or even fractal geometry, in which case small perturbations of the initial state can change the outcome. The basin of an asymptotically stable fixed point is an open set. Its size measures robustness to perturbations: a state pushed beyond the boundary moves to a different attractor. When a parameter changes, basins can shrink, merge or vanish at bifurcations (see [[bifurcation_theory]]).

## Role in modelling
- **Memory and categorisation**: in attractor network models such as Hopfield networks, stored patterns are fixed points and recall is convergence from a partial cue within the pattern's basin.
- **Decision and perception**: bistable percepts and choices are often modelled as two basins, with noise driving switches between them.
- **Resilience**: in ecology and other complex systems, the size of a basin is used as a measure of resilience.

## Relation to Active Inference
The free energy principle describes systems that persist as occupying a limited region of state space, characterised as a non-equilibrium steady state with a density concentrated on a set of characteristic states, sometimes called a random attractor. Under the assumptions of that formalism, the flow can be decomposed so that surprisal acts like a Lyapunov function for the mean dynamics (see [[lyapunov_functions]]), and a system's trajectory returns toward its attracting set. This is an application of dynamical-systems language to the formalism, and how well it describes particular biological systems remains an open empirical and theoretical question.

## Related concepts
- [[attractor_dynamics]] — the attractors whose basins are described
- [[lyapunov_functions]] — a tool for proving convergence within a basin
- [[hysteresis]] — dependence on history arising from coexisting basins
- [[hopfield_networks]] — memory as convergence to an attractor
- [[bistable_perception]] — switching between two perceptual basins
- [[free_energy_principle]] — characteristic states and attracting sets

## References
- Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos (2nd ed.). Westview Press.
- Friston, K. (2013). Life as we know it. Journal of the Royal Society Interface, 10, 20130475.
