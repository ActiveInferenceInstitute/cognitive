---
title: Least Action Principle
description: The use of stationary action to select the most probable path of a stochastic system, including the Onsager-Machlup functional, stationary-phase approximations, and the notion of free action in path-integral formulations of the free energy principle.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - path_integrals
  - stochastic_dynamics
  - variational_principles
semantic_relations:
  - type: relates
    links:
      - "[[principle_of_least_action]]"
      - "[[path_integral]]"
      - "[[path_integral_free_energy]]"
      - "[[langevin_dynamics]]"
      - "[[variational_principles]]"
---

# Least Action Principle

## Overview
In path-integral treatments, an action functional assigns a number to each trajectory, and the trajectories that dominate are those at which it is stationary. This page covers that use for stochastic and inferential systems. The classical statement for mechanical systems, in terms of Hamilton's principle, is given in [[principle_of_least_action]].

## Most probable paths
Consider the Langevin equation $\dot x=f(x)+\sqrt{2D}\,\omega$ with white noise $\omega$ and constant diffusion $D$. The probability density of a path $x(t)$ is proportional to $e^{-S_{\mathrm{OM}}[x]}$, where the Onsager-Machlup action is
$$S_{\mathrm{OM}}[x]=\int\frac{\big(\dot x-f(x)\big)^2}{4D}\,dt+\frac12\int\nabla\!\cdot f\,dt.$$
The second term depends on the discretisation convention, and the form shown corresponds to the midpoint (Stratonovich) rule. Paths that minimise $S_{\mathrm{OM}}$ are the most probable paths, and for weak noise they dominate, a saddle-point or stationary-phase argument that recovers the deterministic flow $\dot x=f(x)$ as the zero-noise limit. Fluctuations around the optimal path are handled by a Gaussian expansion of the action.

## Use in inference
A similar structure appears in approximate inference over trajectories. Variational free energy, evaluated along a path and integrated over time, gives a path-level functional, and the variational density over paths is chosen to make it stationary. The same device appears in stochastic optimal control, where the logarithmic transformation of the value function turns a nonlinear problem into a linear one solvable by path integrals.

## Relation to Active Inference
Friston and colleagues call the time integral of free energy free action, and treat the trajectory of internal states that makes it stationary as the optimal belief trajectory, in direct analogy with Hamilton's principle. In generalised coordinates the principle of stationary action then yields the equations of motion for belief updating. This analogy is an active research programme, and the constructs depend on the assumptions that define the formalism, including a steady-state density and a partition into internal and external states. The word action is used in two senses in this literature, the integral of a Lagrangian and the agent's actions on its environment, and these should not be confused.

## Related concepts
- [[principle_of_least_action]] — the classical mechanical principle
- [[path_integral]] — integration over trajectories weighted by the action
- [[path_integral_free_energy]] — free energy formulated over paths
- [[langevin_dynamics]] — the stochastic dynamics assumed above
- [[generalized_coordinates]] — representation of trajectories by their time derivatives
- [[path_integral_control]] — control by path-integral methods

## References
- Onsager, L., & Machlup, S. (1953). Fluctuations and irreversible processes. Physical Review, 91, 1505-1512.
- Friston, K. (2008). Hierarchical models in the brain. PLoS Computational Biology, 4, e1000211.
- Goldstein, H., Poole, C., & Safko, J. (2002). Classical Mechanics (3rd ed.). Addison-Wesley.
