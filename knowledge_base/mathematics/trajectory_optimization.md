---
title: Trajectory Optimization
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - trajectory_optimization
  - optimal_control
  - planning
semantic_relations:
  - type: foundation
    links:
      - "[[optimal_control]]"
      - "[[optimization_algorithms]]"
  - type: relates
    links:
      - "[[planning_as_inference]]"
      - "[[path_integral]]"
      - "[[expected_free_energy]]"
---

# Trajectory Optimization

## Overview
Trajectory optimization computes a state and control sequence that minimises a cost subject to the system dynamics. It is the open-loop, finite-horizon counterpart of optimal control and is the computational core of model predictive control.

## Definition / Formalism
In discrete time over horizon $T$:

$$\min_{x_{0:T},\,u_{0:T-1}}\; \ell_T(x_T)+\sum_{t=0}^{T-1}\ell(x_t,u_t)\quad\text{s.t.}\quad x_{t+1}=f(x_t,u_t),\; x_0\text{ given},$$

with optional bounds on $x$ and $u$. Methods split into two families. Shooting methods treat only $u$ as decision variables and obtain $x$ by forward simulation. Direct collocation treats both $x$ and $u$ as variables and imposes the dynamics as constraints, which yields a sparse nonlinear program. For linear dynamics and quadratic cost the problem is the LQR and its solution follows from a backward Riccati recursion. Iterative LQR and differential dynamic programming repeat this by linearising $f$ and quadratising the cost about the current trajectory. Pontryagin's principle gives the indirect (adjoint) formulation in continuous time. Model predictive control re-solves the problem at each step and applies only the first control.

A discrete illustration of why structure matters: enumerating all action sequences with 3 actions and horizon 5 means $3^5=243$ candidates, and the count is exponential in $T$, so gradient or dynamic-programming structure replaces enumeration.

## Role in Active Inference
Evaluating a policy $\pi=(a_1,\dots,a_T)$ by summing expected free energy $G(\pi)=\sum_\tau G(\pi,\tau)$ over a horizon is structurally a trajectory cost, with $G$ as the stage cost. Two differences prevent identifying the two. Active inference keeps a distribution over policies via a softmax of $-\gamma G$ instead of returning one optimal trajectory, and its stage cost includes an information-gain term that depends on beliefs, not only on states and controls. Related formulations do connect: planning as inference and KL or path-integral control (Kappen; Todorov) rewrite stochastic optimal control as inference over trajectories. In the repo, planning is by explicit policy enumeration or search, not by continuous trajectory optimisation.

## Related concepts
- [[optimal_control]] — the problem class that trajectory optimization solves
- [[optimization_algorithms]] — the solvers used for the resulting programs
- [[planning_as_inference]] — control recast as probabilistic inference
- [[path_integral]] — trajectory-level formulation of stochastic dynamics
- [[policy_search_methods]] — search over action sequences in this vault
- [[state_space]] — the dynamics $f$ live in a state space
