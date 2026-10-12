---
title: Model Predictive Control
description: A control method that repeatedly solves a finite-horizon optimisation using a dynamics model, applies the first action and re-plans, with its formulation, stability conditions and similarity to policy planning in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - control_theory
  - planning
  - optimization
semantic_relations:
  - type: relates
    links:
      - "[[optimal_control]]"
      - "[[control_theory]]"
      - "[[trajectory_optimization]]"
      - "[[dynamic_programming]]"
      - "[[policy_selection]]"
      - "[[pragmatic_value]]"
wikidata: Q1782962
---

# Model Predictive Control

## Overview
Model predictive control (MPC), also called receding-horizon control, chooses actions by optimising predicted future behaviour. At each time step it uses a model of the system to find the best sequence of control inputs over a finite horizon, applies only the first input, observes the new state and repeats the optimisation. Re-planning at every step provides feedback and corrects for model error and disturbances.

## Formulation
For a discrete-time system $x_{k+1}=f(x_k,u_k)$, the controller solves at time $t$
$$\min_{u_0,\dots,u_{N-1}}\;\sum_{k=0}^{N-1}\ell(x_k,u_k)+V_f(x_N)\quad\text{s.t.}\quad x_0=x_t,\; x_{k+1}=f(x_k,u_k),\; (x_k,u_k)\in\mathcal{X}\times\mathcal{U}.$$
Here $\ell$ is a stage cost, $V_f$ a terminal cost and $N$ the horizon. A key advantage is that constraints on states and inputs are handled directly. For linear dynamics with a quadratic cost and linear constraints the problem is a convex quadratic program. Nonlinear models give nonlinear programs that must be solved within the sampling interval.

## Stability and limits
Closed-loop stability is not automatic with a finite horizon. Standard guarantees use a terminal constraint set and a terminal cost that acts as a control Lyapunov function, so that the optimal cost decreases along closed-loop trajectories. Practical difficulties include computation time, model mismatch, and the loss of guarantees under uncertainty, which stochastic and robust variants try to address. MPC approximates, but does not solve, the infinite-horizon optimal control problem given by dynamic programming.

## Relation to Active Inference
The two share a structure. Discrete-time Active Inference evaluates candidate policies over a planning horizon, scores them, executes an action and updates beliefs before planning again, which is the receding-horizon pattern. The differences are significant: the cost function is replaced by expected free energy, which contains an epistemic term as well as the expected preference-based term often called pragmatic value, and planning is performed over beliefs about hidden states and not over known states. Because of this, Active Inference can choose actions for their information gain, while standard MPC with a known state does not. Control theorists have studied correspondences between the two approaches, and the similarity is structural, not an equivalence.

## Related concepts
- [[optimal_control]] — the problem that MPC solves approximately in a receding-horizon fashion
- [[trajectory_optimization]] — the optimisation of action sequences inside each MPC step
- [[dynamic_programming]] — the infinite-horizon alternative
- [[policy_selection]] — policy scoring over a horizon in Active Inference
- [[active_inference_for_control]] — control applications of the framework

## References
- Rawlings, J. B., Mayne, D. Q., & Diehl, M. (2017). Model Predictive Control: Theory, Computation, and Design (2nd ed.). Nob Hill Publishing.
- Camacho, E. F., & Bordons, C. (2007). Model Predictive Control (2nd ed.). Springer.
- Mayne, D. Q., Rawlings, J. B., Rao, C. V., & Scokaert, P. O. M. (2000). Constrained model predictive control: stability and optimality. Automatica, 36, 789-814.
