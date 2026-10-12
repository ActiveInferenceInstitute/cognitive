---
title: Stochastic Control
description: The optimal control of dynamical systems subject to random disturbances, covering controlled diffusions, the Hamilton-Jacobi-Bellman equation, the linear-quadratic-Gaussian case and control as inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - control_theory
  - optimal_control
  - stochastic_processes
semantic_relations:
  - type: relates
    links:
      - "[[optimal_control]]"
      - "[[hamilton_jacobi_bellman]]"
      - "[[stochastic_differential_equations]]"
      - "[[path_integral_control]]"
      - "[[action_selection]]"
---

# Stochastic Control

## Overview
Stochastic control concerns the choice of control actions for a system whose evolution is affected by random noise, so as to minimise an expected cost. In continuous time, the system is modelled as a controlled diffusion
$$dX_t=b(X_t,u_t)\,dt+\sigma(X_t,u_t)\,dW_t,$$
and the objective is $J(u)=\mathbb{E}\big[\int_0^Tc(X_t,u_t)\,dt+\Phi(X_T)\big]$. Control is chosen as a function of the information available, either the full state or, under partial observation, the history of noisy measurements.

## Dynamic programming
If $V(t,x)$ is the optimal expected cost-to-go from state $x$ at time $t$, the dynamic programming principle leads to the Hamilton-Jacobi-Bellman equation
$$-\partial_tV=\min_u\Big\{c(x,u)+b(x,u)^\top\nabla V+\tfrac12\operatorname{tr}\big(\sigma\sigma^\top(x,u)\,\nabla^2V\big)\Big\},\qquad V(T,x)=\Phi(x).$$
The second-order term, absent in deterministic control, is the effect of noise. The equation typically has no classical solution, and the right notion is the viscosity solution (Fleming & Soner, 2006). See [[hamilton_jacobi_bellman]] and [[dynamic_programming]].

## Special cases
- **Linear-quadratic-Gaussian.** For linear dynamics, quadratic cost and Gaussian noise, $V$ is quadratic and the optimal control is linear in the state. Under partial observation, the separation principle gives the optimal control as the state-feedback law applied to the Kalman filter estimate ([[feedback_control]]).
- **Risk-sensitive and path-integral control.** For a class of problems in which the control cost is quadratic and noise enters in the same channels as control, a logarithmic transformation of $V$ linearises the HJB equation, and the solution can be written as an expectation over uncontrolled paths (Kappen, 2005). This is the basis of [[path_integral_control]].
- **Discrete time.** Controlled Markov chains lead to the Bellman equation of [[markov_decision_process]].

## Relation to Active Inference
Stochastic control is the classical counterpart of action selection in Active Inference. Planning as inference formulations rewrite control problems as inference over trajectories, and the linearly solvable class links the KL-regularised control cost to a divergence between controlled and uncontrolled path distributions (Todorov, 2009). Active Inference differs by replacing the cost function with prior preferences over outcomes and by including an epistemic term in expected free energy, which standard stochastic control, with a fixed state estimator, does not contain. The correspondence is established for restricted cases, and equivalence between the two frameworks in general is not claimed. See [[expected_free_energy]] and [[optimal_control]].

## Related concepts
- [[optimal_control]] — the deterministic precursor
- [[hamilton_jacobi_bellman]] — the central equation
- [[stochastic_differential_equations]] — the models of controlled dynamics
- [[path_integral_control]] — linearly solvable stochastic control
- [[action_selection]] — the cognitive counterpart
- [[model_predictive_control]] — receding-horizon implementation

## References
- Fleming, W. H., & Soner, H. M. (2006). Controlled Markov Processes and Viscosity Solutions (2nd ed.). Springer.
- Kappen, H. J. (2005). Linear theory for control of nonlinear stochastic systems. Physical Review Letters, 95, 200201.
- Bertsekas, D. P. (2017). Dynamic Programming and Optimal Control, Vol. I (4th ed.). Athena Scientific.
- Todorov, E. (2009). Efficient computation of optimal actions. Proceedings of the National Academy of Sciences, 106, 11478-11483.
