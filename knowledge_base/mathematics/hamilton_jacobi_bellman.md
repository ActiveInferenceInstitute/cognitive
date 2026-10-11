---
title: Hamilton-Jacobi-Bellman Equation
description: The partial differential equation satisfied by the optimal value function in continuous-time optimal control, its derivation from dynamic programming, the stochastic and linear-quadratic cases, and links to path-integral control.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - optimal_control
  - dynamic_programming
  - value_function
semantic_relations:
  - type: relates
    links:
      - "[[dynamic_programming]]"
      - "[[optimal_control]]"
      - "[[value_function]]"
      - "[[path_integral_control]]"
      - "[[variational_calculus]]"
---

# Hamilton-Jacobi-Bellman Equation

## Overview
The Hamilton-Jacobi-Bellman (HJB) equation is the continuous-time form of Bellman's principle of optimality. It is a partial differential equation for the value function, the minimal cost-to-go from each state and time, and the optimal control is obtained by minimising the expression inside it. It joins dynamic programming with the Hamilton-Jacobi theory of classical mechanics.

## Formulation
Consider dynamics $\dot x=f(x,u)$ and a cost $J=\int_t^T \ell(x,u)\,ds+\Phi(x(T))$. The value function $V(x,t)=\min_u J$ satisfies
$$-\frac{\partial V}{\partial t}=\min_{u}\Big[\ell(x,u)+\nabla V\cdot f(x,u)\Big],\qquad V(x,T)=\Phi(x).$$
The minimiser gives a feedback policy $u^*(x,t)$. For stochastic dynamics $dx=f\,dt+\sigma\,dW$ an additional term $\tfrac12\mathrm{tr}\big(\sigma\sigma^\top\nabla^2V\big)$ appears on the right. In the linear-quadratic case, the solution is quadratic in $x$ and the equation reduces to a Riccati equation. In general the equation is nonlinear, and the value function may not be differentiable, which is addressed by the notion of viscosity solutions.

## Linearisation
When control cost and noise are related in a particular way, the logarithmic transformation $V=-\lambda\ln\psi$ turns the stochastic HJB equation into a linear equation for $\psi$. By the Feynman-Kac formula its solution is an expectation over uncontrolled trajectories, which is the basis of path-integral control and can be estimated by sampling.

## Relation to Active Inference
Active Inference replaces the value function and cost with a prior over outcomes and expected free energy, so policies are selected by inference and not by solving an HJB equation. The two frameworks are related: continuous-time Active Inference can recover optimal-control behaviour as a special case under suitable assumptions, and the equivalence between control and inference in the path-integral setting is a studied connection. The relation is exact only under stated conditions, and the frameworks differ in treating exploration, since expected free energy contains an epistemic term with no counterpart in a standard cost.

## Related concepts
- [[dynamic_programming]] — the discrete-time principle behind the equation
- [[optimal_control]] — the problem class it solves
- [[value_function]] — the function it characterises
- [[path_integral_control]] — control obtained from the linearised equation
- [[hamiltonian_mechanics]] — the source of the Hamilton-Jacobi structure
- [[markov_decision_process]] — the discrete-time counterpart

## References
- Bertsekas, D. P. (2017). Dynamic Programming and Optimal Control, Vol. I (4th ed.). Athena Scientific.
- Fleming, W. H., & Soner, H. M. (2006). Controlled Markov Processes and Viscosity Solutions (2nd ed.). Springer.
- Kappen, H. J. (2005). Linear theory for control of nonlinear stochastic systems. Physical Review Letters, 95, 200201.
