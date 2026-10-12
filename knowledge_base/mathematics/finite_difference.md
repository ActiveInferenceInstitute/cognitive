---
title: Finite Difference
description: Approximating derivatives and solving differential equations by differences on a grid, with truncation error, stability conditions and the Lax equivalence theorem, and its use in discretising time for stochastic dynamics.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - numerical_methods
  - differential_equations
  - discretization
semantic_relations:
  - type: relates
    links:
      - "[[numerical_methods]]"
      - "[[differential_equations]]"
      - "[[taylor_series]]"
      - "[[numerical_stability]]"
      - "[[finite_elements]]"
wikidata: Q2068418
---

# Finite Difference

## Overview
Finite difference methods replace derivatives by differences of function values on a grid. They convert differential equations into algebraic recurrences or linear systems that a computer can solve. Their accuracy is read off from Taylor expansions, and their usefulness depends on the interplay of consistency, stability and convergence.

## Difference formulas
With grid spacing $h$, Taylor expansion gives
$$f'(x)=\frac{f(x+h)-f(x)}{h}+O(h)\quad\text{(forward)},\qquad f'(x)=\frac{f(x+h)-f(x-h)}{2h}+O(h^2)\quad\text{(central)},$$
$$f''(x)=\frac{f(x+h)-2f(x)+f(x-h)}{h^2}+O(h^2).$$
Reducing $h$ shrinks the truncation error, but floating-point cancellation grows as $h\to0$, so there is an optimal step size that balances the two. The forward Euler step $y_{n+1}=y_n+\Delta t\,f(y_n)$ is the forward difference applied to an ordinary differential equation.

## Partial differential equations and stability
For the heat equation $u_t=\alpha u_{xx}$, the explicit scheme
$$u_j^{n+1}=u_j^n+\frac{\alpha\,\Delta t}{\Delta x^2}\bigl(u_{j+1}^n-2u_j^n+u_{j-1}^n\bigr)$$
is stable only if $\alpha\Delta t/\Delta x^2\le\tfrac12$. Implicit schemes such as backward Euler and Crank-Nicolson remove this restriction at the cost of solving a linear system per step. The Lax equivalence theorem states that, for a consistent finite difference approximation to a well-posed linear initial value problem, stability is equivalent to convergence.

## Relation to Active Inference
Finite differences enter in practical ways. Continuous-time Active Inference models are simulated by discretising their equations of motion, and generalised coordinates encode higher temporal derivatives that can be viewed as finite-difference approximations of the trajectory. Path integral formulations are defined through time-sliced products of short-time propagators, whose limit as the step tends to zero defines the continuum object. These are numerical and definitional uses; the method is not specific to the theory.

## Related concepts
- [[taylor_series]] — the source of truncation-error estimates
- [[differential_equations]] — the equations being approximated
- [[stochastic_differential_equations]] — discretised in time with the Euler-Maruyama scheme
- [[finite_elements]] — an alternative discretisation based on weak formulations
- [[numerical_stability]] — the condition that keeps errors from growing
- [[path_integral]] — defined as a limit of time-discretised integrals

## References
- LeVeque, R. J. (2007). Finite Difference Methods for Ordinary and Partial Differential Equations. SIAM.
- Strikwerda, J. C. (2004). Finite Difference Schemes and Partial Differential Equations (2nd ed.). SIAM.
