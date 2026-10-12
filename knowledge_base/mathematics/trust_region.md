---
title: Trust Region Methods
description: Optimisation methods that minimise a local model of the objective within a region where it is trusted, adapting the region size from the agreement between predicted and actual decrease.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - optimization
  - numerical_methods
  - globalization
semantic_relations:
  - type: relates
    links:
      - "[[quasi_newton]]"
      - "[[optimization_algorithms]]"
      - "[[convex_optimization]]"
      - "[[natural_gradients]]"
      - "[[variational_methods]]"
---

# Trust Region Methods

## Overview
Trust region methods are iterative algorithms for minimising a smooth function $f$. At each iterate $x_k$, they build a local model $m_k$ of $f$, usually a quadratic, and minimise it only within a ball around $x_k$ whose radius $\Delta_k$ expresses how far the model is believed to be accurate. The step is accepted or rejected according to how well the model predicted the real change in $f$, and the radius is adjusted accordingly. They are an alternative to line search for making Newton-like methods converge from poor starting points.

## The subproblem and radius update
With gradient $g_k=\nabla f(x_k)$ and a symmetric matrix $B_k$ approximating the Hessian, the step solves
$$\min_{p}\; m_k(p)=f(x_k)+g_k^\top p+\tfrac12p^\top B_kp\quad\text{subject to}\quad\|p\|\le\Delta_k.$$
The agreement ratio compares the actual and predicted reduction,
$$\rho_k=\frac{f(x_k)-f(x_k+p_k)}{m_k(0)-m_k(p_k)}.$$
If $\rho_k$ is small or negative the step is rejected and $\Delta_k$ shrunk. If it is close to one, and the step reached the boundary, $\Delta_k$ is enlarged. Intermediate values leave the radius unchanged.

## Solving the subproblem
A vector $p^*$ solves the subproblem exactly if there is a multiplier $\lambda\ge0$ such that $(B_k+\lambda I)p^*=-g_k$, $\lambda(\Delta_k-\|p^*\|)=0$, and $B_k+\lambda I$ is positive semidefinite. This characterisation shows that trust region steps interpolate between the Newton step, when $B_k$ is positive definite and the Newton step lies inside the region, and a scaled steepest descent step for small $\Delta_k$. The subproblem is well-posed even when $B_k$ is indefinite, a case in which pure Newton steps fail. Practical algorithms use approximations: the Cauchy point, the dogleg path, or the Steihaug conjugate gradient method that truncates at the boundary or at negative curvature ([[conjugate_gradient]]). Under mild assumptions the iterates have limit points that are stationary points (Conn, Gould & Toint, 2000; Nocedal & Wright, 2006).

## Relation to Active Inference
Trust regions are general optimisation tools and appear in inference through the objective being minimised, for instance when variational free energy is optimised over the parameters of an approximate posterior. A related idea constrains policy updates by a KL divergence, as in trust region policy optimisation in reinforcement learning, where the region is defined in distribution space and not in parameter space. This is closely related to natural gradient methods, where the Fisher metric defines the geometry ([[natural_gradients]], [[policy_gradient]]). Active Inference itself does not mandate a trust region scheme, and the connection is one of shared mathematical technique.

## Related concepts
- [[quasi_newton]] — supplies the Hessian approximation $B_k$
- [[optimization_algorithms]] — the broader class of methods
- [[convex_optimization]] — the case with global guarantees
- [[natural_gradients]] — geometry-aware steps
- [[policy_gradient]] — policy optimisation with divergence-constrained steps
- [[variational_methods]] — settings in which these solvers are applied

## References
- Nocedal, J., & Wright, S. J. (2006). Numerical Optimization (2nd ed.). Springer.
- Conn, A. R., Gould, N. I. M., & Toint, P. L. (2000). Trust-Region Methods. Society for Industrial and Applied Mathematics.
