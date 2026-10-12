---
title: Quasi-Newton Methods
description: Optimisation algorithms that build an approximation to the Hessian or its inverse from successive gradients, including the secant condition, the BFGS update, and limited-memory variants.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - optimization
  - second_order_methods
  - numerical_methods
semantic_relations:
  - type: relates
    links:
      - "[[optimization_algorithms]]"
      - "[[trust_region]]"
      - "[[conjugate_gradient]]"
      - "[[variational_methods]]"
      - "[[convex_optimization]]"
---

# Quasi-Newton Methods

## Overview
Newton's method minimises a smooth function $f$ by solving $\nabla^2f(x_k)\,p_k=-\nabla f(x_k)$ at each step, which converges quadratically near a minimiser but requires second derivatives and a linear solve. Quasi-newton methods keep the iteration $x_{k+1}=x_k+\alpha_kp_k$ with a line search but replace the Hessian by an approximation $B_k$ updated from gradient differences alone. They typically achieve superlinear convergence at a lower cost per iteration than Newton's method.

## The secant condition and BFGS
Let $s_k=x_{k+1}-x_k$ and $y_k=\nabla f(x_{k+1})-\nabla f(x_k)$. A new approximation $B_{k+1}$ is required to satisfy the secant condition
$$B_{k+1}s_k=y_k,$$
which mimics the relation $\nabla^2f\,s\approx y$ for a quadratic function. The condition can be met by a symmetric positive definite matrix only if the curvature condition $y_k^\top s_k>0$ holds, which a line search satisfying the Wolfe conditions ensures. The BFGS update, named for Broyden, Fletcher, Goldfarb and Shanno, selects among such matrices and is conveniently written for the inverse approximation $H_k\approx B_k^{-1}$:
$$H_{k+1}=\Big(I-\rho_ks_ky_k^\top\Big)H_k\Big(I-\rho_ky_ks_k^\top\Big)+\rho_ks_ks_k^\top,\qquad\rho_k=\frac{1}{y_k^\top s_k}.$$
The search direction is $p_k=-H_k\nabla f(x_k)$, so no matrix is inverted. The earlier DFP update is the dual of BFGS, and the symmetric rank-one (SR1) update does not preserve positive definiteness.

## Limited memory
Storing $H_k$ costs $O(n^2)$, which is prohibitive for large $n$. The L-BFGS method stores only the last $m$ pairs $(s_k,y_k)$, typically $m$ between 5 and 20, and computes $H_k\nabla f$ by a two-loop recursion in $O(mn)$ operations (Liu & Nocedal, 1989). It is a standard choice for smooth large-scale problems.

## Relation to Active Inference
Quasi-newton routines are general-purpose optimisers, and the connection to Active Inference is practical. Maximum a posteriori fitting, Laplace approximations ([[laplace_approximation]]), and variational objectives with a smooth parameterisation are often minimised by BFGS or L-BFGS. Other inference schemes use gradient flows or natural gradients ([[natural_gradients]]) in which a metric plays the role of the preconditioner, a comparable idea with a different origin: the Fisher metric is derived from the model, whereas BFGS estimates curvature from observed gradients. Active Inference theory does not prescribe a particular optimiser.

## Related concepts
- [[optimization_algorithms]] — the wider family of methods
- [[trust_region]] — an alternative globalisation strategy
- [[conjugate_gradient]] — a related first-order method with curvature information
- [[natural_gradients]] — preconditioning by a Riemannian metric
- [[variational_methods]] — objectives often optimised with these algorithms
- [[convex_optimization]] — the setting with the strongest convergence results

## References
- Nocedal, J., & Wright, S. J. (2006). Numerical Optimization (2nd ed.). Springer.
- Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. Mathematical Programming, 45, 503-528.
