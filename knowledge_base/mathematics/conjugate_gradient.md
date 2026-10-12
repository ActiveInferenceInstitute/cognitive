---
title: Conjugate Gradient
description: The iterative method for solving symmetric positive definite linear systems and minimising quadratic functions, its algorithm, convergence bound, nonlinear extensions, and role in large-scale inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - optimization
  - numerical_linear_algebra
  - iterative_methods
semantic_relations:
  - type: relates
    links:
      - "[[gradient_descent]]"
      - "[[quasi_newton]]"
      - "[[convex_optimization]]"
      - "[[linear_algebra]]"
      - "[[optimization_methods]]"
      - "[[natural_gradients]]"
---

# Conjugate Gradient

## Overview
The conjugate gradient method of Hestenes and Stiefel (1952) solves a linear system $Ax = b$ in which $A$ is symmetric and positive definite. Equivalently, it minimises the quadratic function $f(x) = \tfrac{1}{2}x^\top A x - b^\top x$, whose gradient is $Ax - b$. It uses only products of $A$ with vectors, so $A$ need not be stored or factorised, which makes it suitable for large sparse problems. It is often grouped with second-order methods, but it uses curvature only through products with $A$.

## Algorithm
Two vectors $p_i$ and $p_j$ are $A$-conjugate if $p_i^\top A p_j = 0$ for $i \neq j$. The search directions $p_k$ generated below are mutually $A$-conjugate. Starting from $x_0$, with residual $r_0 = b - Ax_0$ and direction $p_0 = r_0$, iterate for $k = 0, 1, \dots$:
$$\alpha_k = \frac{r_k^\top r_k}{p_k^\top A p_k}, \quad x_{k+1} = x_k + \alpha_k p_k, \quad r_{k+1} = r_k - \alpha_k A p_k,$$
$$\beta_k = \frac{r_{k+1}^\top r_{k+1}}{r_k^\top r_k}, \quad p_{k+1} = r_{k+1} + \beta_k p_k.$$
Each step needs one matrix-vector product. In exact arithmetic the residuals are mutually orthogonal and the method reaches the solution in at most $n$ steps for an $n \times n$ system. The iterate $x_k$ minimises $f$ over $x_0 + \mathcal{K}_k$, where $\mathcal{K}_k = \mathrm{span}\{r_0, Ar_0, \dots, A^{k-1}r_0\}$ is the Krylov subspace.

## Convergence
With condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ and error $e_k = x_k - x^*$, the bound in the energy norm $\lVert e \rVert_A = \sqrt{e^\top A e}$ is
$$\lVert e_k \rVert_A \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k \lVert e_0 \rVert_A.$$
Convergence is thus governed by $\sqrt{\kappa}$, compared with $\kappa$ for steepest descent. A preconditioner $M \approx A$ replaces the system by one with a smaller effective condition number (Shewchuk, 1994).

## Nonlinear conjugate gradient
For a general smooth function, the residual is replaced by the negative gradient $g_k$, with a line search along $p_k$. Fletcher and Reeves (1964) chose $\beta_k = \lVert g_{k+1}\rVert^2 / \lVert g_k\rVert^2$. Truncated Newton methods apply linear conjugate gradient to the Newton system $H d = -g$ using Hessian-vector products (Nocedal and Wright, 2006).

## Relation to Active Inference
Conjugate gradient is a numerical tool, not part of the theory. It is relevant where inference requires solving large linear systems, such as the Newton step in a Laplace approximation or the natural-gradient step $F\,\delta = \nabla$ with Fisher information matrix $F$ (see [[natural_gradients]]). Active Inference implementations usually use simpler gradient-based updates.

## Related concepts
- [[gradient_descent]] — the first-order baseline that conjugate gradient improves on
- [[quasi_newton]] — an alternative that builds curvature estimates
- [[convex_optimization]] — the setting where quadratic minimisation arises
- [[linear_algebra]] — the matrix theory on which it rests
- [[natural_gradients]] — a use case with Fisher information systems
- [[numerical_methods]] — the wider field of iterative solvers

## References
- Hestenes, M. R., & Stiefel, E. (1952). Methods of conjugate gradients for solving linear systems. Journal of Research of the National Bureau of Standards, 49, 409-436.
- Fletcher, R., & Reeves, C. M. (1964). Function minimization by conjugate gradients. The Computer Journal, 7, 149-154.
- Shewchuk, J. R. (1994). An Introduction to the Conjugate Gradient Method Without the Agonizing Pain. Technical report, Carnegie Mellon University.
- Nocedal, J., & Wright, S. J. (2006). Numerical Optimization (2nd ed.). Springer.
