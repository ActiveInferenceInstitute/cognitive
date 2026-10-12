---
title: Constraint Optimization
description: Minimising or maximising an objective subject to equality and inequality constraints, covering Lagrange multipliers, the KKT conditions, duality, and uses in normalised probability and inference problems.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - optimization
  - lagrange_multipliers
  - duality
semantic_relations:
  - type: relates
    links:
      - "[[optimization_theory]]"
      - "[[convex_optimization]]"
      - "[[action_distribution]]"
      - "[[maximum_entropy_principle]]"
      - "[[probability_simplex]]"
---

# Constraint Optimization

## Overview
Constraint optimization studies problems of the form
$$\min_{x\in\mathbb{R}^n} f(x)\quad\text{subject to}\quad g_i(x)\le 0\ (i=1,\dots,m),\qquad h_j(x)=0\ (j=1,\dots,p).$$
The constraints define a feasible set, and the solution must lie inside it. Even when the unconstrained minimum of $f$ is easy to find, the constrained minimum usually sits on the boundary of the feasible set, so the optimality conditions change.

## Lagrangian and KKT conditions
The Lagrangian is $L(x,\lambda,\nu)=f(x)+\sum_i\lambda_i g_i(x)+\sum_j\nu_j h_j(x)$. At a local minimiser $x^*$ satisfying a constraint qualification (for example linear independence of the active constraint gradients), there exist multipliers such that the Karush-Kuhn-Tucker conditions hold:
- stationarity: $\nabla f(x^*)+\sum_i\lambda_i\nabla g_i(x^*)+\sum_j\nu_j\nabla h_j(x^*)=0$;
- primal feasibility: $g_i(x^*)\le 0$ and $h_j(x^*)=0$;
- dual feasibility: $\lambda_i\ge 0$;
- complementary slackness: $\lambda_i g_i(x^*)=0$.

For a convex problem with affine equality constraints these conditions are also sufficient. For $\lambda\ge0$ the dual function $d(\lambda,\nu)=\inf_x L(x,\lambda,\nu)$ lower-bounds the optimal value, and under Slater's condition a convex problem has zero duality gap. With equality constraints $h_j(x)=c_j$, the multiplier $\nu_j$ equals, up to the sign convention of the Lagrangian, the sensitivity of the optimal value to the level $c_j$ of the $j$-th constraint.

## A standard example
Maximising the entropy $-\sum_k p_k\ln p_k$ subject to $\sum_k p_k=1$ and a fixed mean $\sum_k p_k E_k=\bar E$ gives, by stationarity, $p_k\propto\exp(-\beta E_k)$, with $\beta$ the multiplier of the mean constraint. This is the Gibbs form that appears in statistical mechanics and in softmax choice rules.

## Relation to Active Inference
The link is a direct mathematical one but a modest one. Probabilities live on the simplex, so any variational problem over beliefs or over an action distribution carries the normalisation constraint $\sum_k p_k=1$, $p_k\ge0$. Minimising variational free energy over a categorical belief with this constraint yields the softmax updates used in discrete Active Inference. Treating the Lagrange multiplier as a precision-like quantity is an interpretive choice, not a requirement of the theory. Constraint formulations are also used in planning and control, for example constrained trajectory optimisation.

## Related concepts
- [[convex_optimization]] — the class in which KKT conditions are sufficient and duality is strong
- [[optimization_theory]] — general framework for objectives and feasible sets
- [[maximum_entropy_principle]] — constrained entropy maximisation yielding exponential-family forms
- [[action_distribution]] — a distribution constrained to the probability simplex
- [[variational_calculus]] — the function-space analogue with constraints on functionals

## References
- Boyd, S., & Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press.
- Nocedal, J., & Wright, S. J. (2006). Numerical Optimization (2nd ed.). Springer.
- Bertsekas, D. P. (1999). Nonlinear Programming (2nd ed.). Athena Scientific.
