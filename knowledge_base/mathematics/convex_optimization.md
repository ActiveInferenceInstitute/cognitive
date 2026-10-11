---
title: Convex Optimization
description: Minimisation of convex functions over convex sets, where every local minimum is global, with duality, optimality conditions, standard problem classes and its limited but real role in variational inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - optimization
  - duality
  - convexity
semantic_relations:
  - type: relates
    links:
      - "[[optimization_theory]]"
      - "[[optimization_algorithms]]"
      - "[[variational_inference]]"
      - "[[kl_divergence]]"
      - "[[jensen_inequality]]"
---

# Convex Optimization

## Overview
A convex optimization problem minimises a convex function $f$ over a convex set $C$. A function is convex if $f(\lambda x+(1-\lambda)y)\le\lambda f(x)+(1-\lambda)f(y)$ for all $x,y$ and $\lambda\in[0,1]$. The central fact is that every local minimum is a global minimum, and the set of minimisers is convex. This removes the main difficulty of general nonlinear optimisation and allows algorithms with provable convergence rates.

## Key results and problem classes
- **Optimality conditions**: for differentiable $f$ on an open set, $\nabla f(x^{*})=0$ is necessary and sufficient. With constraints, the Karush-Kuhn-Tucker (KKT) conditions are necessary under a constraint qualification such as Slater's condition and are then also sufficient.
- **Duality**: the Lagrangian dual gives a lower bound on the optimum (weak duality). Under Slater's condition the gap is zero (strong duality), which supports decomposition and certificates of optimality.
- **Problem classes**: linear programs, quadratic programs, second-order cone programs and semidefinite programs form a hierarchy solvable by interior-point methods in polynomial time.
- **First-order methods**: gradient descent with a suitable step size converges at rate $O(1/k)$ for smooth convex functions, and accelerated methods achieve $O(1/k^2)$.

Jensen's inequality is itself a statement of convexity, and it underlies the evidence lower bound used in variational methods.

## Relation to Active Inference
The relationship is a mix of exact results and caveats. The KL divergence $D_{\mathrm{KL}}(q\,\|\,p)$ is convex in $q$ for fixed $p$, so variational free energy $F(q)=D_{\mathrm{KL}}[q(s)\,\|\,p(s\mid o)]-\ln p(o)$ is convex as a function of the belief distribution over the probability simplex. For a mean-field family it is convex in each factor when the others are held fixed, which is why coordinate-wise updates decrease it monotonically. The full problem is generally not convex, because the joint family is not convex and because model parameters enter nonlinearly, so local minima exist, and free energy minimisation in practice carries no global guarantee.

## Related concepts
- [[optimization_theory]] — the broader framework of optimality and constraints
- [[optimization_algorithms]] — the iterative methods used to solve convex problems
- [[variational_inference]] — optimisation over distributions with convex structure
- [[kl_divergence]] — convex in its arguments
- [[jensen_inequality]] — the inequality that characterises convexity

## References
- Boyd, S., & Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press.
- Rockafellar, R. T. (1970). Convex Analysis. Princeton University Press.
- Nesterov, Y. (2018). Lectures on Convex Optimization (2nd ed.). Springer.
