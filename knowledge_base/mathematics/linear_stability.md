---
title: Linear Stability
description: Analysis of the stability of an equilibrium by linearising the dynamics and examining eigenvalues of the Jacobian, with classification in two dimensions, the Hartman-Grobman theorem and its limits.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - dynamical_systems
  - stability
  - eigenvalues
semantic_relations:
  - type: relates
    links:
      - "[[stability_analysis]]"
      - "[[lyapunov_stability]]"
      - "[[bifurcation_theory]]"
      - "[[dynamical_systems]]"
      - "[[linear_algebra]]"
wikidata: Q17098945
---

# Linear Stability

## Overview
For an autonomous system $\dot x=f(x)$ with an equilibrium $x^*$, where $f(x^*)=0$, linear stability analysis asks whether small perturbations die out or grow. Writing $x=x^*+\delta$ and expanding gives, to first order,
$$\dot\delta=J\,\delta,\qquad J=Df(x^*)=\left(\frac{\partial f_i}{\partial x_j}\right)_{x^*}.$$
The solution is $\delta(t)=e^{Jt}\delta(0)$, so the behaviour is governed by the eigenvalues $\lambda_k$ of the Jacobian $J$.

## Criteria
- If every eigenvalue has $\mathrm{Re}\,\lambda_k<0$, the equilibrium is asymptotically stable.
- If some eigenvalue has $\mathrm{Re}\,\lambda_k>0$, it is unstable.
- If the largest real part is zero, the linearisation is inconclusive and nonlinear terms decide. Centre manifold theory reduces the problem to the dynamics along the neutral directions.

For a map $x_{n+1}=F(x_n)$ the corresponding condition is that all eigenvalues of $DF(x^*)$ lie inside the unit circle. In two dimensions, with trace $\tau$ and determinant $\Delta$ of $J$, the equilibrium is a saddle if $\Delta<0$, and for $\Delta>0$ it is stable when $\tau<0$ and unstable when $\tau>0$. Whether it is a node or a spiral depends on the sign of $\tau^2-4\Delta$, spirals having complex eigenvalues. The case $\tau=0,\Delta>0$ is a centre of the linear system.

## Validity
The Hartman-Grobman theorem states that near a hyperbolic equilibrium, one with no eigenvalue on the imaginary axis, the nonlinear flow is topologically conjugate to its linearisation, so the stability conclusion carries over. Linear analysis is local: it gives no information on the size of the basin of attraction, for which Lyapunov functions are used. When a parameter varies and an eigenvalue crosses the imaginary axis, the equilibrium changes stability and a bifurcation occurs, as in saddle-node and Hopf bifurcations.

## Relation to Active Inference
For a gradient flow $\dot x=-\nabla F(x)$ on a free energy, the Jacobian at a critical point is minus the Hessian, so non-degenerate local minima of $F$ (positive-definite Hessian) are linearly stable. This ties the stability of perceptual inference to the curvature of free energy, and stability of the Laplace-approximated posterior is analysed in the same way. Linear stability is also used to study the dynamics around steady states and to discuss critical slowing down. These are applications of standard dynamical systems theory.

## Related concepts
- [[lyapunov_stability]] — a global alternative to linearisation
- [[bifurcation_theory]] — qualitative changes when eigenvalues cross the imaginary axis
- [[hopf_bifurcation]] — loss of stability through a complex eigenvalue pair
- [[structural_stability]] — robustness of qualitative behaviour to perturbation
- [[perturbation_theory]] — approximate analysis of small deviations

## References
- Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos (2nd ed.). Westview Press.
- Hirsch, M. W., Smale, S., & Devaney, R. L. (2013). Differential Equations, Dynamical Systems, and an Introduction to Chaos (3rd ed.). Academic Press.
- Khalil, H. K. (2002). Nonlinear Systems (3rd ed.). Prentice Hall.
