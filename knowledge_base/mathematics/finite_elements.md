---
title: Finite Elements
description: A Galerkin method that solves partial differential equations by approximating their weak form with piecewise polynomial functions on a mesh, with the resulting linear system and the quasi-optimality estimate.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - numerical_methods
  - partial_differential_equations
  - variational_methods
semantic_relations:
  - type: relates
    links:
      - "[[numerical_methods]]"
      - "[[finite_difference]]"
      - "[[differential_equations]]"
      - "[[functional_analysis]]"
      - "[[variational_calculus]]"
---

# Finite Elements

## Overview
The finite element method approximates the solution of a partial differential equation (PDE) by functions that are polynomial on each cell of a mesh. Unlike finite differences, which discretise the differential operator at grid points, it starts from an integral formulation of the problem and so handles irregular domains, varying coefficients and natural boundary conditions with the same machinery.

## Weak form and Galerkin approximation
Take the Poisson problem $-\Delta u=f$ on a domain $\Omega$ with $u=0$ on the boundary. Multiplying by a test function $v$ and integrating by parts gives the weak form: find $u\in V=H_0^1(\Omega)$ such that
$$a(u,v)=\int_\Omega\nabla u\cdot\nabla v\,dx=\int_\Omega f\,v\,dx=\ell(v)\qquad\text{for all }v\in V.$$
The Galerkin method replaces $V$ by a finite-dimensional subspace $V_h$ spanned by basis functions $\varphi_1,\dots,\varphi_N$, typically piecewise linear "hat" functions on a triangulation. Writing $u_h=\sum_j c_j\varphi_j$ yields the linear system $Kc=F$ with stiffness matrix $K_{ij}=a(\varphi_j,\varphi_i)$ and load vector $F_i=\ell(\varphi_i)$. Because each basis function has small support, $K$ is sparse.

## Error estimate
If the bilinear form is bounded with constant $M$ and coercive with constant $\alpha$, Cea's lemma gives
$$\|u-u_h\|_V\le\frac{M}{\alpha}\inf_{v_h\in V_h}\|u-v_h\|_V,$$
so the Galerkin solution is quasi-optimal, and the error is controlled by how well the space $V_h$ can approximate $u$. Refining the mesh or raising the polynomial degree improves the approximation, subject to regularity of the solution.

## Relation to Active Inference
The method has no direct role in the theory. Its relevance is practical: probability densities of continuous stochastic models obey Fokker-Planck equations, and path integral implementations that need to solve such PDEs on irregular domains or in several dimensions may use finite element discretisations. The weak-form viewpoint also parallels variational methods in which an equation is replaced by an optimisation or projection problem, but this is an analogy of technique, not of content.

## Related concepts
- [[finite_difference]] — the grid-based alternative
- [[variational_calculus]] — source of the weak formulation as a minimisation principle
- [[functional_analysis]] — the Sobolev and Hilbert space framework behind the error theory
- [[fokker_planck]] — a PDE for probability densities that may be solved numerically
- [[linear_algebra]] — sparse linear solvers handle the resulting systems

## References
- Brenner, S. C., & Scott, L. R. (2008). The Mathematical Theory of Finite Element Methods (3rd ed.). Springer.
- Ciarlet, P. G. (2002). The Finite Element Method for Elliptic Problems. SIAM.
- Hughes, T. J. R. (2000). The Finite Element Method: Linear Static and Dynamic Finite Element Analysis. Dover.
