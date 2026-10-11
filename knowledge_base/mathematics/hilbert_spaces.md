---
title: Hilbert Spaces
description: Complete inner product spaces, their geometry of orthogonality and projection, and their use in Fourier analysis, quantum theory and kernel methods, with an assessment of their role in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - functional_analysis
  - inner_product
  - orthogonality
semantic_relations:
  - type: relates
    links:
      - "[[banach_spaces]]"
      - "[[functional_analysis]]"
      - "[[linear_algebra]]"
      - "[[quantum_mechanics]]"
      - "[[gaussian_processes]]"
---

# Hilbert Spaces

## Overview
A Hilbert space is a vector space $H$ with an inner product $\langle\cdot,\cdot\rangle$ that is complete with respect to the norm $\|x\|=\sqrt{\langle x,x\rangle}$. It is the infinite-dimensional space that most closely preserves Euclidean geometry: lengths, angles, orthogonality and projections all behave as they do in $\mathbb{R}^n$. Every Hilbert space is a [[banach_spaces|Banach space]], but not conversely.

## Structure
The inner product is linear in one argument, conjugate symmetric and positive definite. It yields the Cauchy-Schwarz inequality $|\langle x,y\rangle|\le\|x\|\|y\|$ and the parallelogram law, which characterises norms arising from inner products. Standard examples are $\mathbb{C}^n$, the sequence space $\ell^2$ and $L^2(\mu)$, the square-integrable functions.

Key results include the following.
- **Projection theorem.** For a closed subspace $M$, every $x$ decomposes uniquely as $x=m+m^\perp$ with $m\in M$ and $m^\perp\in M^\perp$, and $m$ is the closest point of $M$ to $x$.
- **Riesz representation.** Every bounded linear functional has the form $x\mapsto\langle x,y\rangle$ for a unique $y$.
- **Orthonormal bases.** In a separable infinite-dimensional space there is a countable orthonormal basis $\{e_k\}$ with $x=\sum_k\langle x,e_k\rangle e_k$ and Parseval's identity $\|x\|^2=\sum_k|\langle x,e_k\rangle|^2$. Such a space is isometrically isomorphic to $\ell^2$ (a finite-dimensional one, to $\mathbb{C}^n$).

## Uses
Fourier series are expansions in an orthonormal basis of $L^2$. Quantum mechanics represents states as unit vectors, and observables as self-adjoint operators, on a Hilbert space (see [[quantum_mechanics]]). Reproducing kernel Hilbert spaces underlie kernel methods and the covariance structure of [[gaussian_processes]], and least-squares estimation is orthogonal projection.

## Relation to Active Inference
The connection is mostly as supporting mathematics. Linear-Gaussian inference, kernel-based function approximation and spectral methods all use Hilbert-space geometry, and an $L^2$ structure is often placed on spaces of functions of time. Quantum-cognition models also use Hilbert spaces, but that is a separate modelling choice, and whether it adds explanatory power over classical probability is contested. Standard discrete and continuous Active Inference formulations are written with probability simplices and Gaussian densities rather than Hilbert-space operators.

## Related concepts
- [[banach_spaces]] — complete normed spaces, of which Hilbert spaces are a special case
- [[functional_analysis]] — the field in which both are studied
- [[linear_algebra]] — the finite-dimensional case
- [[quantum_mechanics]] — state spaces and operators
- [[gaussian_processes]] — kernels and function-space inference

## References
- Reed, M., & Simon, B. (1980). Methods of Modern Mathematical Physics I: Functional Analysis (revised ed.). Academic Press.
- Rudin, W. (1987). Real and Complex Analysis (3rd ed.). McGraw-Hill.
- Scholkopf, B., & Smola, A. J. (2002). Learning with Kernels. MIT Press.
