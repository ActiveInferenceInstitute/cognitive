---
title: Inner Products
description: Bilinear or sesquilinear forms that define length, angle and orthogonality in vector spaces, with the axioms, Cauchy-Schwarz inequality, standard examples and their role in precision-weighted prediction errors.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - linear_algebra
  - functional_analysis
  - geometry
semantic_relations:
  - type: relates
    links:
      - "[[hilbert_spaces]]"
      - "[[normed_spaces]]"
      - "[[linear_algebra]]"
      - "[[metric_tensor]]"
      - "[[functional_analysis]]"
---

# Inner Products

## Overview
An inner product on a real or complex vector space $V$ is a function $\langle\cdot,\cdot\rangle:V\times V\to\mathbb{F}$ that adds geometry to the space. It supplies lengths, angles and orthogonality, and so makes projection and least-squares approximation possible.

## Axioms and consequences
For all $x,y,z\in V$ and scalars $a,b$, an inner product satisfies:
1. conjugate symmetry, $\langle x,y\rangle=\overline{\langle y,x\rangle}$ (plain symmetry for real spaces);
2. linearity in the first argument, $\langle ax+by,z\rangle=a\langle x,z\rangle+b\langle y,z\rangle$;
3. positive definiteness, $\langle x,x\rangle>0$ for $x\neq0$.

Some authors place linearity in the second argument instead. The inner product induces the norm $\|x\|=\sqrt{\langle x,x\rangle}$, and two vectors are orthogonal when $\langle x,y\rangle=0$. The Cauchy-Schwarz inequality $|\langle x,y\rangle|\le\|x\|\,\|y\|$ holds, which makes the angle $\theta=\arccos(\langle x,y\rangle/\|x\|\|y\|)$ well defined in real spaces. The induced norm obeys the triangle inequality and the parallelogram law, and a norm comes from an inner product exactly when it obeys the latter. A complete inner product space is a Hilbert space.

## Examples
- The dot product $\langle x,y\rangle=x^\top y$ on $\mathbb{R}^n$.
- The weighted form $\langle x,y\rangle_G=x^\top Gy$ for a symmetric positive-definite matrix $G$.
- The $L^2$ product $\langle f,g\rangle=\int f\,\bar g\,d\mu$ on square-integrable functions.
- The Frobenius product $\langle A,B\rangle=\mathrm{tr}(B^\dagger A)$ on matrices.

A Riemannian metric is a smoothly varying choice of inner product on each tangent space, so the weighted form above is the metric tensor at a point.

## Relation to Active Inference
In continuous-state models with Gaussian assumptions, free energy contains terms $\tfrac12\varepsilon^\top\Pi\,\varepsilon$, the squared norm of a prediction error $\varepsilon$ under the inner product defined by the precision matrix $\Pi$, which is the inverse covariance. Precision weighting is therefore a choice of inner product that sets how errors are measured. The Fisher-Rao metric is the inner product on tangent vectors of belief space. This is a direct use of the mathematics, not a claim about neural mechanisms.

## Related concepts
- [[hilbert_spaces]] — complete spaces with an inner product
- [[normed_spaces]] — spaces with length but not necessarily angle
- [[metric_tensor]] — a position-dependent inner product on tangent spaces
- [[precision_parameter]] — the weighting of prediction errors in Active Inference
- [[gaussian_distributions]] — whose exponent is a squared norm under the inverse covariance

## References
- Axler, S. (2015). Linear Algebra Done Right (3rd ed.). Springer.
- Horn, R. A., & Johnson, C. R. (2013). Matrix Analysis (2nd ed.). Cambridge University Press.
- Reed, M., & Simon, B. (1980). Methods of Modern Mathematical Physics I: Functional Analysis (revised and enlarged ed.). Academic Press.
