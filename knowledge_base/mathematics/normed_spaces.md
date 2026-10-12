---
title: Normed Spaces
description: Vector spaces equipped with a norm measuring vector length, their induced metric topology, standard examples, and the passage to Banach and Hilbert spaces.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - functional_analysis
  - linear_algebra
  - topology
semantic_relations:
  - type: relates
    links:
      - "[[banach_spaces]]"
      - "[[hilbert_spaces]]"
      - "[[inner_products]]"
      - "[[functional_analysis]]"
      - "[[topological_spaces]]"
---

# Normed Spaces

## Overview
A normed space is a vector space $V$ over $\mathbb{R}$ or $\mathbb{C}$ together with a function $\|\cdot\|:V\to[0,\infty)$ satisfying three axioms: $\|x\|=0$ if and only if $x=0$; $\|\alpha x\|=|\alpha|\,\|x\|$ for every scalar $\alpha$; and the triangle inequality $\|x+y\|\le\|x\|+\|y\|$. The norm gives a notion of size and, through $d(x,y)=\|x-y\|$, a metric that makes $V$ a topological space in which addition and scalar multiplication are continuous.

## Examples and properties
- **Euclidean and $\ell^p$ norms.** On $\mathbb{R}^n$ the $p$-norm is $\|x\|_p=(\sum_i|x_i|^p)^{1/p}$ for $1\le p<\infty$, and $\|x\|_\infty=\max_i|x_i|$. The triangle inequality for $p$-norms is Minkowski's inequality.
- **Function spaces.** $L^p(\mu)$ with $\|f\|_p=(\int|f|^p\,d\mu)^{1/p}$, after identifying functions equal almost everywhere, and $C([a,b])$ with the supremum norm.
- **Equivalence in finite dimension.** All norms on a finite-dimensional space are equivalent, so they define the same convergent sequences. This fails in infinite dimension, where the choice of norm matters.
- **Bounded operators.** A linear map between normed spaces is continuous if and only if it is bounded, meaning $\|Tx\|\le C\|x\|$ for some $C$. The smallest such constant is the operator norm $\|T\|$.
- **Inner product spaces.** A norm comes from an inner product exactly when it satisfies the parallelogram law $\|x+y\|^2+\|x-y\|^2=2\|x\|^2+2\|y\|^2$, in which case $\|x\|=\sqrt{\langle x,x\rangle}$.

A normed space in which every Cauchy sequence converges is complete and is called a [[banach_spaces|Banach space]]; a complete inner product space is a [[hilbert_spaces|Hilbert space]].

## Relation to Active Inference
Normed spaces provide the setting for much of the analysis used in the vault, but the connection is foundational rather than specific. Convergence of iterative inference schemes, contraction arguments ([[contraction]]), and the function spaces on which operators such as the Fokker-Planck generator act all presuppose a choice of norm. In the [[free_energy_principle]] literature, densities are often treated as elements of $L^1$ or $L^2$, and divergences such as [[kl_divergence]] are not norms, since they are not symmetric and violate the triangle inequality. The distinction between a norm-induced metric and an information-geometric one is addressed in [[information_geometry]].

## Related concepts
- [[banach_spaces]] — complete normed spaces
- [[hilbert_spaces]] — complete spaces with an inner product
- [[inner_products]] — structure that induces a norm
- [[functional_analysis]] — study of operators on normed spaces
- [[topological_spaces]] — the topology a norm induces
- [[contraction]] — maps that shrink distance in a norm

## References
- Kreyszig, E. (1978). Introductory Functional Analysis with Applications. Wiley.
- Rudin, W. (1991). Functional Analysis (2nd ed.). McGraw-Hill.
