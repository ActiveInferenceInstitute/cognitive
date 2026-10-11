---
title: Banach Spaces
description: Complete normed vector spaces, their standard examples and main theorems, and the limited but real roles they play in function-space arguments, contraction mappings and path-space treatments of dynamics.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - functional_analysis
  - normed_spaces
  - function_spaces
semantic_relations:
  - type: relates
    links:
      - "[[hilbert_spaces]]"
      - "[[functional_analysis]]"
      - "[[linear_algebra]]"
      - "[[path_integral_theory]]"
      - "[[markov_decision_process]]"
---

# Banach Spaces

## Overview
A Banach space is a vector space $X$ over $\mathbb{R}$ or $\mathbb{C}$ equipped with a norm $\|\cdot\|$ for which the space is complete: every Cauchy sequence converges to an element of $X$. Named after Stefan Banach, these spaces are the basic setting of functional analysis. They extend finite-dimensional linear algebra to infinite-dimensional spaces of sequences and functions while retaining a usable notion of limit.

## Definition and examples
A norm satisfies $\|x\|\ge 0$ with equality only for $x=0$, homogeneity $\|\alpha x\|=|\alpha|\,\|x\|$, and the triangle inequality $\|x+y\|\le\|x\|+\|y\|$. Completeness is the extra requirement that makes limiting arguments valid.

- **Finite-dimensional spaces.** $\mathbb{R}^n$ with any norm is complete, and all norms on it are equivalent.
- **Sequence and Lebesgue spaces.** $\ell^p$ and $L^p(\mu)$ for $1\le p\le\infty$ are Banach spaces, completeness following from the Riesz-Fischer theorem. Only $p=2$ has a norm that comes from an inner product, which makes it a Hilbert space (see [[hilbert_spaces]]).
- **Continuous functions.** $C(K)$, the continuous functions on a compact set $K$, with the supremum norm $\|f\|_\infty=\sup_{t\in K}|f(t)|$.

## Main results
The bounded linear operators from a normed space $X$ to a Banach space $Y$ themselves form a Banach space, and the dual space $X^*$ of bounded linear functionals is always one. Four theorems carry much of the theory: Hahn-Banach (extension of functionals), the open mapping theorem, the closed graph theorem and the uniform boundedness principle. The Banach fixed-point theorem states that a contraction $T$ on a complete metric space has a unique fixed point, reached by iterating $T$ from any starting point at a geometric rate.

## Relation to Active Inference
The link is infrastructural rather than specific. The fixed-point theorem justifies the convergence of value iteration for a discounted [[markov_decision_process]], where the Bellman operator contracts in the supremum norm. Path-integral treatments of stochastic dynamics need a space of trajectories, such as $C([0,T])$ with the supremum norm, on which Wiener measure is defined (see [[path_integral_theory]]). Beliefs are usually handled as finite-dimensional parameter vectors, so Banach-space structure seldom appears explicitly in Active Inference models. It matters mainly when arguments are made about infinite-dimensional belief or function spaces.

## Related concepts
- [[hilbert_spaces]] — the special case in which the norm comes from an inner product
- [[functional_analysis]] — the field in which Banach spaces are studied
- [[linear_algebra]] — the finite-dimensional theory that Banach spaces generalise
- [[markov_decision_process]] — contraction arguments for value iteration
- [[path_integral_theory]] — function spaces of trajectories

## References
- Rudin, W. (1991). Functional Analysis (2nd ed.). McGraw-Hill.
- Kreyszig, E. (1978). Introductory Functional Analysis with Applications. Wiley.
- Banach, S. (1932). Theorie des operations lineaires. Monografje Matematyczne.
