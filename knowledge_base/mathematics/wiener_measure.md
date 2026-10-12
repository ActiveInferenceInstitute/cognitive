---
title: Wiener Measure
description: The probability measure on continuous paths that defines Brownian motion, with its construction, path properties, Cameron-Martin space and role as the rigorous basis of path integrals and the Feynman-Kac formula.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - measure_theory
  - stochastic_processes
  - path_integrals
semantic_relations:
  - type: relates
    links:
      - "[[wiener_process]]"
      - "[[brownian_motion]]"
      - "[[path_integral_theory]]"
      - "[[measure_theory]]"
      - "[[stochastic_calculus]]"
      - "[[fokker_planck]]"
wikidata: Q124251629
---

# Wiener Measure

## Overview
The Wiener measure $\mathbb{W}$ is a probability measure on the space $C_0[0,\infty)$ of continuous real functions $w$ with $w(0)=0$, under which the coordinate process $W_t(w)=w(t)$ is standard Brownian motion (see [[wiener_process]]). It was constructed by Norbert Wiener in 1923 and is the prototypical example of a measure on an infinite-dimensional path space.

## Definition
$\mathbb{W}$ is determined by its finite-dimensional distributions: for times $0<t_1<\dots<t_n$ the increments $W_{t_1},W_{t_2}-W_{t_1},\dots$ are independent with $W_{t_k}-W_{t_{k-1}}\sim\mathcal{N}(0,t_k-t_{k-1})$. The probability that a path lies in a set of cylinder type is then
$$\mathbb{W}\bigl(w(t_1)\in A_1,\dots,w(t_n)\in A_n\bigr)=\int_{A_1\times\dots\times A_n}\prod_{k=1}^{n}\frac{e^{-(x_k-x_{k-1})^2/2(t_k-t_{k-1})}}{\sqrt{2\pi(t_k-t_{k-1})}}\,dx_1\cdots dx_n,\quad x_0=0.$$
Continuity of the paths follows from the Kolmogorov continuity criterion.

## Path properties
Almost every path is continuous, nowhere differentiable, and locally Hölder continuous of every exponent below $1/2$. In particular the paths almost surely have infinite total variation, so $\int\dot w^2\,dt$ is infinite and the formal expression $\exp(-\tfrac12\int\dot w^2\,dt)\,\mathcal{D}w$ is a heuristic, since no Lebesgue measure exists on infinite-dimensional spaces. The Cameron-Martin space $H=\{h:h(0)=0,\ \dot h\in L^2\}$ has Wiener measure zero, yet translating the measure by $h\in H$ produces an equivalent measure with an explicit density; this is the content of the Cameron-Martin and Girsanov theorems.

## Feynman-Kac formula
Expectations under $\mathbb{W}$ solve partial differential equations. For a suitably regular potential $V$ and initial function $f$, the function
$$u(t,x)=\mathbb{E}\!\left[f(x+W_t)\exp\!\Bigl(\int_0^t V(x+W_s)\,ds\Bigr)\right]$$
solves $\partial_t u=\tfrac12\partial_x^2u+Vu$ with $u(0,\cdot)=f$. This gives path integrals over Brownian paths a precise meaning in the imaginary-time case and links diffusion to the heat equation.

## Relation to Active Inference
Path-integral formulations of stochastic control and of the free energy principle consider the probability of trajectories of Langevin dynamics, and the Wiener measure serves as the reference measure on noise paths in those constructions. The role is mathematical foundation: it justifies the formal path densities used in [[path_integral_theory]]. Whether it yields new empirical predictions for biological systems is a separate matter that the measure itself does not settle.

## Related concepts
- [[wiener_process]] — the process whose law is the Wiener measure
- [[brownian_motion]] — the physical phenomenon being idealised
- [[path_integral_theory]] — where formal densities over paths are applied
- [[stochastic_calculus]] — Itô integration against Wiener paths
- [[measure_theory]] — the framework needed for infinite-dimensional measures

## References
- Wiener, N. (1923). Differential space. Journal of Mathematics and Physics, 2, 131-174.
- Kac, M. (1949). On distributions of certain Wiener functionals. Transactions of the American Mathematical Society, 65, 1-13.
- Karatzas, I., & Shreve, S. E. (1991). Brownian Motion and Stochastic Calculus (2nd ed.). Springer.
- Mörters, P., & Peres, Y. (2010). Brownian Motion. Cambridge University Press.
