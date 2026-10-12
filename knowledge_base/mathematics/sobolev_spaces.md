---
title: Sobolev Spaces
description: Function spaces of functions with weak derivatives in Lebesgue spaces, their norms, embedding theorems and their role in partial differential equations and path-space analysis.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - functional_analysis
  - weak_derivatives
  - pde
semantic_relations:
  - type: relates
    links:
      - "[[functional_analysis]]"
      - "[[hilbert_spaces]]"
      - "[[banach_spaces]]"
      - "[[lebesgue_integration]]"
      - "[[differential_equations]]"
---

# Sobolev Spaces

## Overview
Sobolev spaces collect functions that are not necessarily differentiable in the classical sense but have derivatives in a weak sense that are integrable to a given power. They are the standard setting for the modern theory of partial differential equations and for variational problems, because they are complete and because minimisers of energy functionals typically exist in them and not in spaces of smooth functions.

## Definition
Let $\Omega\subseteq\mathbb{R}^n$ be open. A function $u\in L^p(\Omega)$ has weak derivative $D^\alpha u=v\in L^p(\Omega)$ if $\int_\Omega u\,D^\alpha\varphi\,dx=(-1)^{|\alpha|}\int_\Omega v\,\varphi\,dx$ for every smooth, compactly supported test function $\varphi$. For an integer $k\ge0$ and $1\le p\le\infty$, the Sobolev space $W^{k,p}(\Omega)$ contains the functions whose weak derivatives up to order $k$ lie in $L^p$, with norm
$$\|u\|_{W^{k,p}}=\Big(\sum_{|\alpha|\le k}\|D^\alpha u\|_{L^p}^p\Big)^{1/p}$$
for $p<\infty$. These are Banach spaces, and $H^k=W^{k,2}$ is a Hilbert space with inner product $\sum_{|\alpha|\le k}\langle D^\alpha u,D^\alpha v\rangle_{L^2}$ ([[banach_spaces]], [[hilbert_spaces]]). The space $H^1_0(\Omega)$ is the closure of smooth compactly supported functions and encodes zero boundary values.

## Embeddings and compactness
Sobolev embedding theorems relate smoothness to integrability. On $\mathbb{R}^n$, if $kp<n$ then $W^{k,p}$ embeds continuously in $L^q$ with $\frac1q=\frac1p-\frac kn$. If $kp>n$, its elements have continuous representatives. For bounded domains with regular boundary, the Rellich-Kondrachov theorem states that the embedding of $W^{1,p}$ into $L^q$ is compact for $q$ below the critical exponent. This compactness is the usual tool for proving the existence of weak solutions of elliptic equations such as $-\Delta u=f$, via the Lax-Milgram theorem (Evans, 2010).

## Role in path-space analysis
In stochastic analysis, the Cameron-Martin space of Wiener measure is the Sobolev-type space of paths $h$ with $h(0)=0$ and $\int_0^1|\dot h(t)|^2\,dt<\infty$, identified with $H^1$ on $[0,1]$. Typical Brownian paths are not in this space, since they are almost surely nowhere differentiable. The energy $\tfrac12\int|\dot h|^2dt$ is the action that appears in formal path-integral expressions ([[path_integral_theory]], [[wiener_process]]).

## Relation to Active Inference
The connection is mathematical background only. Rigorous treatments of densities evolving under a Fokker-Planck equation, or of gradient flows on spaces of measures, rely on Sobolev-type function spaces, and the weak formulation of [[finite_elements]] is built on them. The Active Inference literature itself seldom uses them explicitly, and its discrete-state models do not need them.

## Related concepts
- [[functional_analysis]] — the abstract framework
- [[hilbert_spaces]] — the $H^k$ spaces are examples
- [[lebesgue_integration]] — the $L^p$ spaces underlying the definition
- [[differential_equations]] — the main application area
- [[variational_calculus]] — minimisation of energy functionals
- [[finite_elements]] — numerical methods built on weak formulations

## References
- Evans, L. C. (2010). Partial Differential Equations (2nd ed.). American Mathematical Society.
- Adams, R. A., & Fournier, J. J. F. (2003). Sobolev Spaces (2nd ed.). Academic Press.
- Brezis, H. (2011). Functional Analysis, Sobolev Spaces and Partial Differential Equations. Springer.
