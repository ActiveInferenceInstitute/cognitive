---
title: Integration Theory
description: "Riemann and Lebesgue integration, convergence theorems, Fubini and Radon-Nikodym, and why expectations, densities and KL divergences in Active Inference are Lebesgue integrals."
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - mathematics
  - analysis
  - measure_theory
semantic_relations:
  - type: relates
    links:
      - "[[measure_theory]]"
      - "[[measure_theoretic_active_inference]]"
      - "[[probability_theory]]"
      - "[[real_analysis]]"
---

# Integration Theory

## Overview
Integration theory studies how to assign a number $\int f\,d\mu$ to a function $f$ with respect to a measure $\mu$. The Riemann integral suffices for piecewise continuous functions on intervals; the Lebesgue integral, built on [[measure_theory]], is the version in which probability theory is written. Every expectation, marginalisation and free-energy functional in Active Inference is an integral of this kind (a sum in the discrete case).

## Definition and main results
**Construction.** For a simple function $s=\sum_i c_i \mathbf 1_{A_i}$ with $c_i\ge 0$, set $\int s\,d\mu=\sum_i c_i\,\mu(A_i)$. For measurable $f\ge 0$ define $\int f\,d\mu=\sup\int s\,d\mu$ over simple $0\le s\le f$, and for general $f$ split into positive and negative parts. A function is integrable when $\int |f|\,d\mu<\infty$. Riemann partitions the domain; Lebesgue partitions the range, which is why it handles highly discontinuous functions and limits.

**Convergence theorems.** Monotone convergence (increasing nonnegative sequences), Fatou's lemma, and dominated convergence (pointwise convergence plus an integrable bound) justify exchanging limits and integrals.

**Fubini and Tonelli.** For $\sigma$-finite measures, an integral over a product space can be computed iteratively in either order (Tonelli for $f\ge 0$, Fubini when $f$ is integrable). This licenses moving from a joint $p(o,s)$ to marginals.

**Radon-Nikodym.** If $\nu\ll\mu$ (both $\sigma$-finite), there is a density $d\nu/d\mu$ with $\nu(A)=\int_A (d\nu/d\mu)\,d\mu$. Probability densities are such derivatives, and the KL divergence is the integral $D_{KL}(P\|Q)=\int \ln\frac{dP}{dQ}\,dP$, finite only when $P\ll Q$.

## Role in Active Inference
The role is as background language. Writing $F=\mathbb E_q[\ln q-\ln p]$ as a Lebesgue integral makes precise when it exists, when conditional expectations are well defined, and why absolute continuity matters for the free energy bound. [[measure_theoretic_active_inference]] develops this in detail. One limit deserves stating: there is no Lebesgue measure on infinite-dimensional path spaces, so the path integrals used in [[path_integral_free_energy]] are defined either with Wiener measure (see [[stochastic_processes]]) or heuristically, as a formal notation.

## Related concepts
- [[measure_theory]] — supplies the measures and $\sigma$-algebras that integration needs
- [[probability_theory]] — expectation is the integral against a probability measure
- [[kl_divergence]] — defined through the Radon-Nikodym derivative
- [[real_analysis]] — Riemann integration and limit theorems
- [[functional_analysis]] — $L^p$ spaces are built from integrable functions
