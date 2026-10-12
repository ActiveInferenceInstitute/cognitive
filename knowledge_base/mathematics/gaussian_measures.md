---
title: Gaussian Measures
description: Probability measures whose linear functionals are normally distributed, from finite-dimensional normal laws to Wiener measure on path space, with the Cameron-Martin theorem on shifts.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - measure_theory
  - gaussian
  - stochastic_processes
semantic_relations:
  - type: relates
    links:
      - "[[wiener_process]]"
      - "[[measure_theory]]"
      - "[[gaussian_distributions]]"
      - "[[gaussian_processes]]"
      - "[[path_integral]]"
---

# Gaussian Measures

## Overview
A Gaussian measure is a probability measure for which every continuous linear functional has a normal distribution, possibly degenerate. On $\mathbb{R}^n$ these are the multivariate normal laws $\mathcal{N}(m,\Sigma)$. When $\Sigma$ is positive definite the measure has the density
$$\frac{1}{(2\pi)^{n/2}\sqrt{\det\Sigma}}\exp\!\left(-\tfrac12(x-m)^\top\Sigma^{-1}(x-m)\right)$$
with respect to Lebesgue measure. The definition through functionals is what extends to infinite dimensions, where there is no Lebesgue measure and densities of this form make no sense.

## Infinite-dimensional case
On a separable Banach or Hilbert space a Gaussian measure is determined by its mean and a covariance operator. On a Hilbert space the covariance operator must be positive, self-adjoint and of trace class, so a standard Gaussian cannot be placed on an infinite-dimensional Hilbert space as an isotropic "white" measure. The Wiener measure on the space of continuous functions $C[0,1]$ is the law of Brownian motion, with mean zero and covariance $\mathrm{Cov}(B_s,B_t)=\min(s,t)$. Gaussian processes are Gaussian measures on function spaces.

## Cameron-Martin theorem
Let $H$ be the Cameron-Martin space of Wiener measure, the absolutely continuous functions $h$ with $h(0)=0$ and $h'\in L^2$. Translating Wiener measure by $h\in H$ gives an equivalent measure whose density is
$$\exp\!\left(\int_0^1 h'(t)\,dB_t-\tfrac12\int_0^1 h'(t)^2\,dt\right).$$
Translation by a path outside $H$ gives a measure mutually singular with the original (a special case of the Feldman-Hajek dichotomy). Girsanov's change of measure for stochastic differential equations extends the theorem, and it is the rigorous counterpart of the formal quadratic action $\tfrac12\int\dot x^2dt$ in path integrals for diffusion.

## Relation to Active Inference
Gaussian measures matter in two ways. Under the Laplace approximation, beliefs and noise in continuous-state Active Inference models are Gaussian. The path integral formulation of the free energy principle treats trajectories of stochastic dynamics as draws from a measure on path space, and the Wiener measure provides the rigorous version of the noise part of that picture. Both uses are standard mathematics rather than substantive claims.

## Related concepts
- [[wiener_process]] — the process whose law is Wiener measure
- [[gaussian_distributions]] — finite-dimensional Gaussian laws
- [[gaussian_processes]] — Gaussian measures on spaces of functions
- [[kolmogorov_extension]] — construction of measures on infinite products
- [[hilbert_spaces]] — the setting for covariance operators

## References
- Bogachev, V. I. (1998). Gaussian Measures. American Mathematical Society.
- Oksendal, B. (2003). Stochastic Differential Equations: An Introduction with Applications (6th ed.). Springer.
- Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press.
