---
title: Cumulant Expansion
description: The series expansion of the logarithm of a moment generating function in cumulants, with the first four cumulants, additivity, the Gaussian truncation, and use in statistical physics and approximate inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - probability_theory
  - statistical_physics
  - approximation
semantic_relations:
  - type: relates
    links:
      - "[[generating_functional]]"
      - "[[moment_matching]]"
      - "[[gaussian_distributions]]"
      - "[[statistical_physics]]"
      - "[[jarzynski_equality]]"
      - "[[laplace_approximation]]"
---

# Cumulant Expansion

## Overview
Cumulants are quantities derived from the moments of a distribution that capture its shape in an additive way. The cumulant expansion writes the logarithm of the moment generating function as a power series whose coefficients are the cumulants, and it is used to approximate averages of exponentials, which arise constantly in statistical physics and inference.

## Definition
For a random variable $X$ with moment generating function $M(t)=\mathbb{E}[e^{tX}]$, the cumulant generating function is
$$K(t)=\ln M(t)=\sum_{n\ge 1}\kappa_n\frac{t^n}{n!}.$$
The first cumulants are the mean $\kappa_1=\mu$, the variance $\kappa_2=\sigma^2$, the third central moment $\kappa_3$, and $\kappa_4=\mu_4-3\sigma^4$, the excess fourth moment. Cumulants of sums of independent variables add, which moments do not, and cumulants of order higher than two measure departure from Gaussian behaviour. A Gaussian has all cumulants above the second equal to zero, and this characterises it, so truncating the expansion at second order is equivalent to a Gaussian approximation. The expansion is generally asymptotic and need not converge.

## Use in physics and inference
In statistical physics, the free energy is $-\beta^{-1}\ln Z$, and $\ln Z$ is a cumulant generating function of the energy, so expansions in cumulants give high-temperature series and perturbative corrections. In non-equilibrium thermodynamics, truncating at second order for a Gaussian work distribution gives $\Delta F\approx\langle W\rangle-\tfrac{\beta}{2}\mathrm{Var}(W)$, an approximation to the Jarzynski equality. In approximate inference the cumulants of a posterior are a compact summary, and methods that fit a mean and covariance are second-order closures (see [[moment_matching]]).

## Relation to Active Inference
The link is formal and indirect. Variational free energy under the Laplace assumption is a second-order approximation of the log joint density, so it is effectively a truncation of this kind. Cumulant methods also appear in the field-theoretic and path-integral formulations of the free energy principle, where connected correlation functions are generated from a functional (see [[generating_functional]]). These uses are borrowed from physics and form part of a research programme, not of standard Active Inference practice.

## Related concepts
- [[generating_functional]] — the functional analogue for fields and paths
- [[moment_matching]] — fitting the first moments of an approximating distribution
- [[gaussian_distributions]] — the distributions with no cumulants above second order
- [[jarzynski_equality]] — an exponential average that is approximated by cumulants
- [[statistical_physics]] — the setting where cumulants of the energy are central
- [[perturbation_theory]] — systematic corrections in a small parameter

## References
- Kardar, M. (2007). Statistical Physics of Particles. Cambridge University Press.
- van Kampen, N. G. (2007). Stochastic Processes in Physics and Chemistry (3rd ed.). North-Holland.
- Billingsley, P. (1995). Probability and Measure (3rd ed.). Wiley.
