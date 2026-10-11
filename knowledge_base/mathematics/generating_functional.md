---
title: Generating Functional
description: A functional of a source field from which correlation functions or cumulants are obtained by functional differentiation, central to statistical field theory and path-integral treatments of stochastic dynamics.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - field_theory
  - path_integrals
  - statistical_physics
semantic_relations:
  - type: relates
    links:
      - "[[path_integral]]"
      - "[[path_integral_theory]]"
      - "[[cumulant_expansion]]"
      - "[[field_theory]]"
      - "[[partition_function]]"
      - "[[stochastic_processes]]"
---

# Generating Functional

## Overview
A generating functional is a function of an auxiliary source $J$ that packages all moments or correlation functions of a random field. Differentiating it with respect to the source yields the quantities of interest. It is the infinite-dimensional version of the moment generating function and is the main bookkeeping device of statistical and quantum field theory.

## Definition
For a field $\phi$ with probability weight $e^{-S[\phi]}$, the generating functional is
$$Z[J]=\int \mathcal{D}\phi\;e^{-S[\phi]+\int J(x)\phi(x)\,dx},$$
and the $n$-point correlation function is a functional derivative at zero source,
$$\langle\phi(x_1)\cdots\phi(x_n)\rangle=\frac{1}{Z[0]}\left.\frac{\delta^n Z[J]}{\delta J(x_1)\cdots\delta J(x_n)}\right|_{J=0}.$$
Taking the logarithm, $W[J]=\ln Z[J]$ generates connected correlation functions, which are the cumulants of the field. A Legendre transform of $W$ gives the effective action, which organises corrections to the mean-field description. For a Gaussian action, $Z[J]$ is the exponential of a quadratic form in $J$, and interactions are handled by perturbation expansions.

## Stochastic dynamics
For a stochastic differential equation, one can write the probability of a trajectory as a path integral, introducing a response field in the Martin-Siggia-Rose or Janssen-De Dominicis construction. The generating functional of the trajectory then yields moments and response functions of the dynamics, and expansions in the noise amplitude give systematic approximations. The measure-theoretic status of these integrals requires care (see [[kolmogorov_extension]]).

## Relation to Active Inference
Path-integral and field-theoretic formulations of the free energy principle use generating functionals to compute averages over trajectories of beliefs and states and to build effective descriptions at coarser scales. This is an active research programme and uses borrowed techniques, and it does not form part of the discrete-state schemes used in most applications of Active Inference. Where it is used, the functional is a computational device for correlation functions, with no additional empirical commitment of its own.

## Related concepts
- [[path_integral]] — the integral over trajectories that defines $Z[J]$
- [[path_integral_theory]] — path-integral formulations of the free energy principle
- [[cumulant_expansion]] — the finite-dimensional counterpart
- [[partition_function]] — the value $Z[0]$ and its role in free energy
- [[field_theory]] — the setting in which functionals of fields are studied
- [[quantum_field_theory]] — where generating functionals are standard

## References
- Zinn-Justin, J. (2002). Quantum Field Theory and Critical Phenomena (4th ed.). Oxford University Press.
- Kardar, M. (2007). Statistical Physics of Fields. Cambridge University Press.
