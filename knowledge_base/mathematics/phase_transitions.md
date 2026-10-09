---
title: Phase Transitions
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - phase_transitions
  - statistical_physics
  - critical_phenomena
  - ising_model
semantic_relations:
  - type: foundation
    links:
      - "[[statistical_physics]]"
      - "[[thermodynamics]]"
  - type: relates
    links:
      - "[[renormalization_group]]"
      - "[[critical_phenomena]]"
      - "[[self_organization]]"
---

# Phase Transitions

## Overview
A phase transition is a qualitative change in the macroscopic state of a many-body system as a control parameter such as temperature or coupling is varied, marked by non-analytic behaviour of thermodynamic quantities. Sharp transitions exist strictly only in the thermodynamic limit of infinitely many components; finite systems show smooth crossovers.

## Definition / Formalism
Transitions are first-order (discontinuity in a first derivative of the free energy, latent heat, coexistence) or continuous (diverging correlation length $\xi\sim|T-T_c|^{-\nu}$, power-law order parameter). Landau theory expands the free energy in an order parameter $m$: $f(m)=a(T-T_c)m^2+bm^4$ and predicts symmetry breaking below $T_c$.

For the mean-field Ising model with coordination number $z$ and coupling $J$, the self-consistency condition in units $k_B=1$ is $m=\tanh(T_c m/T)$ with $T_c=zJ$. Solving it numerically near the critical point gives $m\approx\sqrt{3(1-T/T_c)}$, so the order parameter exponent is $\beta=1/2$ (at $T/T_c=0.999$ iteration gives $m=0.05475$ against $0.05477$ from the formula). Mean-field exponents fail below an upper critical dimension: the exact two-dimensional Ising value is $\beta=1/8$ (Onsager, Yang). The renormalisation group explains universality, in which systems with the same symmetry and dimensionality share exponents regardless of microscopic detail.

## Role in Active Inference
Two connections should be kept apart. The first is formal: the softmax over policies, $P(\pi)=\sigma(-\gamma G(\pi))$, has the form of a Boltzmann distribution with precision $\gamma$ as inverse temperature, but with a finite number of policies it varies smoothly from uniform ($\gamma\to0$) to argmin ($\gamma\to\infty$) and shows no true transition. The second is substantive: the hypothesis that brains operate near a critical point, and that systems with self-organised Markov blankets sit near critical regimes, appears in the free energy literature but is contested and is better treated as an open research claim than a result. Phase-transition language is also used for abrupt behavioural or learning shifts, where the analogy is to bifurcations of the underlying dynamics.

## Related concepts
- [[statistical_physics]] — the framework in which transitions are defined
- [[renormalization_group]] — explains universality and critical exponents
- [[critical_phenomena]] — critical behaviour in connection with the free energy principle
- [[temperature_parameter]] — inverse-temperature role of precision in policy selection
- [[synchronization]] — transition to collective coherence in coupled oscillators
- [[attractor_dynamics]] — attractors created or destroyed across a bifurcation
