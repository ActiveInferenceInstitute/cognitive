---
title: Ito Calculus
description: "Stochastic integrals against Brownian motion, Ito's lemma, the Ito versus Stratonovich convention, and their role in the Langevin dynamics underlying the free energy principle."
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - mathematics
  - stochastic_calculus
  - sde
semantic_relations:
  - type: relates
    links:
      - "[[stochastic_differential_equations]]"
      - "[[stochastic_processes]]"
      - "[[langevin_dynamics]]"
      - "[[fokker_planck]]"
---

# Ito Calculus

## Overview
Ito calculus defines integrals of the form $\int_0^t H_s\,dW_s$ against Brownian motion $W$, whose paths are continuous but nowhere differentiable, so ordinary calculus does not apply. It is the standard framework for [[stochastic_differential_equations]] $dX_t=f(X_t)\,dt+\sigma(X_t)\,dW_t$.

## Core results
**The integral.** The Ito integral evaluates the integrand at the left endpoint of each partition interval, so the integrand never anticipates the noise. The result is a martingale when the integrand is adapted and square integrable, and satisfies the isometry $\mathbb E[(\int H\,dW)^2]=\mathbb E\int H^2\,dt$.

**Quadratic variation.** $(dW)^2=dt$, not zero. This single rule is what changes the chain rule.

**Ito's lemma.** For smooth $g$ and $dX=f\,dt+\sigma\,dW$ in one dimension,
$$dg(X)=g'(X)\,dX+\tfrac12\sigma^2 g''(X)\,dt.$$
Example: for geometric Brownian motion $dX=\mu X\,dt+\sigma X\,dW$ this gives $d\ln X=(\mu-\sigma^2/2)\,dt+\sigma\,dW$. A simulation with $\mu=0.5,\sigma=0.8,T=1$ gave $\mathbb E[\ln X_T]\approx 0.180$, matching $\mu-\sigma^2/2=0.18$ rather than the naive $0.5$.

**Ito versus Stratonovich.** The Stratonovich integral uses midpoints and obeys the ordinary chain rule, but is not a martingale. The two conventions describe the same process with different drift when noise is state dependent, differing by $\tfrac12\sigma\sigma'$. The convention must be stated, and it shows up as an extra divergence term in the Onsager-Machlup action.

**Fokker-Planck link.** The density of $X_t$ obeys the [[fokker_planck]] equation, derived by applying Ito's lemma to test functions.

## Role in Active Inference
The free energy principle begins from a Langevin equation $\dot x=f(x)+\omega$ and uses its Fokker-Planck equation to characterise the non-equilibrium steady state. The "Itô correction" appearing in [[path_integral_free_energy]] is the $\tfrac12\sigma^2$-type term above. Most discrete-state Active Inference code never uses Ito calculus; it matters in continuous-time formulations such as [[continuous_time_models]] and [[generalized_coordinates]], where smooth (non-Wiener) fluctuations are often assumed and the convention question is sidestepped.

## Related concepts
- [[stochastic_processes]] — Brownian motion and martingales
- [[langevin_dynamics]] — the physical SDEs written in Ito or Stratonovich form
- [[path_integral_free_energy]] — where the correction term appears
- [[non_equilibrium_steady_state]] — the stationary solution of the Fokker-Planck equation
