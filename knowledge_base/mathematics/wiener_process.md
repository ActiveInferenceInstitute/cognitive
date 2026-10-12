---
title: Wiener Process
description: The continuous-time stochastic process underlying Brownian motion, defined by independent Gaussian increments and continuous paths, and the noise term dW in stochastic differential equations and Langevin models.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - stochastic_processes
  - brownian_motion
  - noise
semantic_relations:
  - type: relates
    links:
      - "[[stochastic_differential_equations]]"
      - "[[ito_calculus]]"
      - "[[stochastic_processes]]"
      - "[[langevin_dynamics]]"
      - "[[gaussian_distributions]]"
wikidata: Q1056809
---

# Wiener Process

## Overview
The Wiener process $W_t$, also called standard Brownian motion, is the basic continuous-time model of random fluctuation. It is the scaling limit of a simple random walk and is the source of the noise term $dW$ in stochastic differential equations of the form $dx = f(x)\,dt + \sigma\,dW$.

## Definition
A real-valued process $(W_t)_{t\ge 0}$ is a Wiener process if:
1. $W_0 = 0$.
2. Increments over disjoint intervals are independent.
3. For $s<t$, $W_t - W_s \sim \mathcal{N}(0,\,t-s)$.
4. Sample paths are continuous with probability one.

The existence of such a process on path space is established by constructions such as the Kolmogorov extension combined with a continuity argument, and the law of $W$ on continuous paths is the Wiener measure (see [[kolmogorov_extension]]).

## Key properties
- **Roughness**: paths are continuous but almost surely nowhere differentiable, so $dW/dt$ exists only as a generalised function (white noise).
- **Scaling**: $dW$ has size of order $\sqrt{dt}$, and heuristically $dW^2 = dt$. The quadratic variation of $W$ on $[0,t]$ equals $t$, which is why ordinary calculus fails and Ito or Stratonovich calculus is needed (see [[ito_calculus]]).
- **Martingale**: $W_t$ is a martingale, and $W_t^2 - t$ is also a martingale.
- **Self-similarity**: $c^{-1/2}W_{ct}$ has the same law as $W_t$.
- **Link to diffusion**: the density of $W_t$ solves the heat equation, a special case of the Fokker-Planck equation (see [[fokker_planck]]).

## Relation to Active Inference
Generative models in continuous-time Active Inference and the free energy principle describe states as stochastic flows $\dot x = f(x) + \omega$, where $\omega$ is a random fluctuation. Written in Ito form this is driven by a Wiener process, and the Langevin and Fokker-Planck treatments that give a steady-state density over states rely on this noise model. The derivation is standard stochastic mechanics. Some Active Inference schemes instead assume smooth, analytic fluctuations in generalised coordinates, which is a different noise model from a Wiener process.

## Related concepts
- [[stochastic_differential_equations]] — equations driven by $dW$
- [[ito_calculus]] — the calculus that handles non-differentiable paths
- [[random_walk]] — the discrete process whose limit is $W_t$
- [[martingale_theory]] — $W_t$ is a canonical martingale
- [[langevin_dynamics]] — physical dynamics with Wiener noise
- [[fokker_planck]] — evolution equation for the density of the process

## References
- Karatzas, I., & Shreve, S. E. (1991). Brownian Motion and Stochastic Calculus (2nd ed.). Springer.
- Oksendal, B. (2003). Stochastic Differential Equations: An Introduction with Applications (6th ed.). Springer.
