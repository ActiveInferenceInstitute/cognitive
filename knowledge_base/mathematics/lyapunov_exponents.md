---
title: Lyapunov Exponents
description: Rates at which nearby trajectories of a dynamical system diverge or converge, their definition and spectrum, use as a signature of chaos, and relevance to stability arguments in free energy dynamics.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - dynamical_systems
  - chaos
  - stability
semantic_relations:
  - type: relates
    links:
      - "[[chaos_theory]]"
      - "[[lyapunov_stability]]"
      - "[[dynamical_systems]]"
      - "[[attractor_dynamics]]"
      - "[[free_energy_principle]]"
---

# Lyapunov Exponents

## Overview
Lyapunov exponents quantify the average exponential rate at which infinitesimally close trajectories separate or approach one another. If an initial perturbation $\delta(0)$ evolves to $\delta(t)$, the largest exponent is
$$\lambda_1=\lim_{t\to\infty}\frac{1}{t}\ln\frac{\lVert\delta(t)\rVert}{\lVert\delta(0)\rVert}.$$
A positive value means that small differences grow exponentially, so long-term prediction is limited by sensitive dependence on initial conditions. A negative value means perturbations decay.

## Spectrum and interpretation
An $n$-dimensional flow has $n$ exponents $\lambda_1\ge\dots\ge\lambda_n$, which describe growth or contraction along different directions. Their sum equals the long-run average of the divergence of the vector field, which is the rate at which phase-space volume expands or contracts. In a dissipative system the sum is negative.

- **Fixed point**: the exponents equal the real parts of the Jacobian eigenvalues, and all are negative for a stable fixed point.
- **Limit cycle**: one exponent is zero, corresponding to the direction along the orbit, and the others are negative.
- **Chaotic attractor**: a bounded trajectory with at least one positive exponent. The Lorenz system with its standard parameters has a largest exponent of about 0.9 per unit time.

The existence and well-definedness of the exponents for almost every initial condition follow from Oseledets' multiplicative ergodic theorem.

## Estimation
Exponents are computed numerically by integrating the system together with its linearisation and periodically re-orthonormalising the tangent vectors. From experimental time series, they are estimated from embedded reconstructions, although results are sensitive to noise, finite data and the choice of embedding parameters.

## Relation to Active Inference
The free energy principle describes systems that remain near a set of characteristic states, so stability measures are natural tools in its dynamics. Negative exponents for the directions pulling a system back toward its attracting set express the resilience of that set, and a near-zero exponent indicates critical slowing, in which recovery from perturbation becomes slow. Some presentations of the principle relate the exponents of the flow to the statistics of a system's blanket and internal states. This connection is a conceptual use of the tool and not a theorem that applies to every system.

## Related concepts
- [[chaos_theory]] — positive exponents are the standard indicator of chaos
- [[lyapunov_stability]] — qualitative stability of equilibria and sets
- [[limit_cycles]] — periodic orbits, which have a zero exponent
- [[attractor_dynamics]] — attractors classified by exponent signs
- [[free_energy_principle]] — stability of characteristic states

## References
- Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos (2nd ed.). Westview Press.
- Ott, E. (2002). Chaos in Dynamical Systems (2nd ed.). Cambridge University Press.
- Wolf, A., Swift, J. B., Swinney, H. L., & Vastano, J. A. (1985). Determining Lyapunov exponents from a time series. Physica D, 16, 285-317.
