---
title: Chaos Theory
description: The study of deterministic dynamical systems with sensitive dependence on initial conditions, including Lyapunov exponents and strange attractors, and its limited role in Active Inference models.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - dynamical_systems
  - nonlinear_dynamics
  - attractors
semantic_relations:
  - type: relates
    links:
      - "[[dynamical_systems]]"
      - "[[attractor_dynamics]]"
      - "[[complex_systems]]"
      - "[[phase_transitions]]"
aliases:
  - chaos
---

# Chaos Theory

## Overview
Chaos is aperiodic, bounded behaviour of a deterministic system that depends sensitively on initial conditions. Trajectories starting arbitrarily close separate exponentially, so long-term prediction fails even with exact equations, yet the dynamics are not random: they lie on a structured set, often a fractal strange attractor. Chaos requires nonlinearity and, for continuous flows, at least three dimensions (a consequence of the Poincare-Bendixson theorem for planar flows).

## Quantifying chaos
The largest **Lyapunov exponent** $\lambda$ measures the mean exponential rate of separation, $|\delta(t)|\approx|\delta_0|e^{\lambda t}$. A bounded system with $\lambda>0$ is chaotic. Two reference values, both reproduced numerically:
- The logistic map $x_{n+1}=rx_n(1-x_n)$ at $r=4$ has $\lambda=\ln 2\approx0.693$.
- The Lorenz system $\dot x=\sigma(y-x)$, $\dot y=x(\rho-z)-y$, $\dot z=xy-\beta z$ with $\sigma=10$, $\rho=28$, $\beta=8/3$ has $\lambda\approx0.906$ (per unit time). Its attractor is fractal, with dimension slightly above 2.

Other tools are bifurcation diagrams, which show period-doubling routes (Feigenbaum), Poincare sections, power spectra, and correlation dimension. The prediction horizon scales as $\lambda^{-1}\ln(\Delta/\delta_0)$, where $\Delta$ is the tolerance and $\delta_0$ the initial uncertainty, so improving measurement precision extends forecasts only logarithmically. Noise and chaos are distinguished by determinism tests, which is difficult in short, noisy empirical series.

## Role in Active Inference
The role is real but narrow. Generative models in continuous-time Active Inference include nonlinear flows, and published simulations use chaotic attractors such as the Lorenz system as the hidden dynamics that agents learn to track, for example in models of sequence perception. In such cases inference is possible because the agent models the flow, and sensitivity to initial conditions is reduced by continually correcting with sensory data, which is state estimation from observations with Lyapunov-limited prediction. The free-energy principle in its general form describes systems at non-equilibrium steady state with random fluctuations, so it does not require or predict chaos. Claims that brains operate at the edge of chaos belong to separate empirical literature on criticality, which Active Inference accommodates but does not derive.

## Related concepts
- [[dynamical_systems]] — formal setting for flows and maps
- [[attractor_dynamics]] — fixed points, limit cycles and strange attractors
- [[complex_systems]] — emergent behaviour in nonlinear, many-component systems
- [[phase_transitions]] — qualitative changes in behaviour with parameters
- [[stochastic_differential_equations]] — adding noise to deterministic flows
- [[generalized_coordinates]] — representation used to track nonlinear dynamics in filtering

## References
- Lorenz, E. N. (1963). Deterministic nonperiodic flow. *Journal of the Atmospheric Sciences*, 20, 130-141.
- Strogatz, S. H. (2015). *Nonlinear Dynamics and Chaos*, 2nd ed. Westview Press.
