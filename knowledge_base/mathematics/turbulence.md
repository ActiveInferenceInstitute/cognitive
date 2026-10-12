---
title: Turbulence
description: Irregular, multi-scale fluid motion at high Reynolds number, its energy cascade and Kolmogorov scaling, the open mathematical problems, and its use as a standard example of complex dynamics.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - fluid_dynamics
  - multiscale_systems
  - chaos
semantic_relations:
  - type: relates
    links:
      - "[[chaos_theory]]"
      - "[[nonlinearity]]"
      - "[[scale_invariance]]"
      - "[[power_laws]]"
      - "[[emergence_self_organization]]"
---

# Turbulence

## Overview
Turbulence is the irregular, seemingly random flow that fluids exhibit when inertial forces dominate viscous ones. The ratio is measured by the Reynolds number $\mathrm{Re}=UL/\nu$, with characteristic speed $U$, length $L$ and kinematic viscosity $\nu$. At low $\mathrm{Re}$ flow is smooth (laminar), while at high $\mathrm{Re}$ it develops fluctuations over a wide range of spatial and temporal scales. The governing equations, the incompressible Navier-Stokes equations, are deterministic, so turbulence is a form of deterministic but unpredictable behaviour in a strongly nonlinear system ([[nonlinearity]]).

## Energy cascade and Kolmogorov scaling
In the picture associated with Richardson and Kolmogorov, energy is injected at large scales, transferred by nonlinear interactions to successively smaller eddies, and dissipated by viscosity at the smallest scales. Kolmogorov's 1941 theory assumes that in the intermediate inertial range the statistics are universal and depend only on the mean energy dissipation rate $\varepsilon$. Dimensional analysis then gives the energy spectrum
$$E(k)\propto\varepsilon^{2/3}k^{-5/3},$$
a power law in the wavenumber $k$ ([[power_laws]]). The smallest scale, the Kolmogorov length, is $\eta=(\nu^3/\varepsilon)^{1/4}$. The $-5/3$ law agrees well with many measurements, but higher-order statistics show intermittency, deviations from the 1941 scaling that are attributed to the spatial concentration of dissipation, and a complete theory remains unavailable (Frisch, 1995; Pope, 2000).

## Mathematical status
Whether solutions of the three-dimensional Navier-Stokes equations with smooth initial data remain smooth for all time is an open problem, listed among the Clay Mathematics Institute Millennium Problems. The transition to turbulence has also been studied through dynamical systems. Ruelle and Takens (1971) proposed that turbulence could arise after a small number of bifurcations leading to a strange attractor, in contrast to the earlier view of an infinite sequence of independent oscillations ([[chaos_theory]], [[bifurcation_theory]]). Subsequent experiments support this for some flows, while fully developed turbulence involves many degrees of freedom.

## Relation to Active Inference
There is no direct derivation connecting turbulence to Active Inference. Turbulence serves as a reference example for multi-scale organisation with scale-free statistics, and the renormalisation group has been applied to it, ideas that also appear in discussions of scale-free brain dynamics and hierarchical generative models ([[renormalization_group]], [[scale_invariance]]). Such parallels are analogies of mathematical form, not evidence of shared mechanism. The vault uses turbulence mainly as an illustration in discussions of emergence and self-organisation ([[emergence_self_organization]]).

## Related concepts
- [[chaos_theory]] — deterministic unpredictability
- [[nonlinearity]] — the source of the cascade
- [[scale_invariance]] — statistics without a characteristic scale
- [[power_laws]] — the form of the inertial-range spectrum
- [[renormalization_group]] — a method applied to multi-scale problems
- [[emergence_self_organization]] — collective structure from interacting parts

## References
- Frisch, U. (1995). Turbulence. Cambridge University Press.
- Pope, S. B. (2000). Turbulent Flows. Cambridge University Press.
- Davidson, P. A. (2015). Turbulence: An Introduction for Scientists and Engineers (2nd ed.). Oxford University Press.
- Ruelle, D., & Takens, F. (1971). On the nature of turbulence. Communications in Mathematical Physics, 20, 167-192.
- Колмогоров, А. Н. (1941). Локальная структура турбулентности в несжимаемой вязкой жидкости при очень больших числах Рейнольдса. Доклады Академии наук СССР, 30, 299-303. English translation: Kolmogorov, A. N. (1991). The local structure of turbulence in incompressible viscous fluid for very large Reynolds numbers. Proceedings of the Royal Society A, 434, 9-13.
