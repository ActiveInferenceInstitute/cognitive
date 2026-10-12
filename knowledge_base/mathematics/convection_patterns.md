---
title: Convection Patterns
description: Convection patterns are ordered flow cells such as rolls and hexagons that form in fluid layers heated from below once the temperature gradient exceeds a threshold, a standard example of self-organisation.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - pattern_formation
  - self_organization
  - fluid_dynamics
semantic_relations:
  - type: relates
    links:
      - "[[pattern_formation]]"
      - "[[dissipative_structures]]"
      - "[[bifurcation_theory]]"
      - "[[heat_transfer]]"
      - "[[emergence_self_organization]]"
---

# Convection Patterns

## Overview
When a horizontal layer of fluid is heated from below and cooled from above, heat first moves by conduction. Beyond a critical temperature difference the uniform state becomes unstable and the fluid organises into regular convection cells, with warm fluid rising in some regions and cool fluid sinking in others. This is Rayleigh-Bénard convection, and it is a standard laboratory example of a pattern arising spontaneously in a driven, dissipative system.

## Onset and control parameter
The relevant dimensionless quantity is the Rayleigh number
$$\mathrm{Ra}=\frac{g\,\alpha\,\Delta T\,d^{3}}{\nu\,\kappa},$$
where $g$ is gravitational acceleration, $\alpha$ the thermal expansion coefficient, $\Delta T$ the temperature difference across a layer of depth $d$, $\nu$ the kinematic viscosity and $\kappa$ the thermal diffusivity. Linear stability analysis shows that conduction is stable for $\mathrm{Ra}$ below a critical value, about 1708 for a layer between two rigid plates, and unstable above it. At onset the preferred horizontal wavelength is of the order of twice the layer depth. The transition is a pattern-forming bifurcation: the translation symmetry of the layer is broken by a periodic structure.

## Patterns
Just above onset the usual pattern is a set of parallel rolls. Other planforms occur under different conditions, for example hexagonal cells when the fluid properties vary with temperature or when surface tension drives the flow in a free-surface layer, the case in Bénard's original experiments. At higher $\mathrm{Ra}$ the rolls become unsteady, then time-dependent and eventually turbulent. Lorenz's three-variable system, a drastic truncation of the convection equations, became a founding example of deterministic chaos.

Weakly nonlinear theory describes the slow modulation of the roll amplitude with amplitude equations, which are shared across many pattern-forming systems and explain why the same types of pattern appear in otherwise unrelated settings.

## Relation to Active Inference
Convection cells are often cited as a physical example of a dissipative structure that maintains order by exporting entropy, which is the setting in which the free energy principle is motivated. The connection is an analogy: the principle is formulated in terms of random dynamical systems with Markov blankets and does not derive convection patterns, and convection cells are not usually treated as having a generative model.

## Related concepts
- [[pattern_formation]] — the general theory of spontaneous spatial order
- [[dissipative_structures]] — Prigogine's term for order sustained by flux
- [[bifurcation_theory]] — the mathematics of the transition at critical Ra
- [[heat_transfer]] — conduction and convection as transport modes
- [[chaos_theory]] — the route from cells to unsteady flow
- [[emergence_self_organization]] — the broader theme of order from local interactions

## References
- Chandrasekhar, S. (1961). Hydrodynamic and Hydromagnetic Stability. Clarendon Press.
- Lorenz, E. N. (1963). Deterministic nonperiodic flow. Journal of the Atmospheric Sciences, 20, 130-141.
- Cross, M. C., & Hohenberg, P. C. (1993). Pattern formation outside of equilibrium. Reviews of Modern Physics, 65, 851-1112.
