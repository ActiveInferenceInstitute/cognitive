---
title: Bifurcation Diagrams
description: Plots of equilibria and periodic orbits against a control parameter, the standard saddle-node, transcritical, pitchfork and Hopf examples, the period-doubling route to chaos, and uses in neural dynamics.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - dynamical_systems
  - bifurcation
  - nonlinear_dynamics
semantic_relations:
  - type: relates
    links:
      - "[[bifurcation_theory]]"
      - "[[saddle_node_bifurcation]]"
      - "[[hopf_bifurcation]]"
      - "[[period_doubling]]"
      - "[[dynamical_systems]]"
      - "[[critical_transitions]]"
---

# Bifurcation Diagrams

## Overview
A bifurcation diagram is a graph that shows how the long-term behaviour of a dynamical system changes as a control parameter is varied. The horizontal axis is the parameter $r$, and the vertical axis shows a state variable at its equilibria or, for periodic and chaotic regimes, at its extrema. By convention stable branches are drawn as solid curves and unstable ones as dashed curves. The qualitative changes visible in the diagram, where branches appear, vanish, meet or change stability, are bifurcations (see [[bifurcation_theory]]).

## Standard one-dimensional examples
For $\dot{x} = f(x, r)$, the equilibria satisfy $f(x, r) = 0$ and are stable where $\partial f / \partial x < 0$.
- **Saddle-node:** $\dot{x} = r + x^2$. For $r < 0$ there are two equilibria $x = \pm\sqrt{-r}$, the one at $-\sqrt{-r}$ stable and the other unstable. They collide at $r = 0$ and disappear for $r > 0$ (see [[saddle_node_bifurcation]]).
- **Transcritical:** $\dot{x} = rx - x^2$. Equilibria at $x = 0$ and $x = r$ cross at $r = 0$ and exchange stability.
- **Supercritical pitchfork:** $\dot{x} = rx - x^3$. The origin is stable for $r < 0$ and loses stability at $r = 0$, while a symmetric pair $x = \pm\sqrt{r}$ appears for $r > 0$. In the subcritical case $\dot{x} = rx + x^3$ the branches are unstable and the transition can be abrupt.
- **Hopf:** in the plane, a fixed point loses stability as a pair of complex eigenvalues crosses the imaginary axis, and a limit cycle is born. In the supercritical case its amplitude grows from zero as the parameter passes the critical value (see [[hopf_bifurcation]]).

## Period doubling
For the logistic map $x_{n+1} = r x_n (1 - x_n)$, the fixed point loses stability at $r = 3$ and a period-2 cycle appears. A cascade of period doublings follows, accumulating near $r \approx 3.57$, beyond which chaotic dynamics occur. Successive bifurcation values converge at a rate given by the Feigenbaum constant $\delta \approx 4.669$, which is the same for a wide class of unimodal maps (see [[period_doubling]]).

## Relation to Active Inference
The stability of attracting sets can change when parameters such as precision or coupling vary. In neural mass and dynamic causal models, bifurcation analysis shows when oscillations appear and how activity changes near criticality, where recovery from perturbation slows (critical slowing down). The broader claim that brains operate near bifurcations is a hypothesis with mixed support.

## Related concepts
- [[bifurcation_theory]] — the theory behind the diagrams
- [[saddle_node_bifurcation]] — the creation and annihilation of equilibria
- [[hopf_bifurcation]] — the birth of oscillations
- [[period_doubling]] — the cascade leading to chaos
- [[critical_transitions]] — abrupt shifts in complex systems
- [[hysteresis]] — path-dependent switching where stable branches overlap

## References
- Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering (2nd ed.). Westview Press.
- Kuznetsov, Y. A. (2004). Elements of Applied Bifurcation Theory (3rd ed.). Springer.
- Guckenheimer, J., & Holmes, P. (1983). Nonlinear Oscillations, Dynamical Systems, and Bifurcations of Vector Fields. Springer.
