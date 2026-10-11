---
title: Bifurcation Theory
description: The study of qualitative changes in the long-term behaviour of parameterised dynamical systems, including saddle-node, pitchfork and Hopf bifurcations, and their use in models of neural and self-organising systems.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - dynamical_systems
  - stability
  - nonlinear_dynamics
semantic_relations:
  - type: relates
    links:
      - "[[dynamical_systems]]"
      - "[[stability_analysis]]"
      - "[[attractor_dynamics]]"
      - "[[phase_transitions]]"
      - "[[symmetry_breaking]]"
      - "[[free_energy_principle]]"
---

# Bifurcation Theory

## Overview
Bifurcation theory studies how the qualitative structure of solutions to a parameterised system $\dot{x}=f(x,\mu)$ changes as a parameter $\mu$ is varied. A bifurcation occurs at a parameter value where the number, location or stability of equilibria, cycles or other invariant sets changes. The theory explains why small, smooth changes in a parameter can produce abrupt changes in behaviour.

## Local bifurcations
Local bifurcations are detected from the linearisation at an equilibrium. A bifurcation requires a non-hyperbolic equilibrium, that is, a Jacobian with an eigenvalue whose real part is zero. Near such a point, centre-manifold reduction and normal forms reduce the dynamics to a low-dimensional canonical equation.

- **Saddle-node (fold).** $\dot{x}=\mu-x^2$: two equilibria collide and annihilate as $\mu$ crosses zero.
- **Transcritical.** $\dot{x}=\mu x-x^2$: two equilibria exchange stability.
- **Pitchfork.** $\dot{x}=\mu x-x^3$ (supercritical): a stable symmetric state loses stability and two symmetric-broken states appear, a simple model of [[symmetry_breaking]].
- **Hopf.** A pair of complex-conjugate eigenvalues crosses the imaginary axis transversally, and a limit cycle is born or destroyed. This is the generic route from rest to oscillation.

Near a bifurcation the system recovers slowly from perturbations, known as critical slowing down. Systems with fold bifurcations can also show hysteresis, so the path along which the parameter changes affects the outcome.

## Relation to Active Inference
The link is an active modelling practice more than a derivation. Neural mass and neural field models, used in dynamic causal modelling, have parameters such as synaptic gains and time constants whose changes can move the system across stability boundaries, producing the onset of oscillations or seizure-like activity. The [[free_energy_principle]] characterises systems through a (non-equilibrium) steady-state density, and bifurcations describe how that density's attractor structure could reorganise as parameters change. The principle does not itself predict which bifurcations occur in a given system.

## Related concepts
- [[dynamical_systems]] — the general framework in which bifurcations are defined
- [[stability_analysis]] — eigenvalue and Lyapunov tools used to locate bifurcations
- [[attractor_dynamics]] — the invariant sets whose structure changes
- [[phase_transitions]] — bifurcation-like transitions in many-body systems
- [[symmetry_breaking]] — the pitchfork as the prototype

## References
- Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos (2nd ed.). Westview Press.
- Kuznetsov, Y. A. (2004). Elements of Applied Bifurcation Theory (3rd ed.). Springer.
- Guckenheimer, J., & Holmes, P. (1983). Nonlinear Oscillations, Dynamical Systems, and Bifurcations of Vector Fields. Springer.
