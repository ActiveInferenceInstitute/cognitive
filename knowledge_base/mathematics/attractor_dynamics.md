---
title: Attractor Dynamics
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - attractors
  - dynamical_systems
  - stability
semantic_relations:
  - type: foundation
    links:
      - "[[dynamical_systems]]"
      - "[[stability_analysis]]"
  - type: relates
    links:
      - "[[non_equilibrium_steady_state]]"
      - "[[self_organization]]"
      - "[[nonlinear_dynamics]]"
aliases:
  - strange_attractors
  - attractors
---

# Attractor Dynamics

## Overview
An attractor of a dynamical system is a closed, invariant set of states toward which nearby trajectories converge. The set of initial conditions that converge to it is its basin of attraction. Attractors can be fixed points, limit cycles, tori (quasi-periodic motion), or strange (chaotic) sets with sensitive dependence on initial conditions.

## Definition / Formalism
For a flow $\dot x = f(x)$, a compact invariant set $\mathcal A$ is an attractor if it has an open neighbourhood $U$ with $\operatorname{dist}(\phi_t(x),\mathcal A)\to 0$ for all $x\in U$, and $\mathcal A$ is minimal with this property. For a fixed point $x^*$, linearisation gives $\dot\delta = J\delta$ with $J=\partial f/\partial x|_{x^*}$; if every eigenvalue of $J$ has negative real part, $x^*$ is asymptotically stable. Chaotic attractors have a positive largest Lyapunov exponent.

A gradient system $\dot x = -\nabla V(x)$ has only fixed-point attractors, located at local minima of $V$, because along trajectories $\dot V = -\lVert\nabla V\rVert^2\le 0$. Example: $V=x^4/4-x^2/2$ has stable minima at $x=\pm1$ (derivative of the vector field $-3x^2+1$ equals $-2$ there) and an unstable point at $x=0$ (equal to $+1$). Hopfield networks use the same idea: stored patterns are minima of an energy, so memory retrieval is convergence to an attractor.

## Role in Active Inference
Two uses are well grounded. First, belief updating as gradient flow $\dot\mu=-\partial_\mu F$ makes free energy a Lyapunov function, so beliefs settle at local minima of $F$; this is the sense in which perception is a fixed-point attractor of the recognition dynamics. Second, the free energy principle assumes the agent's states have a non-equilibrium steady-state density with low entropy, which is a statistical statement that the system occupies a characteristic, attractor-like set of states. The step from this to specific neural attractor models (for example working memory as attractor networks) is a modelling hypothesis, not a consequence of the principle. Continuous attractor and multistable models are used in the literature on neural dynamics, but the repo does not implement them as part of its core agents.

## Related concepts
- [[dynamical_systems]] — the setting in which attractors are defined
- [[stability_analysis]] — Lyapunov and linearisation tools for classifying attractors
- [[non_equilibrium_steady_state]] — the statistical analogue of an attractor under noise
- [[neural_dynamics]] — attractor models of neural activity
- [[self_organization]] — how attractors arise from local interactions
- [[phase_transitions]] — attractors appearing or vanishing as parameters change
- [[neural_network_theory]] — Hopfield networks as attractor memories
