---
title: Hopf Bifurcation
description: The bifurcation in which an equilibrium loses stability as a pair of complex eigenvalues crosses the imaginary axis, creating a limit cycle, with its normal form and uses in oscillation models.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - bifurcation_theory
  - oscillations
  - dynamical_systems
semantic_relations:
  - type: relates
    links:
      - "[[bifurcation_theory]]"
      - "[[limit_cycles]]"
      - "[[saddle_node_bifurcation]]"
      - "[[oscillations]]"
      - "[[stability_analysis]]"
---

# Hopf Bifurcation

## Overview
A Hopf bifurcation is a local bifurcation in which a fixed point of a dynamical system changes stability while a periodic orbit appears. In linear terms, a pair of complex-conjugate eigenvalues of the Jacobian at the fixed point crosses the imaginary axis as a parameter $\mu$ varies, so that at the critical value they are $\pm i\omega$. Before and after the crossing the fixed point is a stable or unstable spiral, and the system oscillates with frequency close to $\omega$ near the transition.

## Normal form
In complex notation the simplest example is
$$\dot z = (\mu + i\omega)\,z - |z|^2 z,\qquad z\in\mathbb{C}.$$
In polar coordinates $z=re^{i\theta}$ this becomes $\dot r=\mu r - r^3$ and $\dot\theta=\omega$. For $\mu<0$ the origin is a stable spiral. For $\mu>0$ it is unstable and there is a stable limit cycle of radius $\sqrt{\mu}$. This is the supercritical case, in which the oscillation grows continuously from zero amplitude. In the subcritical case, represented by $\dot r=\mu r + r^3 - r^5$, an unstable cycle exists before the bifurcation and the system jumps to a large-amplitude oscillation, with hysteresis. Which case occurs depends on the sign of the first Lyapunov coefficient, a nonlinear term. The theorem guaranteeing the bifurcation also requires that the eigenvalues cross with non-zero speed.

## Applications
The bifurcation is a standard route to oscillation in chemical reactions, ecological models, fluid flows and neurons. In neuroscience it describes the onset of repetitive firing in some neuron models and the emergence of rhythms in coupled excitatory and inhibitory populations. It has also been used in models of active amplification in the inner ear, where a system poised near a Hopf point is highly sensitive.

## Relation to Active Inference
The link is through modelling. Neural mass models used in dynamic causal modelling of electrophysiological data are dynamical systems whose parameters can place them near such transitions, so that inferred coupling strengths determine whether the model is damped or oscillates. Active Inference itself makes no specific claim about Hopf bifurcations. It provides a framework for fitting and comparing such models.

## Related concepts
- [[bifurcation_theory]] — the general classification of parameter-driven changes
- [[limit_cycles]] — the periodic orbits created at the bifurcation
- [[saddle_node_bifurcation]] — the other basic codimension-one bifurcation
- [[oscillations]] — neural and physical rhythms
- [[nonlinear_dynamics]] — the field in which bifurcations are analysed
- [[neural_dynamics]] — models of neuronal and population activity

## References
- Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos (2nd ed.). Westview Press.
- Kuznetsov, Y. A. (2004). Elements of Applied Bifurcation Theory (3rd ed.). Springer.
