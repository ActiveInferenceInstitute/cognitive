---
title: Timescale Separation
description: The mathematical and conceptual use of fast and slow variables evolving at very different rates, including slow manifolds, adiabatic elimination and the slaving principle, and its role in hierarchical Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - slow_fast_systems
  - hierarchy
  - dynamical_systems
semantic_relations:
  - type: relates
    links:
      - "[[dynamical_systems]]"
      - "[[synergetics]]"
      - "[[order_parameters]]"
      - "[[hierarchical_models]]"
      - "[[temporal_models]]"
---

# Timescale Separation

## Overview
Timescale separation refers to systems in which some variables change much faster than others. This is common in nature: chemical reactions equilibrate faster than gene expression changes, neural activity fluctuates faster than synaptic strengths, and ecological populations change faster than the evolution of their traits. When the separation is large, the system can be analysed in pieces, treating the slow variables as constants for the fast dynamics and the fast variables as already settled for the slow dynamics.

## Slow-fast systems
A standard form is
$$\dot{x}=f(x,y),\qquad \varepsilon\,\dot{y}=g(x,y),$$
with slow variable $x$, fast variable $y$ and small $\varepsilon>0$. For $\varepsilon\to0$ the fast variable relaxes onto the set $g(x,y)=0$, called the slow or critical manifold, provided it is stable, and the slow dynamics then evolve on that manifold. Tikhonov's theorem gives conditions under which this reduction is valid, and geometric singular perturbation theory due to Fenichel extends it. Where the manifold loses stability, the fast variable jumps, producing relaxation oscillations and abrupt transitions (see [[perturbation_theory]]).

## Adiabatic elimination and slaving
In physics the same reduction is called adiabatic elimination. Haken's synergetics calls it the slaving principle: near an instability the few slow modes act as order parameters, and the many fast, stable modes are enslaved, determined by the slow ones. This is used to explain how high-dimensional systems collapse onto low-dimensional collective descriptions (see [[synergetics]] and [[order_parameters]]).

## Relation to Active Inference
The link is strong and used in practice. Hierarchical generative models assign higher levels to slower dynamics, so that fast sensory fluctuations are explained by lower levels evolving quickly under slowly changing contexts above them (see [[hierarchical_models]] and [[temporal_models]]). Learning of parameters is treated as inference at a slower timescale than inference of states. Some derivations in the free energy literature also assume that certain variables evolve much faster than others. How far real brains follow a clean hierarchy of timescales is an empirical matter.

## Related concepts
- [[synergetics]] — the slaving principle and order parameters
- [[order_parameters]] — slow variables that summarise collective behaviour
- [[dynamical_systems]] — the mathematical setting for slow-fast analysis
- [[hierarchical_models]] — layered models with slower deeper levels
- [[temporal_models]] — models of sequences and dynamics at different depths
- [[perturbation_theory]] — the general expansion technique behind the reduction

## References
- Haken, H. (1983). Synergetics: An Introduction (3rd ed.). Springer.
- Kuehn, C. (2015). Multiple Time Scale Dynamics. Springer.
- Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos (2nd ed.). Westview Press.
