---
title: Saddle-Node Bifurcation
description: The bifurcation in which two equilibria, one stable and one unstable, collide and annihilate as a parameter varies, underlying tipping points, hysteresis and the onset of firing in some neurons.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - bifurcation_theory
  - tipping_points
  - dynamical_systems
semantic_relations:
  - type: relates
    links:
      - "[[bifurcation_theory]]"
      - "[[hysteresis]]"
      - "[[critical_transitions]]"
      - "[[hopf_bifurcation]]"
      - "[[basin_of_attraction]]"
---

# Saddle-Node Bifurcation

## Overview
A saddle-node bifurcation, also called a fold or tangent bifurcation, is a local bifurcation in which two fixed points approach each other, merge at a critical parameter value, and disappear. It is the generic way for equilibria to be created or destroyed in a system with one parameter, and it is the basic mechanism behind abrupt changes in state.

## Normal form
The prototypical equation is
$$\dot x = \mu + x^2.$$
For $\mu<0$ there are two fixed points at $x=\pm\sqrt{-\mu}$. The one at $-\sqrt{-\mu}$ is stable and the one at $+\sqrt{-\mu}$ is unstable. As $\mu\to 0^-$ they approach, at $\mu=0$ they coincide in a single half-stable point, and for $\mu>0$ none exist and every trajectory leaves. At the bifurcation point the eigenvalue of the linearisation passes through zero. Other conditions required for the generic fold are a non-zero quadratic coefficient and a non-zero derivative with respect to the parameter, which together ensure the normal form describes the system. Near the fold, relaxation to the stable state becomes slow, a phenomenon called critical slowing down.

## Consequences
- **Hysteresis**: a system with two folds in sequence has a range of parameter values with two stable states and shows a loop of jumps (see [[hysteresis]]).
- **Tipping points**: when a stable state disappears at a fold, the system moves to a distant state, often irreversibly on the time scale of interest. Rising variance and autocorrelation have been proposed as early-warning signals of such transitions.
- **Neuronal excitability**: in neuron models with a saddle-node on an invariant circle, firing starts at arbitrarily low frequency as the input current increases.

## Relation to Active Inference
Saddle-node structure is a property of dynamical models and not a feature of the free energy principle. It is relevant where Active Inference models describe multistable beliefs, or where the underlying generative process has regimes that appear and disappear, since an agent's inference will show delayed or abrupt changes near a fold. Proposals to use critical slowing down as a signature of failing inference are conjectures to be tested.

## Related concepts
- [[bifurcation_theory]] — the framework that classifies this transition
- [[hysteresis]] — history dependence caused by pairs of folds
- [[critical_transitions]] — abrupt state changes in complex systems
- [[hopf_bifurcation]] — the bifurcation to oscillation
- [[basin_of_attraction]] — the basin that vanishes at the fold
- [[stability_analysis]] — the linear analysis that identifies the fold

## References
- Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos (2nd ed.). Westview Press.
- Izhikevich, E. M. (2007). Dynamical Systems in Neuroscience: The Geometry of Excitability and Bursting. MIT Press.
- Scheffer, M., et al. (2009). Early-warning signals for critical transitions. Nature, 461, 53-59.
