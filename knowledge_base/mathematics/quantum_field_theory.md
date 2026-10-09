---
title: Quantum Field Theory
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - quantum_field_theory
  - path_integral
  - renormalization
semantic_relations:
  - type: foundation
    links:
      - "[[quantum_mechanics]]"
      - "[[path_integral]]"
  - type: relates
    links:
      - "[[renormalization_group]]"
      - "[[statistical_physics]]"
      - "[[bayesian_renormalization]]"
---

# Quantum Field Theory

## Overview
Quantum field theory (QFT) combines quantum mechanics with special relativity by taking the fundamental objects to be fields: operator-valued distributions defined at every point of spacetime, whose excitations are particles. It is the framework of the Standard Model of particle physics and, in its Euclidean form, of statistical field theory.

## Definition / Formalism
A field theory is specified by an action $S[\phi]=\int d^dx\,\mathcal L(\phi,\partial\phi)$. For a free real scalar field $\mathcal L=\tfrac12(\partial\phi)^2-\tfrac12m^2\phi^2$. In the path-integral formulation the generating functional is

$$Z[J]=\int\mathcal D\phi\;\exp\!\Big(\tfrac{i}{\hbar}\big(S[\phi]+\!\int J\phi\big)\Big),$$

and correlation functions are functional derivatives of $\ln Z$ with respect to the source $J$. Rotating to imaginary time (Wick rotation) gives the Euclidean form $Z=\int\mathcal D\phi\,e^{-S_E[\phi]}$, which is formally a statistical-mechanical partition function with $S_E$ playing the role of energy. For the free scalar field the Euclidean propagator is $(-\partial^2+m^2)^{-1}$. Interactions are handled perturbatively by Feynman diagrams, and the divergences that appear are absorbed by renormalisation; the renormalisation group describes how effective couplings change with scale, with fixed points governing critical behaviour.

## Role in Active Inference
The link is through shared mathematics, not through a claim that cognition is quantum-field-theoretic. Euclidean path integrals have the same structure as Bayesian posteriors $p\propto e^{-S}$, with the action as a negative log-density; perturbation expansions in Feynman diagrams are close in spirit to message-passing and Bethe-type approximations on factor graphs. Renormalisation-group ideas have been used to discuss coarse-graining of generative models and the emergence of hierarchical structure, and the free energy principle itself has been connected to a path-integral formulation of dynamics. These are analogies and research proposals of varying maturity; no agent in this repository uses QFT, and the empirical claims of active inference do not depend on it.

## Related concepts
- [[quantum_mechanics]] — the non-relativistic base on which QFT is built
- [[path_integral]] — the functional-integral formulation of dynamics
- [[renormalization_group]] — scale-dependence of effective theories
- [[bayesian_renormalization]] — coarse-graining of Bayesian models
- [[statistical_physics]] — the Euclidean counterpart in statistical mechanics
- [[phase_transitions]] — critical behaviour described by fixed points of the RG flow
