---
title: Complexity Theory
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - complexity
  - systems_theory
  - information_theory
semantic_relations:
  - type: relates
    links:
      - "[[complex_systems]]"
      - "[[active_inference]]"
      - "[[emergence]]"
---

# Complexity Theory

## Overview
"Complexity" names several distinct research traditions, and a precise use needs to say which. Computational complexity classifies problems by resource cost, algorithmic information theory measures the description length of objects, and the science of complex systems studies collectives whose interacting parts produce organized behaviour. This page separates them; the dynamics of complex systems themselves are treated in [[complex_systems]].

## Core claims
**Computational complexity.** Classes such as P and NP are defined by how time or memory grows with input size. Exact Bayesian inference over a general discrete graphical model is intractable, which is the formal reason for approximations like variational inference.

**Algorithmic and statistical complexity.** Kolmogorov complexity is the length of the shortest program producing an object, and is uncomputable in general. Crutchfield's statistical complexity is the entropy of a process's minimal predictive states, so unlike Kolmogorov complexity it is low for both perfect order and pure randomness, peaking in between.

**Complex adaptive systems.** The Santa Fe tradition (Holland, Kauffman, Gell-Mann) studies many interacting agents whose adaptation yields emergence, nonlinearity, feedback and path dependence. Claims such as "life sits at the edge of chaos" and the ubiquity of power laws are contested and should not be treated as established laws.

**Measures are not interchangeable.** Entropy, mutual information, effective complexity and graph-theoretic measures answer different questions, and no single scalar captures "how complex" a system is.

## Role in Active Inference
Two links are solid and one is aspirational. First, the tractability link: variational free energy is a bound chosen because exact posteriors are computationally hard, and enumerating policies costs `num_actions ** horizon`, which is why `enumerate_policies` takes a `limit`. Second, the model-complexity link: the complexity term in free energy is a KL divergence from posterior to prior, a measure of how far the data moved the beliefs; this shares a name with, but is not the same quantity as, the notions above (see [[model_complexity]]). The aspirational link is the claim that the free-energy principle unifies complex-systems phenomena such as self-organization through nonequilibrium steady states and Markov blankets. That is a research programme with formal results under stated assumptions, and critics question how widely the assumptions hold.

## Related concepts
- [[complex_systems]] — dynamics and structure of complex systems
- [[complexity_and_philosophy]] — reductionism, emergence and explanation
- [[emergence]] — higher-level organization from interaction
- [[self_organization]] — order arising without central control
- [[model_complexity]] — the KL complexity term in free energy
- [[markov_blanket]] — the statistical boundary used to define systems

## References
- Kolmogorov, A. N. (1965). Three approaches to the quantitative definition of information. *Problems of Information Transmission*.
- Crutchfield, J. P. & Young, K. (1989). Inferring statistical complexity. *Physical Review Letters*.
