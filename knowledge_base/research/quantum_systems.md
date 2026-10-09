---
title: Quantum Systems
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - research
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/research/quantum_cognition]]"
      - "[[knowledge_base/research/quantum_computation]]"
      - "[[knowledge_base/mathematics/markov_blankets]]"
      - "[[knowledge_base/mathematics/non_equilibrium_steady_state]]"
      - "[[knowledge_base/mathematics/statistical_physics]]"
      - "[[knowledge_base/mathematics/information_geometry]]"
---

# Quantum Systems

## Overview

This page is the entry point for the three distinct ways quantum theory touches Active Inference research. They share a vocabulary and are often conflated, so keeping them separate matters. The first is quantum probability as a model of human cognition. The second is quantum computing as a possible substrate or tool. The third is the extension of the free energy principle to physical systems described by quantum mechanics.

## Quantum background

A closed quantum system has a state vector evolving unitarily. Open systems, which interact with an environment, are described by density matrices $\rho$ and evolve under maps such as the Lindblad equation. Interaction with the environment produces decoherence, which suppresses interference between certain states and underlies the apparent classicality of macroscopic objects. The quantum analogue of Shannon entropy is the von Neumann entropy $S(\rho) = -\mathrm{Tr}(\rho \ln \rho)$, and the quantum relative entropy plays the role of the Kullback-Leibler divergence.

## The three threads

**Quantum cognition** applies the Hilbert-space probability calculus to judgement data and does not assert quantum physics in the brain. See [[knowledge_base/research/quantum_cognition]].

**Quantum computation** concerns algorithms and hardware; links to Active Inference are so far speculative. See [[knowledge_base/research/quantum_computation]].

**The free energy principle for quantum systems** is the most formal of the three. The classical principle describes a system that is separated from its environment by a Markov blanket and maintained in a non-equilibrium steady state. Friston's "A free energy principle for a particular physics" (2019) develops that account for classical stochastic dynamics. Fields, Friston, Glazebrook and Levin (2022) argue that the same structure can be stated for any quantum system with a separable interaction between a system and its environment, using the idea of a holographic screen on which the interaction occurs and an associated quantum reference frame. This is a theoretical proposal published in the physics and biophysics literature. It has not been tested experimentally; its status is that of a mathematical framework, not a validated physical law.

Interpretations of quantum mechanics also enter the discussion. QBism treats quantum states as an agent's personal degrees of belief, which resembles Bayesian accounts of the brain, but this is a statement about interpretation and carries no experimental consequences distinguishing it from other interpretations.

## What is not claimed

Active Inference does not require quantum effects in neurons. No evidence currently supports the idea that quantum coherence plays a functional role in neural computation at physiological temperatures, and the standard decoherence arguments are against it. Statements that the free energy principle "is quantum" refer to the formal generality of the 2022 construction.

## In this repository

The classical foundations are in [[knowledge_base/mathematics/markov_blankets]], [[knowledge_base/mathematics/non_equilibrium_steady_state]], [[knowledge_base/mathematics/statistical_physics]] and [[knowledge_base/mathematics/information_geometry]]. A study path is [[active_inference_quantum_cognitive_systems_path]].

## References

- Friston, K. (2019). A free energy principle for a particular physics. arXiv:1906.10184.
- Fields, C., Friston, K., Glazebrook, J. F., & Levin, M. (2022). A free energy principle for generic quantum systems. *Progress in Biophysics and Molecular Biology*, 173, 36-59.

## Related Resources

- [[knowledge_base/research/quantum_cognition]]
- [[knowledge_base/research/quantum_computation]]
