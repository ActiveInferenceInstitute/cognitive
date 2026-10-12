---
title: Determinism
description: The thesis that the world's state at one time plus the laws of nature fixes all other states, its status in physics, its bearing on free will, and stochastic dynamics in Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - metaphysics
  - causation
  - free_will
semantic_relations:
  - type: relates
    links:
      - "[[free_will]]"
      - "[[causality]]"
      - "[[chaos_theory]]"
      - "[[stochastic_processes]]"
      - "[[langevin_dynamics]]"
wikidata: Q131133
---

# Determinism

## Overview
Causal determinism is the thesis that the complete state of the world at one time, together with the laws of nature, entails the complete state at every other time. Laplace (1814) gave the classic image of an intellect that knew all forces and positions at one moment and could compute the whole future and past from them. Determinism concerns what is the case in the world, not what can be predicted by any observer.

## Determinism and physics
- **Classical mechanics.** Its equations of motion are deterministic for well-behaved systems, although there are known exotic cases where solutions are not unique.
- **Chaos.** Deterministic systems can be sensitive to initial conditions, so that long-term prediction fails in practice without any indeterminism in the laws (see [[chaos_theory]]).
- **Quantum mechanics.** The Schrödinger equation evolves the wavefunction deterministically, but the status of measurement outcomes depends on the interpretation. Bohmian mechanics and many-worlds interpretations are deterministic in a suitable sense, while collapse interpretations are not.
- **Statistical descriptions.** Stochastic models in thermodynamics and biology may reflect ignorance of microscopic detail, not fundamental randomness.

Hoefer's survey emphasises that whether the actual world is deterministic is an open empirical and interpretive matter, not something that logic or classical physics settles.

## Free will
Three positions are standard. Hard determinists hold that determinism is true and rules out free will. Libertarians hold that free will requires indeterminism and that we have it. Compatibilists hold that free will is compatible with determinism, since it concerns the way actions are produced, not the absence of antecedent causes. Van Inwagen (1983) gave an influential argument that determinism is incompatible with free will (see [[free_will]]).

## Relation to Active Inference
Active Inference models describe systems by stochastic differential equations, such as Langevin dynamics with random fluctuations, and policy selection is usually written as sampling from a softmax distribution over policies. These are modelling choices that encode uncertainty and say nothing on whether the underlying world is deterministic, since the fluctuations can be read as epistemic or as physical. The framework is therefore neutral on determinism. Questions about agency and choice are handled at the level of inference over policies, and any bearing on free will is interpretive.

## Related concepts
- [[free_will]] — the main philosophical issue raised by determinism
- [[causality]] — the notion of cause on which the thesis depends
- [[chaos_theory]] — deterministic yet unpredictable dynamics
- [[stochastic_processes]] — models of randomness used in place of exact prediction
- [[langevin_dynamics]] — noisy gradient flows underlying the free energy principle
- [[quantum_mechanics]] — the main source of doubt about determinism

## References
- Laplace, P.-S. (1814). Essai philosophique sur les probabilités. Courcier.
- Van Inwagen, P. (1983). An Essay on Free Will. Clarendon Press.
- Hoefer, C. Causal Determinism. Stanford Encyclopedia of Philosophy.
