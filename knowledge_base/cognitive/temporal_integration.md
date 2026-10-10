---
title: Temporal Integration
description: "Combining information across time in neurons and perception, from membrane time constants and Bloch's law to evidence accumulation, and the leaky-integrator form of belief updating."
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - temporal-processing
  - neural-computation
  - perception
semantic_relations:
  - type: relates
    links:
      - "[[temporal_processing]]"
      - "[[evidence_accumulation]]"
      - "[[neural_computation]]"
      - "[[belief_updating]]"
---

# Temporal Integration

## Overview
Temporal integration is the combination of inputs received at different times into a single response, estimate, or percept. The phrase is used at several levels: neurons sum synaptic input over a membrane time constant, sensory systems pool stimulus energy over a short window, and decision circuits accumulate evidence over hundreds of milliseconds or more. Linked pages use it for "time-based combination"; this page covers all three senses and keeps them separate.

## Levels
- **Cellular.** A passive membrane acts as a leaky integrator, $\tau\,\dot v = -v + I(t)$, so a constant input $I$ produces $v(t) = I\,(1 - e^{-t/\tau})$. The time constant $\tau$ sets how far apart in time two inputs can be and still sum (temporal summation).
- **Perceptual.** For brief stimuli, detection depends roughly on the product of intensity and duration (Bloch's law), up to a critical duration beyond which intensity alone matters. Visual persistence and the integration of successive frames in [[apparent_motion]] reflect similar windows. Multisensory events within a limited temporal window tend to be perceived as one ([[multisensory_integration]], [[temporal_binding]]).
- **Cognitive.** Decisions accumulate noisy evidence until a criterion is reached ([[evidence_accumulation]], [[threshold_crossing]]). Episodic memory binds elements of an experience across time ([[episodic_memory]]).
- **Trade-off.** Longer windows average out noise but blur rapid change, so the optimal window depends on the signal's statistics.

## Role in Active Inference
Belief updating in Active Inference is itself temporal integration. In the gradient-based schemes used in the process theory, the log posterior over a hidden state is updated by a leaky accumulation of log-likelihood and log-prior terms, giving exponential-type dynamics comparable to the leaky integrator above, and prediction errors over generalised coordinates integrate evidence across time ([[belief_updating]], [[generalized_coordinates]]). Whether neurons implement exactly this scheme is a modelling hypothesis tested through simulated electrophysiological responses.

## Related concepts
- [[temporal_processing]] - the broader domain of time in the nervous system
- [[evidence_accumulation]] - integration toward a decision
- [[temporal_coding]] - the information integrated may be in spike timing
- [[temporal_binding]] - integrating features and events into one episode
- [[temporal_attention]] - selecting moments for processing

## References
- Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: a process theory. Neural Computation.
