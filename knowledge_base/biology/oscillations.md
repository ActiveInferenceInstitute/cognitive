---
title: Neural Oscillations
description: "Rhythmic neural activity in frequency bands from delta to gamma, how it arises in circuits, proposed functions in communication and memory, and its role in predictive-coding accounts."
type: biology
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - neuroscience
  - rhythms
  - synchrony
semantic_relations:
  - type: relates
    links:
      - "[[neural_synchrony]]"
      - "[[neural_dynamics]]"
      - "[[electrophysiology]]"
aliases:
  - brain_rhythms
  - oscillation
---

# Neural Oscillations

## Overview
Neural oscillations are rhythmic fluctuations in the electrical activity of neurons and neural populations, visible in the local field potential, EEG and MEG ([[electrophysiology]]). The term here refers to brain rhythms, not to oscillations in general physical systems. They are conventionally grouped into frequency bands that differ in typical origin and association with brain state.

## Bands and generation
Approximate bands are delta (1-4 Hz), theta (4-8 Hz), alpha (8-12 Hz), beta (about 13-30 Hz) and gamma (above about 30 Hz); boundaries vary between authors. Delta is prominent in deep sleep, theta in hippocampus during movement and memory tasks, alpha over sensory cortex at rest with eyes closed, beta in sensorimotor cortex during maintained postures, and gamma during active sensory processing.

Rhythms arise from interactions of excitatory and inhibitory neurons. In gamma, fast-spiking inhibitory interneurons impose a cycle in which excitation recruits inhibition that then silences the network until it decays (pyramidal-interneuron network gamma). Slower rhythms depend on thalamocortical loops, intrinsic membrane resonance and longer-range feedback. Oscillations couple across frequencies; phase-amplitude coupling, with gamma bursts locked to theta phase, has been proposed to organize sequences of items in working memory ([[working_memory]]).

Proposed functions include communication through coherence (Fries): neuronal groups that oscillate with a consistent phase relation have aligned windows of excitability and so exchange information more effectively ([[neural_synchrony]]). Alpha has been linked to inhibition of task-irrelevant regions and to the allocation of [[attention]]. Whether oscillations are causal for computation or by-products of circuit activity is debated, and many observed correlations lack interventional tests.

## Role in Active Inference
Active Inference accounts of cortical message passing associate frequency bands with directions of signalling: prediction errors travel forward in higher (gamma) frequencies from superficial layers, predictions travel backward in lower (alpha and beta) frequencies from deep layers. Bastos and colleagues reported results in macaque visual areas consistent with this pattern, although it is not universal across areas and tasks and other interpretations exist ([[predictive_coding]], [[neural_circuits]]). Dynamic causal models of oscillatory dynamics are also used to estimate coupling parameters from recordings.

## Related concepts
- [[synchronization]] — general theory of coupled oscillators
- [[neural_coding]] — how rhythm may carry or gate information
- [[perception_attention]] — attention-related modulation of rhythms
- [[consciousness_and_awareness]] — oscillatory markers in consciousness research

## References
- Fries, P. (2005). A mechanism for cognitive dynamics: neuronal communication through neuronal coherence. *Trends in Cognitive Sciences*, 9.
- Bastos, A. M., et al. (2015). Visual areas exert feedforward and feedback influences through distinct frequency channels. *Neuron*, 85.
