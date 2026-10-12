---
title: Excitability
description: The capacity of a neuron to generate action potentials in response to input, set by membrane properties and ion channels, and how intrinsic excitability changes with development, learning and homeostasis.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neurophysiology
  - ion_channels
  - plasticity
semantic_relations:
  - type: relates
    links:
      - "[[ion_channels]]"
      - "[[neuron_models]]"
      - "[[homeostatic_plasticity]]"
      - "[[neural_plasticity]]"
      - "[[neuromodulation]]"
      - "[[cognitive_development_expanded]]"
---

# Excitability

## Overview
Excitability is the ease with which a cell can be driven to produce an action potential. Neurons, muscle cells and some endocrine cells are excitable. For a neuron, it depends on the resting membrane potential, the voltage threshold for firing, the input resistance, and the density and properties of voltage-gated ion channels. A more excitable neuron fires in response to weaker or fewer inputs.

## Biophysical basis
Hodgkin and Huxley's analysis of the squid giant axon showed that the action potential arises from the voltage- and time-dependent conductances of sodium and potassium. Depolarisation opens sodium channels, producing regenerative inward current, and the later opening of potassium channels and inactivation of sodium channels restore the resting state. Differences in channel types and distributions among neurons produce characteristic firing patterns such as regular spiking, bursting and fast spiking. Passive properties, such as cell size and membrane resistance, also matter. Neuromodulators change excitability by acting on channels through second messengers.

## Plasticity and development
Excitability is not fixed. Intrinsic excitability changes with learning in some neurons, in parallel with synaptic changes, and these changes may store part of a memory trace (Zhang and Linden, 2003). Homeostatic mechanisms adjust excitability to keep firing rates within a working range. During development, expression of channels changes, neurons become faster and more reliable, and early spontaneous activity helps to shape circuits. Altered excitability is implicated in conditions such as epilepsy, where excessive synchronous firing occurs; the picture involves many factors and is not reducible to excitability alone.

## Relation to Active Inference
In predictive-coding implementations of Active Inference, attention and precision are proposed to be implemented by the gain of neurons that report prediction errors, which is a statement about how excitable these units are to their inputs, and neuromodulators are the proposed mediators of that gain. This is a hypothesis about neural implementation with some empirical support, not an established fact. The biophysics of excitability is independently well characterised.

## Related concepts
- [[ion_channels]] — the proteins that set excitability
- [[neuron_models]] — mathematical descriptions of firing, including Hodgkin-Huxley
- [[homeostatic_plasticity]] — regulation of excitability to stabilise activity
- [[neuromodulation]] — chemical control of neuronal gain
- [[precision_weighting]] — the Active Inference concept of gain on prediction errors

## References
- Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. Journal of Physiology, 117, 500-544.
- Hille, B. (2001). Ion Channels of Excitable Membranes (3rd ed.). Sinauer Associates.
- Zhang, W., & Linden, D. J. (2003). The other side of the engram: experience-driven changes in neuronal intrinsic excitability. Nature Reviews Neuroscience, 4, 885-900.
