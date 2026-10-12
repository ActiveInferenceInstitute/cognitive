---
title: Spike Timing
description: The use of the precise times of action potentials, rather than only firing rates, to carry or process information, with evidence, learning rules and the rate-versus-timing debate.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neural_coding
  - spiking_neurons
  - temporal_coding
semantic_relations:
  - type: relates
    links:
      - "[[temporal_coding]]"
      - "[[spike_trains]]"
      - "[[first_spike_latency]]"
      - "[[spike_timing_dependent_plasticity]]"
      - "[[neural_coding]]"
      - "[[phase_coding]]"
---

# Spike Timing

## Overview
Spike timing refers to the exact moments at which neurons fire action potentials, and to the idea that these moments, not only the average number of spikes per unit time, carry information. A rate code reads out the count of spikes in a window, whereas a timing code uses features such as the latency of the first spike after a stimulus, the phase of firing relative to an ongoing oscillation, or the relative timing among neurons. The two are not exclusive, and which one a given circuit uses is an empirical question.

## Timing codes
- **Latency coding.** Stronger stimuli often produce earlier spikes, so latency can signal intensity. Thorpe and colleagues (1996) argued that the speed of human object categorisation, within about 150 ms, leaves time for each neuron in a visual pathway to fire only a few spikes, which favours codes based on latency or rank order over long-window rate averaging (see [[first_spike_latency]]).
- **Phase coding.** Spikes occur at particular phases of a local oscillation, as in hippocampal place cells that fire at progressively earlier theta phases as an animal crosses a place field (see [[phase_coding]]).
- **Synchrony and relative timing.** Coincident spikes from several neurons can drive a target neuron more effectively than the same spikes spread out in time.

Rieke and colleagues (1997) showed in sensory systems that spike trains carry information at a timing resolution of milliseconds, and in several preparations that timing contains information beyond that in rate alone. Other work in cortex shows considerable variability in spike times across repeated trials, so the usable precision depends on the area and the stimulus.

## Role in learning
Spike-timing-dependent plasticity makes synaptic change depend on the order and interval between presynaptic and postsynaptic spikes, typically strengthening a synapse when the presynaptic spike precedes the postsynaptic spike by a few to a few tens of milliseconds, and weakening it in the reverse order. Gerstner and colleagues (1996) analysed how such a rule could train neurons to represent temporal structure at sub-millisecond precision.

## Relation to Active Inference
The standard formulations of predictive coding and Active Inference describe neural activity as continuous variables or firing rates and do not specify spike timing. Spiking implementations, in which prediction errors and beliefs are carried by spikes, are a topic of ongoing research rather than part of the core framework.

## Related concepts
- [[temporal_coding]] — the general class of timing-based codes
- [[spike_trains]] — the sequences whose timing is analysed
- [[inter_spike_intervals]] — a standard measure of spike timing
- [[spike_timing_dependent_plasticity]] — learning that depends on timing
- [[population_coding]] — information spread over many neurons

## References
- Thorpe, S., Fize, D., & Marlot, C. (1996). Speed of processing in the human visual system. Nature, 381, 520-522.
- Gerstner, W., Kempter, R., van Hemmen, J. L., & Wagner, H. (1996). A neuronal learning rule for sub-millisecond temporal coding. Nature, 383, 76-78.
- Rieke, F., Warland, D., de Ruyter van Steveninck, R., & Bialek, W. (1997). Spikes: Exploring the Neural Code. MIT Press.
