---
title: Multi-Unit Activity
description: The pooled spiking signal from several neurons near one electrode, how it differs from sorted single units and local field potentials, and when it is used.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - electrophysiology
  - neural_recording
  - spikes
semantic_relations:
  - type: relates
    links:
      - "[[single_unit]]"
      - "[[spike_sorting]]"
      - "[[spike_trains]]"
      - "[[population_coding]]"
      - "[[electrode_arrays]]"
---

# Multi-Unit Activity

## Overview
Multi-unit activity (MUA) is the combined spiking of a small local population of neurons recorded by one electrode, without separating the contributions of individual cells. It is obtained by high-pass filtering the extracellular voltage and counting events that cross a threshold. The result is an estimate of local firing intensity rather than of any one neuron's spike train.

## Relation to other signals
The extracellular voltage contains several components. Spikes from neurons close to the electrode tip appear as brief, large deflections that can be sorted into single units by waveform shape (see [[spike_sorting]] and [[single_unit]]). Spikes from more distant cells are smaller and overlap, and together form a noisy background that thresholding converts into multi-unit events. Low-frequency components form the local field potential, which reflects mainly synaptic and other transmembrane currents summed over a larger tissue volume (Buzsaki et al., 2012). MUA therefore lies between single-unit and field signals in spatial scale, with a recording radius commonly described as tens to a few hundred micrometres.

## Uses and trade-offs
MUA is more stable than single-unit isolation: sorted units are often lost as electrodes drift, whereas threshold crossings continue to exist, which is one reason chronically implanted arrays used for brain-machine interfaces often rely on them (see [[motor_prosthetics]]). Sorting is also laborious, so MUA is practical for large arrays. The cost is lost information: tuning properties of individual neurons are averaged, and a channel may mix cell types, so the interpretation of a change in MUA is less specific. Whether single-unit information offers a decisive advantage over MUA depends on the task and on how many channels are available, and results are mixed. For recording methods in general, see Buzsaki (2004).

## Relation to Active Inference
MUA is a measurement choice and has no specific link to Active Inference. It enters only indirectly, as one of the data types from which population-level predictions about neural coding are tested.

## Related concepts
- [[single_unit]] — isolated individual neurons
- [[spike_sorting]] — separating units, which MUA omits
- [[spike_trains]] — the underlying event sequences
- [[population_coding]] — how groups of neurons carry information
- [[neural_decoding]] — reading out intent or stimulus from such signals

## References
- Buzsaki, G. (2004). Large-scale recording of neuronal ensembles. Nature Neuroscience, 7, 446-451.
- Buzsaki, G., Anastassiou, C. A., & Koch, C. (2012). The origin of extracellular fields and currents: EEG, ECoG, LFP and spikes. Nature Reviews Neuroscience, 13, 407-420.
