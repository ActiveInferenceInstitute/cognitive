---
title: Temporal Coding
description: "Neural codes in which the precise timing of spikes, rather than average firing rate, carries information, with examples such as phase locking and phase precession, and links to spiking predictive coding."
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - neural-coding
  - spikes
  - timing
semantic_relations:
  - type: relates
    links:
      - "[[neural_coding]]"
      - "[[population_coding]]"
      - "[[neural_synchrony]]"
      - "[[temporal_processing]]"
---

# Temporal Coding

## Overview
A temporal code is a neural code in which information is carried by the timing of spikes at a resolution finer than the window used to count them, rather than by the number of spikes in that window (a rate code). The distinction depends on the timescale at which an experimenter or a downstream neuron can resolve, so a rigorous definition compares the information available with and without spike-timing detail.

## Forms
- **Phase locking.** Auditory nerve fibres fire at a preferred phase of a low-frequency sound waveform, so spike times encode its period and, together with interaural timing comparisons, source direction.
- **First-spike latency.** The delay between stimulus onset and the first spike can vary systematically with stimulus strength or identity, a candidate for fast sensory readout.
- **Phase of firing.** In the hippocampus, a place cell fires at progressively earlier phases of the theta rhythm as the animal crosses its place field (phase precession), so spike phase carries position information beyond rate.
- **Synchrony and correlation.** Coincident spikes across neurons can signal binding or increase downstream impact ([[neural_synchrony]], [[temporal_binding]]).
- **Ensemble sequences.** Order of firing across a population can represent sequences or trajectories ([[population_coding]]).

Rate and temporal codes are not exclusive; many systems show both, and demonstrating that downstream circuits actually use the timing is harder than showing that it contains information.

## Role in Active Inference
Standard Active Inference models usually describe neural activity with firing rates or mean-field variables, so temporal coding is not assumed. Some work implements the same prediction-error minimisation in networks of spiking units, with spike times arising from balanced excitation and inhibition (Boerlin, Machens, and Denève). Hierarchical models also use timescale separation between levels ([[temporal_models]]). Whether cortex uses precise spike timing to implement belief updating is an open empirical question, and the link is an area of active research, not an established result.

## Related concepts
- [[neural_coding]] - the broader family of codes
- [[sparse_coding]] - another efficiency-oriented code
- [[neural_decoding]] - reading out information from spikes
- [[neural_synchrony]] - coordinated timing between neurons
- [[temporal_integration]] - how neurons accumulate input over time

## References
- Theunissen, F., & Miller, J. P. (1995). Temporal encoding in nervous systems: a rigorous definition. Journal of Computational Neuroscience.
- O'Keefe, J., & Recce, M. L. (1993). Phase relationship between hippocampal place units and the EEG theta rhythm. Hippocampus.
- Boerlin, M., Machens, C. K., & Denève, S. (2013). Predictive coding of dynamical variables in balanced spiking networks. PLoS Computational Biology.
