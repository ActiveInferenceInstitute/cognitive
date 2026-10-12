---
title: Spike Sorting
description: The process of assigning detected action potentials in extracellular recordings to individual neurons, covering detection, feature extraction, clustering, quality metrics and its role in neural coding studies.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - electrophysiology
  - neural_coding
  - signal_processing
semantic_relations:
  - type: relates
    links:
      - "[[neural_coding]]"
      - "[[spike_trains]]"
      - "[[electrophysiology]]"
      - "[[inter_spike_intervals]]"
      - "[[firing_rate]]"
---

# Spike Sorting

## Overview
Spike sorting is the analysis step that converts a voltage trace recorded by an extracellular electrode into the spike trains of individual neurons. An electrode placed in tissue picks up action potentials from several nearby cells at once, and each cell contributes a waveform whose shape and amplitude depend on the cell's geometry and its distance and orientation relative to the contact. Because many experiments require the activity of single neurons, spikes must be attributed to their sources before quantities such as [[firing_rate]] or spike timing can be analysed.

## Typical pipeline
1. **Filtering and detection**: the signal is band-pass filtered, commonly to a range of a few hundred hertz up to a few kilohertz, and events are detected, for example when the voltage crosses a threshold set relative to an estimate of background noise.
2. **Feature extraction**: each snippet is described by a small number of features, such as peak amplitude, width, or projections onto principal components or wavelet coefficients.
3. **Clustering**: spikes are grouped in feature space, using methods ranging from manual cluster cutting to Gaussian mixture models and template matching, and the groups are interpreted as putative units.
4. **Quality assessment**: a unit is judged by its refractory period, since interspike intervals shorter than a few milliseconds should be rare for a single neuron (see [[inter_spike_intervals]]), and by measures of separation from other clusters.

Modern high-density probes record hundreds to thousands of channels, which led to automated algorithms that exploit spatial information across channels and handle overlapping spikes and drift of the electrode relative to tissue. Remaining difficulties are overlapping spikes, units with changing waveforms such as bursting cells, and the fact that sorted units are often a mix of single and multiple neurons, so the conclusions of coding studies can depend on sorting quality. Methods are validated against ground-truth recordings that combine extracellular and intracellular measurements.

## Relation to Active Inference
Spike sorting is measurement infrastructure rather than a theoretical component of Active Inference. Its relevance is indirect: tests of predictive coding and related theories of neural computation, including claims about prediction-error and expectation units (see [[predictive_coding]]), depend on reliable identification of single-unit activity, so errors in sorting can affect conclusions drawn about neural codes (see [[neural_coding]]).

## Related concepts
- [[neural_coding]] — questions that depend on sorted spike trains
- [[spike_trains]] — the sequences of spikes that sorting produces
- [[electrophysiology]] — recording methods that supply the data
- [[inter_spike_intervals]] — used to assess unit isolation
- [[firing_rate]] — a basic quantity computed from sorted spikes
- [[spike_timing]] — temporal precision that depends on accurate detection

## References
- Lewicki, M. S. (1998). A review of methods for spike sorting: The detection and classification of neural action potentials. Network: Computation in Neural Systems, 9, R53-R78.
- Quiroga, R. Q., Nadasdy, Z., & Ben-Shaul, Y. (2004). Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering. Neural Computation, 16, 1661-1687.
