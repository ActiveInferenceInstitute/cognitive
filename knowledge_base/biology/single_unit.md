---
title: Single-Unit Recording
description: "Extracellular recording of action potentials from individual neurons, how spikes are sorted and tuning is characterized, landmark findings, and its use in testing predictive-coding theories."
type: biology
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - neuroscience
  - methods
  - neural-coding
semantic_relations:
  - type: relates
    links:
      - "[[electrophysiology]]"
      - "[[neural_coding]]"
      - "[[population_coding]]"
aliases:
  - single_unit_recording
wikidata: Q4392016
---

# Single-Unit Recording

## Overview
Single-unit recording measures the spiking activity of one neuron at a time, usually with a fine microelectrode placed outside the cell in the brain of an awake or anaesthetized animal. It is the technique that established much of what is known about what individual neurons represent, and it is a main branch of [[electrophysiology]].

## Method
An electrode near a neuron picks up the extracellular voltage transient of each action potential. Because one electrode usually detects several neurons, spikes are separated by spike sorting, which clusters waveforms by shape and amplitude; tetrodes and high-density silicon probes make clustering more reliable by recording the same spike on several nearby channels. Isolation quality varies, so recordings are classified as well-isolated single units or as multi-unit activity.

The basic analysis relates firing rate or spike timing to stimulus or behavior. A tuning curve plots firing rate against a stimulus parameter, such as orientation. Peri-stimulus time histograms show rate relative to an event. Spike trains are often modelled as point processes, with Poisson variability as a reference. Landmark findings come from this method: Hubel and Wiesel's orientation-selective cells in cat visual cortex; O'Keefe and Dostrovsky's hippocampal place cells in rats; grid cells in entorhinal cortex; and neurons in monkey premotor cortex that respond both when an animal performs and when it observes an action ([[action_recognition]]).

Limitations deserve attention. Recordings sample neurons with large, easily detected spikes and may miss small or silent cells; experimenters often search for responsive cells, which biases what is reported. A single neuron's tuning may not describe how a population encodes information, which is why modern work records many neurons at once, as with Neuropixels probes with hundreds of recording sites, and analyses the population ([[population_coding]], [[neural_decoding]]).

## Role in Active Inference
Single-unit data test predictions of [[predictive_coding]]. If cortex computes prediction errors, some neurons should respond to mismatch between expected and actual input. Studies in mouse visual cortex have found neurons whose responses are consistent with signed prediction error and with mismatch between visual flow and movement, reviewed by Keller and Mrsic-Flogel. Interpretation is contested, because other mechanisms such as adaptation and attention can produce similar response patterns.

## Related concepts
- [[neural_circuits]] — circuits whose members are sampled by unit recording
- [[sparse_coding]] — theory of how few active units carry information
- [[visual_neuroscience]] — field built on tuning measurements
- [[neural_efficiency]] — resource-constrained coding in neurons

## References
- Hubel, D. H., & Wiesel, T. N. (1962). Receptive fields, binocular interaction and functional architecture in the cat's visual cortex. *Journal of Physiology*, 160.
- Keller, G. B., & Mrsic-Flogel, T. D. (2018). Predictive processing: a canonical cortical computation. *Neuron*, 100.
