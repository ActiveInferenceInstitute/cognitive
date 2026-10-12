---
title: Electrophysiology
description: "Methods for recording electrical activity of neurons and brain tissue, from patch clamp and single-unit recording to LFP, EEG and ECoG, and how they constrain models of neural coding."
type: biology
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - neuroscience
  - methods
  - recording
semantic_relations:
  - type: relates
    links:
      - "[[neural_coding]]"
      - "[[single_unit]]"
      - "[[oscillations]]"
wikidata: Q1154774
---

# Electrophysiology

## Overview
Electrophysiology measures the electrical properties of cells and tissues. In neuroscience it covers recordings of membrane potential and ionic currents in single cells, spiking of neurons in living brains, and population signals recorded at the scalp or cortical surface. It supplies the empirical data against which theories of [[neural_coding]] and [[neural_dynamics]] are tested.

## Techniques by scale
Intracellular methods record the membrane potential directly. Sharp electrodes and whole-cell patch clamp, developed by Neher and Sakmann, can hold voltage (voltage clamp) to measure ion currents or hold current to observe voltage (current clamp). Extracellular microelectrodes sit near neurons and detect spikes; isolating one neuron's spikes is the basis of [[single_unit]] recording. Multi-electrode arrays and high-density probes record hundreds of channels at once.

The local field potential (LFP) is the low-frequency part of the extracellular signal and reflects summed synaptic and membrane currents in a volume of tissue. At larger scale, electrocorticography records from the cortical surface and electroencephalography (EEG) from the scalp; magnetoencephalography measures the associated magnetic fields. Spatial resolution falls and temporal resolution stays high as scale increases, with the signals dominated by synchronous activity of aligned pyramidal cells ([[oscillations]], [[neural_synchrony]]).

The theoretical anchor is the Hodgkin-Huxley model, in which the membrane voltage obeys $C\,\dot V = -g_{Na} m^3 h (V-E_{Na}) - g_K n^4 (V-E_K) - g_L (V-E_L) + I$, with gating variables $m$, $h$, $n$ following first-order kinetics. Its form is the template for most conductance-based neuron models.

## Role in Active Inference
Electrophysiology links Active Inference to data in two ways. First, dynamic causal modelling fits neural-mass generative models to EEG, MEG and LFP recordings and compares hypotheses about connectivity by Bayesian model comparison. Second, predictive-coding accounts yield testable electrophysiological signatures, such as mismatch and prediction-error responses and frequency-specific feedforward and feedback signalling. Evidence for these signatures is partial and under active debate; electrophysiology constrains the theory rather than confirming it wholesale.

## Related concepts
- [[single_unit]] — recording from isolated neurons
- [[neural_circuits]] — circuit-level organization the recordings probe
- [[neural_decoding]] — inferring stimuli or states from recorded activity
- [[predictive_coding]] — a framework generating electrophysiological predictions
- [[computational_neuroscience]] — modelling that interprets the recordings

## References
- Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. *Journal of Physiology*, 117.
- Friston, K. J., Harrison, L., & Penny, W. (2003). Dynamic causal modelling. *NeuroImage*, 19.
