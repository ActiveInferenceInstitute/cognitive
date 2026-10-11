---
title: EEG Recording
description: Electroencephalography records scalp voltage fluctuations produced by synchronous neural activity, offering millisecond timing for studying perception, memory and prediction error but limited spatial precision.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - neuroimaging
  - methods
  - oscillations
semantic_relations:
  - type: relates
    links:
      - "[[neural_synchrony]]"
      - "[[predictive_processing]]"
      - "[[fmri_studies]]"
      - "[[prediction_error]]"
aliases:
  - eeg
---

# EEG Recording

## Overview
Electroencephalography (EEG) measures electrical potential differences at the scalp with an array of electrodes. The signal reflects summed postsynaptic currents of populations of aligned cortical pyramidal neurons that are active in synchrony, so it is a coarse and indirect readout of neural activity. Hans Berger recorded the first human EEG in 1929.

## Measurement and analysis
Electrodes are positioned by conventions such as the international 10-20 system and referenced to a common or averaged site. Recordings are non-invasive and inexpensive, with temporal resolution of milliseconds.

Two main analysis families are used:
- **Event-related potentials (ERPs).** Many epochs time-locked to a stimulus are averaged to reveal a small stimulus-related waveform buried in background activity. Well-known components include the mismatch negativity, the P300 and the N400.
- **Oscillatory analysis.** Activity is decomposed into frequency bands conventionally labelled delta (below about 4 Hz), theta (about 4 to 8 Hz), alpha (about 8 to 12 Hz), beta (about 13 to 30 Hz) and gamma (above about 30 Hz), and studied through power, phase and cross-region coherence.

Limitations: spatial resolution is low because currents are smeared by the skull and scalp, and estimating the cortical sources of a scalp pattern is an ill-posed inverse problem with no unique solution. EEG also favours superficial, geometrically aligned sources and is vulnerable to artefacts from eye movements, muscle activity and line noise. It is commonly paired with fMRI to combine temporal and spatial information.

## Role in Active Inference
EEG is a principal data source for predictive-processing hypotheses because those hypotheses make claims about timing. The mismatch negativity, elicited by a deviant in a stream of regular stimuli, has been interpreted as a prediction-error response (Garrido et al., 2009). Dynamic causal modelling can be applied to evoked EEG responses to infer connectivity between cortical sources (David et al., 2006). A frequently discussed proposal assigns gamma activity to feedforward prediction errors and beta activity to feedback predictions, which is supported by some laminar and cross-area studies but remains contested. Such interpretations are model-dependent: an ERP difference alone does not identify prediction error uniquely.

## Related concepts
- [[fmri_studies]] — complementary imaging method with higher spatial resolution
- [[neural_synchrony]] — oscillatory coupling measured from EEG
- [[hyperscanning]] — simultaneous EEG from interacting people
- [[perceptual_inference]] — perceptual phenomena probed with ERPs
- [[prediction_error]] — the quantity proposed to underlie mismatch responses

## References
- Garrido, M. I., Kilner, J. M., Stephan, K. E., & Friston, K. J. (2009). The mismatch negativity: a review of underlying mechanisms. Clinical Neurophysiology.
- David, O., et al. (2006). Dynamic causal modeling of evoked responses in EEG and MEG. NeuroImage.
