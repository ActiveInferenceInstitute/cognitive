---
title: Event-Related Potentials
description: Averaged EEG voltage changes time-locked to sensory, cognitive or motor events, their main components, strengths and limits, and their interpretation as prediction-error signals.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - electrophysiology
  - eeg
  - methods
semantic_relations:
  - type: relates
    links:
      - "[[eeg_recording]]"
      - "[[electrophysiology]]"
      - "[[meg]]"
      - "[[error_monitoring]]"
      - "[[prediction_error]]"
---

# Event-Related Potentials

## Overview
An event-related potential (ERP) is a small, stereotyped change in scalp-recorded electrical activity that is time-locked to a sensory, cognitive or motor event. Single-trial responses are buried in ongoing EEG, so the signal is extracted by averaging many trials aligned to the event, which attenuates activity unrelated to it (Luck, 2014). The result is a waveform with positive and negative deflections whose amplitude, latency and scalp distribution are interpreted.

## Components
Components are named by polarity and either latency or order. Early components such as the N1 and P1 reflect sensory processing and are modulated by attention. The P3 or P300, a late positivity evoked by rare, task-relevant stimuli, was described by Sutton and colleagues (1965) in work on stimulus uncertainty. The N400, a negativity peaking about 400 ms after a word, is enlarged for semantically unexpected words (Kutas and Hillyard, 1980). The mismatch negativity follows a deviant sound within a sequence of repeated sounds, even without attention. The error-related negativity follows mistakes in speeded tasks (see [[error_monitoring]]).

## Strengths and limits
ERPs offer millisecond temporal resolution, are non-invasive and inexpensive, and can be recorded without overt responses. Their spatial resolution is limited, because the same scalp pattern can be produced by many arrangements of sources, an inverse problem that requires modelling assumptions. Components can overlap in time, so a difference between conditions does not always reflect a change in a single process, and averaging hides trial-to-trial variability. Interpretation therefore requires careful design and controls.

## Relation to Active Inference
Active Inference and predictive coding give ERPs a specific interpretation: many components are read as prediction-error signals or as updates in beliefs after a violated expectation. The mismatch negativity is the best-studied case; it has been modelled as a change in synaptic efficacy and top-down prediction as a stimulus repeats (Garrido et al., 2009). Dynamic causal modelling, a Bayesian method developed in the same framework, fits neural mass models to evoked responses to infer connectivity (David et al., 2006). This is an active research programme with considerable empirical work, but other accounts of mismatch negativity, based on neural adaptation, compete with it.

## Related concepts
- [[eeg_recording]] — the measurement technique from which ERPs are derived
- [[electrophysiology]] — recording of neural electrical activity
- [[meg]] — the magnetic counterpart, giving event-related fields
- [[error_monitoring]] — a domain where ERP components are widely used
- [[prediction_error]] — the quantity many components are proposed to reflect

## References
- Sutton, S., Braren, M., Zubin, J., & John, E. R. (1965). Evoked-potential correlates of stimulus uncertainty. Science, 150, 1187-1188.
- Kutas, M., & Hillyard, S. A. (1980). Reading senseless sentences: brain potentials reflect semantic incongruity. Science, 207, 203-205.
- Luck, S. J. (2014). An Introduction to the Event-Related Potential Technique (2nd ed.). MIT Press.
- Garrido, M. I., Kilner, J. M., Stephan, K. E., & Friston, K. J. (2009). The mismatch negativity: a review of underlying mechanisms. Clinical Neurophysiology, 120, 453-463.
- David, O., Kiebel, S. J., Harrison, L. M., Mattout, J., Kilner, J. M., & Friston, K. J. (2006). Dynamic causal modeling of evoked responses in EEG and MEG. NeuroImage, 30, 1255-1272.
