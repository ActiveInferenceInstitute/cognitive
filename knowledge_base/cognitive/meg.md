---
title: Magnetoencephalography
description: A non-invasive technique that records the weak magnetic fields produced by neural currents, its instrumentation, strengths and limits relative to EEG, and use in dynamic causal modelling.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neuroimaging
  - electrophysiology
  - methods
semantic_relations:
  - type: relates
    links:
      - "[[eeg_recording]]"
      - "[[erp]]"
      - "[[neuroimaging]]"
      - "[[electrophysiology]]"
      - "[[neural_synchrony]]"
---

# Magnetoencephalography

## Overview
Magnetoencephalography (MEG) records the magnetic fields produced outside the head by electrical currents in active neurons. The signal arises mainly from the synchronous postsynaptic currents of many pyramidal cells whose apical dendrites are aligned in parallel. Cohen (1968) made the first recordings of the fields from alpha-rhythm currents, and the technique became practical with sensitive superconducting detectors (Hämäläinen et al., 1993).

## Instrumentation
The magnetic fields of the brain are extremely small, on the order of tens to hundreds of femtotesla, many orders of magnitude weaker than ambient magnetic noise. Conventional systems use arrays of superconducting quantum interference devices (SQUIDs) cooled in liquid helium and housed in a magnetically shielded room. Newer systems use optically pumped magnetometers, which operate at room temperature and can be placed close to the scalp, which increases sensitivity and permits participants to move more freely.

## Strengths and limits
MEG has millisecond temporal resolution, like EEG, but magnetic fields are distorted less by the skull and scalp than electric potentials, which simplifies source modelling. It is most sensitive to currents tangential to the scalp, such as those in sulci, and less sensitive to radial and deep sources. Reconstructing sources from field measurements is an ill-posed inverse problem that requires assumptions, though localisation is typically better than with scalp EEG (Baillet, 2017). Cost, shielding requirements and sensitivity to movement and nearby magnetic sources are practical limits. MEG and EEG are complementary, because they weight different source orientations.

## Applications
Uses include mapping sensory and language cortex before neurosurgery, localising epileptic activity, studying oscillations and large-scale network dynamics in cognition, and tracking the time course of perception and prediction.

## Relation to Active Inference
MEG is a measurement tool, not a theory, so its link to Active Inference is methodological. Dynamic causal modelling, developed within the free-energy framework, fits generative neural mass models to MEG and EEG data in order to infer effective connectivity and compare hypotheses about message passing, including predictive-coding architectures. Frequency-specific signals recorded with MEG, such as gamma and beta rhythms, have been used to test claims about feedforward and feedback communication. Such inferences depend heavily on the assumed model.

## Related concepts
- [[eeg_recording]] — the electrical counterpart of the technique
- [[erp]] — event-related potentials, whose magnetic analogue is the event-related field
- [[neuroimaging]] — the broader family of methods
- [[electrophysiology]] — recording neural electrical activity
- [[neural_synchrony]] — synchronised activity that MEG detects well

## References
- Cohen, D. (1968). Magnetoencephalography: evidence of magnetic fields produced by alpha-rhythm currents. Science, 161, 784-786.
- Hämäläinen, M., Hari, R., Ilmoniemi, R. J., Knuutila, J., & Lounasmaa, O. V. (1993). Magnetoencephalography: theory, instrumentation, and applications to noninvasive studies of the working human brain. Reviews of Modern Physics, 65, 413-497.
- Baillet, S. (2017). Magnetoencephalography for brain electrophysiology and imaging. Nature Neuroscience, 20, 327-339.
