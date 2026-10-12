---
title: Neuroimaging
description: Neuroimaging methods such as fMRI, EEG, MEG, and PET measure brain structure and activity in cognitive neuroscience, and this page covers their resolution, inferential limits, and use in Active Inference research.
type: research
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - methods
  - fmri
  - eeg
semantic_relations:
  - type: relates
    links:
      - "[[cognitive_neuroscience]]"
      - "[[computational_neuroscience]]"
      - "[[neural_synchrony]]"
aliases:
  - brain_imaging
wikidata: Q551875
---

# Neuroimaging

## Overview
Neuroimaging refers to techniques that measure the structure or activity of the living brain non-invasively, and it is the main empirical route for linking cognitive theories to human neural data. No method gives both fine spatial and fine temporal resolution, so studies choose among them and increasingly combine them.

## Methods
- **Functional MRI (fMRI)**: measures the blood-oxygenation-level-dependent (BOLD) signal (Ogawa et al., 1990), an indirect, slow correlate of neural activity. Spatial resolution is a few millimetres or better, but the hemodynamic response peaks several seconds after neural events, so temporal resolution is on the order of seconds.
- **EEG and MEG**: record electrical and magnetic fields produced by synchronized postsynaptic currents, with millisecond resolution. Source localization is an ill-posed inverse problem, so spatial resolution is coarser, especially for EEG. Oscillatory power and phase are used to study rhythms ([[neural_synchrony]]).
- **PET**: uses radioactive tracers to measure metabolism, blood flow, or receptor binding, giving molecular specificity (for example dopamine receptors) at coarse temporal resolution.
- **Structural and diffusion MRI**: anatomy, cortical thickness, and white-matter tracts, including connectivity estimates.
- **fNIRS and intracranial recordings**: portable optical measurement of hemoglobin changes, and invasive electrode recordings in patients, respectively.

## Inferential limits
Group differences and correlations are not mechanisms. A central caution is reverse inference: concluding that a cognitive process occurred because a region was active is only as strong as that region's selectivity for the process (Poldrack, 2006). Other issues include small samples, flexible analysis pipelines, multiple comparisons, and the weak relation between BOLD and underlying neural variables.

## Role in Active Inference
Neuroimaging supplies tests of predictions from the framework, such as repetition suppression and mismatch responses interpreted as prediction errors, and attention modulating precision. A distinctive contribution is dynamic causal modelling (DCM), which fits generative models of neural dynamics and the measurement process to fMRI, EEG, or MEG data and compares models by an approximation to model evidence from variational free energy (Friston, Harrison, and Penny, 2003). Such analyses estimate effective connectivity, including forward, backward, and lateral coupling ([[lateral_connections]]). Conclusions depend on the assumed model space; the same signals are often explained by other accounts, so imaging provides evidence compatible with Active Inference but rarely discriminates it from rivals.

## Related concepts
- [[cognitive_neuroscience]] — field that depends on these methods
- [[computational_neuroscience]] — modelling that is fitted to imaging data
- [[neural_synchrony]] — oscillatory phenomena measured by EEG and MEG
- [[model_comparison]] — selecting among generative models of data
- [[variational_free_energy]] — bound used for model evidence in DCM
- [[prediction_error]] — quantity interpreted from imaging signals

## References
- Ogawa, S., Lee, T. M., Kay, A. R., & Tank, D. W. (1990). Brain magnetic resonance imaging with contrast dependent on blood oxygenation. *Proceedings of the National Academy of Sciences*, 87, 9868-9872.
- Friston, K. J., Harrison, L., & Penny, W. (2003). Dynamic causal modelling. *NeuroImage*, 19, 1273-1302.
- Poldrack, R. A. (2006). Can cognitive processes be inferred from neuroimaging data? *Trends in Cognitive Sciences*, 10, 59-63.
