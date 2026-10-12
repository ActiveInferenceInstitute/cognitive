---
title: Calcium Imaging
description: Optical recording of neuronal activity using fluorescent calcium indicators, covering indicators, microscopy methods, interpretation limits, and its use in studying plasticity and population coding.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neuroscience
  - methods
  - imaging
semantic_relations:
  - type: relates
    links:
      - "[[calcium_signaling]]"
      - "[[neural_plasticity]]"
      - "[[optogenetics]]"
      - "[[neuroimaging]]"
      - "[[computational_neuroscience]]"
wikidata: Q5018833
---

# Calcium Imaging

## Overview
Calcium imaging records neuronal activity optically by monitoring the concentration of intracellular calcium with fluorescent indicators. Action potentials open voltage-gated calcium channels, so calcium concentration in the cell rises after firing and the indicator's fluorescence changes accordingly. The method allows many neurons, or subcellular structures such as dendritic spines, to be observed at once in living tissue.

## Indicators and microscopy
Early indicators were synthetic dyes loaded into cells. Genetically encoded indicators, such as the GCaMP family built from a fluorescent protein and a calcium-binding domain, can be expressed in defined cell types and followed over days or weeks. Successive engineered versions have improved sensitivity and speed (Chen et al., 2013). Imaging is performed with wide-field and confocal microscopes, two-photon microscopes that image deeper in scattering tissue, light-sheet systems, and head-mounted miniature microscopes for freely moving animals. Fibre photometry records bulk fluorescence from a population.

## Interpretation and limits
Calcium is an indirect and slow readout of spiking. Indicator kinetics blur the timing of action potentials, the relationship between spike number and fluorescence is nonlinear and varies between indicators and cell types, and the indicator itself buffers calcium. Spike inference methods estimate the underlying activity from the fluorescence trace, with imperfect accuracy. Electrophysiology remains the reference for precise spike timing, and calcium imaging trades temporal precision for large-scale cellular coverage.

## Relation to Active Inference
Calcium imaging supplies data, not theory. Large-scale recordings of identified neurons are used to test predictive coding and related ideas, for example by asking whether specific cell populations signal prediction errors during learning, and to constrain computational models of plasticity. The connection to Active Inference is therefore indirect: the method makes some of its neuronal predictions testable, while the theory does not depend on any particular recording technique. Calcium also acts as a signalling ion in [[neural_plasticity]], so imaging dendritic calcium informs studies of synaptic learning rules.

## Related concepts
- [[calcium_signaling]] — the intracellular signalling that the method reports on
- [[neural_plasticity]] — a field that uses imaging to track synaptic and cellular change
- [[optogenetics]] — a complementary method for controlling activity with light
- [[neuroimaging]] — larger-scale imaging methods in humans
- [[population_coding]] — what large-scale recordings are used to study

## References
- Denk, W., Strickler, J. H., & Webb, W. W. (1990). Two-photon laser scanning fluorescence microscopy. Science, 248, 73-76.
- Grienberger, C., & Konnerth, A. (2012). Imaging calcium in neurons. Neuron, 73, 862-885.
- Chen, T.-W., et al. (2013). Ultrasensitive fluorescent proteins for imaging neuronal activity. Nature, 499, 295-300.
