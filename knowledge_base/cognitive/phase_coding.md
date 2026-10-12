---
title: Phase Coding
description: The representation of information in the timing of neural spikes relative to the phase of an ongoing network oscillation, as in hippocampal theta phase precession and proposed theta-gamma codes.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neural_coding
  - oscillations
  - hippocampus
semantic_relations:
  - type: relates
    links:
      - "[[neural_coding]]"
      - "[[oscillations]]"
      - "[[temporal_coding]]"
      - "[[hippocampus]]"
      - "[[neural_synchrony]]"
      - "[[grid_cells]]"
---

# Phase Coding

## Overview
Phase coding, or phase-of-firing coding, is a scheme in which information is conveyed not only by how many spikes a neuron fires but by when they occur relative to the cycle of a local oscillation. The oscillation acts as a reference clock. Two neurons with the same firing rate can carry different information if their spikes fall at different phases of the cycle. It is one member of the family of temporal codes ([[temporal_coding]]) and complements rate-based descriptions ([[neural_coding]]).

## Hippocampal phase precession
The best-known example is theta phase precession. When a rat runs through the firing field of a hippocampal place cell, the cell fires at progressively earlier phases of the 6-10 Hz theta rhythm in successive cycles (O'Keefe and Recce, 1993). Phase therefore carries information about position within the field that is not available from the firing rate alone.

## Theta-gamma coding
Lisman and Jensen (2013) proposed that items in a short sequence are represented in successive gamma cycles nested inside one theta cycle, so that the number of gamma cycles per theta cycle limits the number of items held in working memory. The proposal is influential but remains a hypothesis, and the evidence in humans is largely correlational. In neocortex, recordings in macaque primary visual cortex have reported that the phase of a low-frequency local field potential at which neurons fire carries information about natural stimuli beyond that in spike counts (Montemurro et al., 2008).

## Debates
Phase codes require that downstream neurons can read out timing relative to a reference, and it is debated whether the brain does this or whether the observed effects are by-products of network dynamics. Oscillation phase is also correlated with firing rate in many circuits, so showing that phase carries independent information needs careful controls (see [[oscillations]]).

## Relation to Active Inference
The relation is that of a candidate neural implementation, not a derivation. Proposals for the cortical implementation of predictive coding assign prediction errors and predictions to different cortical layers (Bastos et al., 2012), and related work associates faster gamma activity with superficial layers and slower alpha or beta activity with deep layers. Such a division could use oscillation phase to organise message passing, but the phase-coding claims discussed here originate in hippocampal and working-memory research, and they have not been formally integrated into the framework.

## Related concepts
- [[neural_coding]] — the general problem of how neural activity represents information
- [[oscillations]] — the rhythmic population activity that supplies the reference phase
- [[temporal_coding]] — codes that use spike timing
- [[hippocampus]] — the structure where theta phase precession was found
- [[neural_synchrony]] — coordinated timing between neurons

## References
- O'Keefe, J., & Recce, M. L. (1993). Phase relationship between hippocampal place units and the EEG theta rhythm. Hippocampus, 3, 317-330.
- Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical networks. Science, 304, 1926-1929.
- Lisman, J. E., & Jensen, O. (2013). The theta-gamma neural code. Neuron, 77, 1002-1016.
- Bastos, A. M., Usrey, W. M., Adams, R. A., Mangun, G. R., Fries, P., & Friston, K. J. (2012). Canonical microcircuits for predictive coding. Neuron, 76, 695-711.
- Montemurro, M. A., Rasch, M. J., Murayama, Y., Logothetis, N. K., & Panzeri, S. (2008). Phase-of-firing coding of natural visual stimuli in primary visual cortex. Current Biology, 18, 375-380.
