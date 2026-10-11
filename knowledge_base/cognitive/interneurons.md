---
title: Interneurons
description: Neurons that form local connections within a circuit, mainly inhibitory GABAergic cells in cortex, their major classes and functions, and their proposed roles in predictive coding.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neural_circuits
  - inhibition
  - cortex
semantic_relations:
  - type: relates
    links:
      - "[[local_circuits]]"
      - "[[cortical_columns]]"
      - "[[gain_control]]"
      - "[[neural_synchrony]]"
      - "[[predictive_coding]]"
---

# Interneurons

## Overview
Interneurons are neurons whose axons remain within a local circuit instead of projecting to distant regions. In the neocortex and hippocampus the term usually refers to inhibitory cells that release the neurotransmitter GABA, which form a minority of cortical neurons (roughly a fifth) but have a large influence on circuit dynamics. Some excitatory local neurons, such as spiny stellate cells in cortical layer 4, are also sometimes called interneurons, so usage varies with context.

## Classes
Cortical inhibitory interneurons are diverse in morphology, connectivity, physiology and molecular markers (Markram et al., 2004). Three broad classes cover most of them, defined by markers.
- **Parvalbumin-expressing cells**, including basket cells, are fast-spiking and target the cell bodies and proximal dendrites of pyramidal cells, providing strong perisomatic inhibition.
- **Somatostatin-expressing cells**, including Martinotti cells, target distal dendrites and regulate dendritic integration.
- **VIP-expressing cells** mainly inhibit other interneurons, particularly somatostatin cells, and so disinhibit pyramidal cells.

Kepecs and Fishell (2014) argue that cell types should be understood in terms of the functions they perform in circuits, and that the diversity reflects specialisation for distinct computations.

## Functions
Interneurons balance excitation, prevent runaway activity, and sharpen tuning. They control the gain of principal cells, support divisive normalisation, shape timing and spike synchrony, and contribute to the generation of gamma-band rhythms, notably through parvalbumin cells. Disturbed inhibition is implicated in epilepsy, and altered interneuron function is discussed in schizophrenia and autism, though the causal story remains uncertain.

## Relation to Active Inference
In predictive coding, prediction error is computed by subtracting predictions from incoming activity, a role often assigned to inhibitory connections, and the precision of error signals is thought to be regulated by gain control. Canonical microcircuit proposals therefore assign inhibitory interneurons to computing error and setting gain (Bastos et al., 2012). Studies in mouse visual cortex suggest that the VIP, somatostatin and pyramidal cell motif could support the comparison of top-down predictions with bottom-up input (Keller and Mrsic-Flogel, 2018). These are active research hypotheses supported by partial evidence, and alternative computational roles for the same cell types are also proposed.

## Related concepts
- [[local_circuits]] — the circuits in which interneurons operate
- [[cortical_columns]] — the laminar modules that contain them
- [[gain_control]] — a main function of inhibition
- [[neural_synchrony]] — coordinated activity shaped by inhibition
- [[predictive_coding]] — the framework assigning them to error and precision computation
- [[prediction_error]] — the signal that inhibition is proposed to compute

## References
- Markram, H., et al. (2004). Interneurons of the neocortical inhibitory system. Nature Reviews Neuroscience, 5, 793-807.
- Kepecs, A., & Fishell, G. (2014). Interneuron cell types are fit to function. Nature, 505, 318-326.
- Bastos, A. M., et al. (2012). Canonical microcircuits for predictive coding. Neuron, 76, 695-711.
- Keller, G. B., & Mrsic-Flogel, T. D. (2018). Predictive processing: a canonical cortical computation. Neuron, 100, 424-435.
