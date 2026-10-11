---
title: Dendritic Computation
description: The processing of synaptic inputs within neuronal dendrites through passive and active properties, including nonlinear integration, local spikes and compartments, and speculative links to predictive processing.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neuroscience
  - neuron
  - computation
semantic_relations:
  - type: relates
    links:
      - "[[neural_computation]]"
      - "[[synaptic_plasticity]]"
      - "[[nmda_receptors]]"
      - "[[calcium_signaling]]"
      - "[[bayesian_brain]]"
---

# Dendritic Computation

## Overview
Dendritic computation refers to the information processing carried out within the dendrites of a neuron, before inputs reach the cell body and the site of action potential initiation. The classical point-neuron view treats a neuron as summing its inputs and comparing the sum to a threshold. Dendrites are now known to have active conductances and a structured geometry that allow integration of inputs to be nonlinear, spatially organised and dependent on timing. London and Häusser (2005) review the evidence.

## Mechanisms
- **Passive filtering.** Dendritic cable properties attenuate and delay signals, so the effect of a synapse depends on its distance from the soma and on neighbouring conductances.
- **Active conductances.** Voltage-gated sodium and calcium channels and NMDA receptors can produce regenerative dendritic spikes. NMDA receptor activation requires both glutamate and depolarisation, which makes dendrites sensitive to coincident input.
- **Compartmentalisation.** Branches can act as semi-independent subunits, so a neuron resembles a two-layer network: each branch applies a local nonlinearity, and the results are combined at the soma.
- **Local calcium signalling.** Calcium influx in a branch or spine can trigger plasticity in a restricted region, supporting branch-specific learning (see [[calcium_signaling]]).
- **Inhibition and neuromodulation.** Inhibitory synapses placed on dendrites can veto or gate excitation, and neuromodulators change dendritic excitability.

## Functional proposals
Proposals include detection of coincident inputs, direction selectivity in sensory neurons, and association of signals arriving at different dendritic locations. Larkum (2013) proposed that pyramidal neurons in neocortex associate feedforward input at basal dendrites with feedback input at distal apical dendrites through a calcium spike mechanism. Much of the evidence comes from slice preparations and from selected cell types, and the role of dendritic events in behaving animals is still under investigation.

## Relation to Active Inference
The link is speculative. Predictive coding and the Bayesian brain hypothesis assign distinct neural populations to predictions and prediction errors, and some authors have suggested that apical and basal dendritic compartments could carry top-down and bottom-up signals respectively. This is a conjectured implementation, not a finding, and the canonical microcircuit proposals remain debated.

## Related concepts
- [[neural_computation]] — computation by neurons and circuits
- [[synaptic_plasticity]] — activity-dependent change in synaptic strength
- [[nmda_receptors]] — receptors that underlie dendritic nonlinearities
- [[calcium_signaling]] — the signal coupling activity to plasticity
- [[bayesian_brain]] — the framework in which implementations are sought

## References
- London, M., & Häusser, M. (2005). Dendritic computation. Annual Review of Neuroscience, 28, 503-532.
- Larkum, M. (2013). A cellular mechanism for cortical associations: An organizing principle for the cerebral cortex. Trends in Neurosciences, 36, 141-151.
- Stuart, G., Spruston, N., & Häusser, M. (Eds.). (2016). Dendrites (3rd ed.). Oxford University Press.
