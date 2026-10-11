---
title: NMDA Receptors
description: Glutamate-gated ion channels that act as coincidence detectors of presynaptic release and postsynaptic depolarisation, central to long-term potentiation, learning and several predictive-coding hypotheses.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - glutamate
  - synaptic_plasticity
  - receptors
semantic_relations:
  - type: relates
    links:
      - "[[ltp]]"
      - "[[synaptic_plasticity]]"
      - "[[hebbian_learning]]"
      - "[[spike_timing_dependent_plasticity]]"
      - "[[ion_channels]]"
      - "[[calcium_signaling]]"
---

# NMDA Receptors

## Overview
N-methyl-D-aspartate (NMDA) receptors are ionotropic receptors for the excitatory neurotransmitter glutamate. They are found at most excitatory synapses in the brain, where they work alongside fast AMPA receptors. Their unusual properties make them a molecular implementation of the Hebbian principle that synapses strengthen when pre- and postsynaptic activity coincide (see [[hebbian_learning]]).

## Properties
- **Structure.** The receptors are heteromeric tetramers, typically two GluN1 subunits with two GluN2 subunits (types A to D), whose composition affects kinetics and pharmacology (Traynelis et al., 2010).
- **Dual requirement.** Activation needs glutamate and a co-agonist, glycine or D-serine.
- **Voltage-dependent block.** At resting potential the pore is plugged by extracellular Mg$^{2+}$, and depolarisation relieves the block (Nowak et al., 1984).
- **Calcium permeability.** An open receptor passes Na$^+$, K$^+$ and Ca$^{2+}$, and the calcium entry acts as a second messenger (see [[calcium_signaling]]).

The receptor therefore opens substantially only when glutamate is released and the postsynaptic cell is already depolarised, which makes it a coincidence detector.

## Role in plasticity and learning
Calcium entering through NMDA receptors triggers kinases and phosphatases that regulate AMPA receptor number and function, leading to long-term potentiation or depression (see [[ltp]]). Blocking NMDA receptors in the hippocampus impairs induction of LTP and spatial learning in rodents (Bliss & Collingridge, 1993). Overactivation, however, is toxic to neurons. Channel blockers such as ketamine and memantine are used clinically, and autoimmune antibodies against the receptor cause an encephalitis.

## Relation to Active Inference
NMDA receptors are discussed in predictive-processing accounts as mediators of synaptic gain and of voltage-dependent coupling among cortical levels, and the effects of NMDA antagonists have been used to test models of psychosis and altered perception. This is an active research programme in computational psychiatry, and the specific claims remain hypotheses under test. Plasticity in Active Inference models is generally more abstract than the receptor mechanism.

## Related concepts
- [[ltp]] — the persistent strengthening that depends on NMDA receptor activation
- [[synaptic_plasticity]] — the general class of changes in synaptic strength
- [[hebbian_learning]] — the learning rule the receptor implements
- [[spike_timing_dependent_plasticity]] — plasticity depending on spike timing
- [[ion_channels]] — the broader class of membrane channels

## References
- Nowak, L., Bregestovski, P., Ascher, P., Herbet, A., & Prochiantz, A. (1984). Magnesium gates glutamate-activated channels in mouse central neurones. Nature, 307, 462-465.
- Bliss, T. V. P., & Collingridge, G. L. (1993). A synaptic model of memory: long-term potentiation in the hippocampus. Nature, 361, 31-39.
- Traynelis, S. F., et al. (2010). Glutamate receptor ion channels: structure, regulation, and function. Pharmacological Reviews, 62, 405-496.
