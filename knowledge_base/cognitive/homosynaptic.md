---
title: Homosynaptic Plasticity
description: Activity-dependent change confined to the synapses that were stimulated, including long-term potentiation and depression, input specificity, and its role in Hebbian and prediction-based learning.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - synaptic_plasticity
  - ltp
  - learning
semantic_relations:
  - type: relates
    links:
      - "[[heterosynaptic]]"
      - "[[synaptic_plasticity]]"
      - "[[ltp]]"
      - "[[ltd]]"
      - "[[hebbian_learning]]"
wikidata: Q17030659
---

# Homosynaptic Plasticity

## Overview
Homosynaptic plasticity is a lasting change in synaptic strength that occurs at the synapses that were themselves activated, and that is specific to those inputs. Synapses on the same neuron that were not stimulated are not changed. It is the form of plasticity most commonly examined in experiments on learning and memory, and the one most directly modelled in Hebbian learning rules. It contrasts with [[heterosynaptic]] plasticity, in which change spreads to inactive synapses.

## Forms
- **Long-term potentiation (LTP).** A persistent increase in synaptic efficacy following high-frequency or correlated activity. It was first reported in the dentate area of the anaesthetised rabbit by Bliss and Lømo (1973). In many pathways, such as the Schaffer collateral input to CA1, it depends on NMDA receptors, which open only when presynaptic glutamate release coincides with postsynaptic depolarisation, so the synapse detects correlated activity (see [[ltp]]).
- **Long-term depression (LTD).** A persistent decrease in efficacy, often induced by prolonged low-frequency stimulation or by weakly correlated activity (see [[ltd]]).
- **Short-term forms.** Habituation, depression and facilitation of transmission by repeated stimulation, which in simple reflexes such as the Aplysia gill-withdrawal reflex is homosynaptic; the short-term form lasts minutes, with longer-lasting forms induced by extended training.

## Properties
Input specificity is the defining property: potentiation is restricted to the stimulated pathway. Associativity, meaning that a weakly stimulated pathway can be potentiated if paired with strong activity in another, depends on concurrent activity and is usually treated as a separate property. Spike timing can determine the sign of change, with presynaptic activity just before postsynaptic firing producing potentiation and the reverse order producing depression (see [[spike_timing_dependent_plasticity]]). Because pure Hebbian potentiation is unstable, homosynaptic changes are usually combined with homeostatic and heterosynaptic mechanisms.

## Relation to Active Inference
Active Inference learning rules for model parameters have the form of associative updates between pre- and post-synaptic quantities, weighted by prediction errors or precision. Homosynaptic changes are the natural biological candidate for implementing them, because they are local and input specific. This mapping is a hypothesis from process theories, and the biophysical details of LTP and LTD are not derived from the framework.

## Related concepts
- [[heterosynaptic]] — change at inactive synapses
- [[ltp]] — long-term increase in efficacy
- [[ltd]] — long-term decrease in efficacy
- [[hebbian_learning]] — the associative principle behind these changes
- [[spike_timing_dependent_plasticity]] — timing-dependent sign of change

## References
- Bliss, T. V. P., & Lømo, T. (1973). Long-lasting potentiation of synaptic transmission in the dentate area of the anaesthetized rabbit following stimulation of the perforant path. Journal of Physiology, 232, 331-356.
- Kandel, E. R., Koester, J. D., Mack, S. H., & Siegelbaum, S. A. (2021). Principles of Neural Science (6th ed.). New York: McGraw-Hill.
