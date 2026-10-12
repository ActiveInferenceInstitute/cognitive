---
title: Spike-Timing-Dependent Plasticity
description: A form of Hebbian synaptic plasticity in which the sign and size of a change depend on the relative timing of pre- and postsynaptic spikes, and its relation to free-energy minimisation accounts of learning.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - synaptic_plasticity
  - learning_rule
  - neuroscience
semantic_relations:
  - type: relates
    links:
      - "[[hebbian_learning]]"
      - "[[synaptic_plasticity]]"
      - "[[ltp]]"
      - "[[ltd]]"
      - "[[free_energy_principle]]"
aliases:
  - spike_timing_plasticity
  - stdp
---

# Spike-Timing-Dependent Plasticity

## Overview
Spike-timing-dependent plasticity (STDP) is a family of learning rules in which the change in strength of a synapse depends on the precise relative timing of spikes in the presynaptic and postsynaptic neurons. In the canonical form, a presynaptic spike that arrives shortly before a postsynaptic spike strengthens the synapse (potentiation), and the reverse order weakens it (depression). The relevant time window is of the order of tens of milliseconds. STDP refines Hebb's postulate by adding causal order to coincidence.

## Experimental basis and mechanism
- **Evidence.** Markram and colleagues (1997) showed that pairing postsynaptic action potentials with synaptic inputs in cortical neurons changed synaptic efficacy depending on which came first. Bi and Poo (1998) mapped the dependence on spike-time difference in cultured hippocampal neurons and found potentiation for pre-before-post and depression for post-before-pre.
- **Mechanism.** Calcium entry through NMDA receptors, shaped by back-propagating action potentials, is widely thought to determine the direction of change, with larger and faster calcium rises favouring potentiation and smaller ones favouring depression (see [[ltp]] and [[ltd]]).
- **Variability.** The shape of the learning window differs among synapse types, brain regions and firing rates, and in vivo conditions with many interacting spikes do not always reproduce pairwise results. STDP is therefore best viewed as a class of rules rather than one law.

## Relation to Active Inference
The link is an active research programme rather than a settled derivation. Under the free-energy principle, synaptic changes are expected to follow gradients of variational free energy, and Hebbian-like rules emerge naturally when neurons are modelled as performing inference. Isomura and Friston (2018) reported that cultured neuronal networks self-organised in ways consistent with minimising variational free energy, and that their plasticity was compatible with such a scheme. Specific claims that STDP implements a particular free-energy gradient depend on modelling assumptions, and other normative accounts, such as temporal-difference learning, also reproduce STDP-like windows. See [[hebbian_learning]] and [[predictive_coding]] for related treatments.

## Related concepts
- [[hebbian_learning]] — the coincidence principle that STDP extends
- [[synaptic_plasticity]] — the broader class of activity-dependent synaptic change
- [[ltp]] — long-term potentiation, the strengthening arm of the rule
- [[ltd]] — long-term depression, the weakening arm of the rule
- [[free_energy_principle]] — the framework in which plasticity is cast as inference
- [[predictive_coding]] — a scheme in which local learning rules arise from error minimisation

## References
- Markram, H., Lubke, J., Frotscher, M., & Sakmann, B. (1997). Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs. Science, 275, 213-215.
- Bi, G.-Q., & Poo, M.-M. (1998). Synaptic modifications in cultured hippocampal neurons: Dependence on spike timing, synaptic strength, and postsynaptic cell type. Journal of Neuroscience, 18, 10464-10472.
- Isomura, T., & Friston, K. (2018). In vitro neural networks minimise variational free energy. Scientific Reports, 8, 16926.
