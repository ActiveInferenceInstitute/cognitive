---
title: Voltage Channels
description: Ion channels whose opening is controlled by membrane potential, including sodium, potassium and calcium channels, their role in action potentials and calcium entry, and their importance for synaptic plasticity.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - ion_channels
  - excitability
  - neuroscience
semantic_relations:
  - type: relates
    links:
      - "[[ion_channels]]"
      - "[[calcium_signaling]]"
      - "[[transmitter_release]]"
      - "[[nmda_receptors]]"
      - "[[synaptic_plasticity]]"
---

# Voltage Channels

## Overview
Voltage-gated ion channels are membrane proteins that form a selective pore which opens or closes in response to changes in the transmembrane potential. They are central to electrical signalling in neurons, muscle and many other cells. The main families are voltage-gated sodium, potassium and calcium channels, each with several subtypes (Hille, 2001).

## Mechanism
The channel contains voltage-sensing domains rich in charged residues that move in the membrane field when the potential changes, and this movement is coupled to a gate. Hodgkin and Huxley (1952) described the currents underlying the squid axon action potential in terms of voltage- and time-dependent conductances. Sodium channels open on depolarisation and then inactivate, giving the rapid upstroke and a refractory period. Potassium channels open more slowly and repolarise the membrane. Their later identification as molecular entities confirmed the broad outlines of this scheme (Catterall, 2000).

## Calcium entry
Voltage-gated calcium channels open on depolarisation and admit calcium, a key intracellular signal (Catterall, 2011). Different subtypes are positioned at particular sites: those at presynaptic terminals trigger [[transmitter_release]], and those on dendrites and cell bodies couple electrical activity to calcium-dependent kinases and gene expression (see [[calcium_signaling]]). In dendritic spines, calcium also enters through [[nmda_receptors]], which depend on both ligand binding and depolarisation to relieve a magnesium block, so they act as a coincidence detector, and voltage-gated calcium channels contribute to calcium signals in some forms of plasticity, particularly those requiring strong postsynaptic depolarisation.

## Regulation and disease
Channel properties are modulated by phosphorylation, G proteins and accessory subunits, which is one route by which neuromodulators change excitability (see [[receptor_phosphorylation]]). Mutations in channel genes are linked to disorders of excitability such as some epilepsies, pain syndromes and cardiac arrhythmias.

## Relation to Active Inference
The link is to biophysical implementation. Active Inference does not model individual channels. Detailed conductance-based models of neurons provide the physical substrate on which message passing in neural-mass and predictive-coding models is thought to be implemented, but the mapping from channel dynamics to the variables of a generative model is not fixed by the theory.

## Related concepts
- [[ion_channels]] — the wider class of channels
- [[calcium_signaling]] — the signalling that calcium entry initiates
- [[transmitter_release]] — presynaptic calcium channels couple spikes to release
- [[nmda_receptors]] — ligand-gated receptors with voltage-dependent block
- [[synaptic_plasticity]] — activity-dependent calcium entry drives plasticity

## References
- Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. Journal of Physiology, 117, 500-544.
- Hille, B. (2001). Ion Channels of Excitable Membranes (3rd ed.). Sinauer Associates.
- Catterall, W. A. (2011). Voltage-gated calcium channels. Cold Spring Harbor Perspectives in Biology, 3, a003947.
- Catterall, W. A. (2000). From ionic currents to molecular mechanisms: the structure and function of voltage-gated sodium channels. Neuron, 26, 13-25.
