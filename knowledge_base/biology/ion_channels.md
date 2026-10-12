---
title: Ion Channels
description: Pore-forming membrane proteins that let ions cross cell membranes down their electrochemical gradients, their gating and selectivity, and their role in excitability, signalling and models of neurons.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - membrane_proteins
  - excitability
  - electrophysiology
semantic_relations:
  - type: relates
    links:
      - "[[calcium_signaling]]"
      - "[[nmda_receptors]]"
      - "[[neuromuscular_junction]]"
      - "[[synaptic_plasticity]]"
      - "[[cell_biology]]"
      - "[[protein_structure]]"
aliases:
  - ion_channel
---

# Ion Channels

## Overview
Ion channels are membrane proteins that form aqueous pores through which ions pass across the lipid bilayer. Flow is passive and follows the electrochemical gradient, set up by pumps and transporters that use energy. Channels can open and close in milliseconds, and the resulting currents underlie electrical signalling in neurons, muscle and many other cells.

## Selectivity and gating
Channels discriminate between ions. Potassium channels, for instance, pass K$^+$ while excluding the smaller Na$^+$, a property explained by the crystal structure of the bacterial channel KcsA, in which a selectivity filter coordinates dehydrated K$^+$ ions (Doyle et al., 1998). Gating is the opening and closing of the pore and is controlled by different stimuli:
- **Voltage-gated** channels respond to membrane potential and include the Na$^+$, K$^+$ and Ca$^{2+}$ channels of action potentials.
- **Ligand-gated** channels open when a molecule binds, as with nicotinic and NMDA receptors (see [[nmda_receptors]]).
- **Mechanosensitive** and temperature-sensitive channels respond to stretch or heat.

The equilibrium potential for an ion $X$ is given by the Nernst equation, $E_X=\frac{RT}{zF}\ln\frac{[X]_o}{[X]_i}$, where $[X]_o$ and $[X]_i$ are outside and inside concentrations and $z$ is the valence. Hodgkin and Huxley (1952) described the membrane current of the squid giant axon as a sum of ionic conductances, written as $C_m\,dV/dt=-\sum_i g_i(V-E_i)+I$, where $g_i$ are voltage-dependent. The framework remains the basis of conductance-based neuron models (Hille, 2001).

## Disease and drugs
Mutations in channel genes cause channelopathies, including some forms of epilepsy, cardiac arrhythmia and cystic fibrosis. Many drugs, toxins and anaesthetics act directly on channels.

## Relation to Active Inference
The relationship is through modelling practice. Biophysical neuron models with ion channel conductances are used in generative models of neural activity such as those employed in dynamic causal modelling, and the channels realise the membrane-potential dynamics that the Free Energy Principle describes at a more abstract level (see [[free_energy_principle]]). The principle itself does not require channel-level detail.

## Related concepts
- [[calcium_signaling]] — intracellular signalling that depends on calcium channels
- [[nmda_receptors]] — ligand-gated and voltage-dependent glutamate channels
- [[neuromuscular_junction]] — a synapse that relies on ligand-gated and voltage-gated channels
- [[synaptic_plasticity]] — activity-dependent change that depends on channel currents
- [[protein_structure]] — structural basis of channel selectivity and gating

## References
- Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. Journal of Physiology, 117, 500-544.
- Doyle, D. A., et al. (1998). The structure of the potassium channel: molecular basis of K+ conduction and selectivity. Science, 280, 69-77.
- Hille, B. (2001). Ion Channels of Excitable Membranes (3rd ed.). Sinauer Associates.
