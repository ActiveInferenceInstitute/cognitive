---
title: Calcium Signaling
description: Calcium ions as an intracellular messenger that links neural activity to plasticity, covering influx pathways, spatial dynamics, downstream effectors and its relevance to learning and to models of synaptic change.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neuroscience
  - cell_biology
  - plasticity
semantic_relations:
  - type: relates
    links:
      - "[[neural_plasticity]]"
      - "[[synaptic_plasticity]]"
      - "[[nmda_receptors]]"
      - "[[ltp]]"
      - "[[ltd]]"
      - "[[ion_channels]]"
---

# Calcium Signaling

## Overview
Calcium signaling is the use of calcium ions as a second messenger. Resting intracellular calcium concentration is held very low relative to the extracellular space, so a brief opening of channels or release from internal stores produces a large, rapid and often local rise that cells can detect. Berridge, Lipp and Bootman (2000) describe how the same ion regulates processes as varied as fertilisation, muscle contraction, secretion and gene transcription. In neurons it couples electrical activity to biochemical change.

## Sources and dynamics
Calcium enters through voltage-gated calcium channels and through ligand-gated channels such as the NMDA receptor, which opens only when glutamate is bound and the membrane is sufficiently depolarised. It is also released from the endoplasmic reticulum through inositol trisphosphate and ryanodine receptors. Buffers, pumps and exchangers remove it, which shapes the timing and spread of a signal. Dendritic spines act as small compartments in which calcium rises can stay local to a single synapse. The amplitude, duration and location of a calcium signal carry information, so different patterns can lead to different outcomes.

## Role in plasticity
Calcium is central to activity-dependent change in synaptic strength. A large, fast rise in a spine, typically from strong NMDA receptor activation, is associated with long-term potentiation, while a modest, prolonged rise is associated with long-term depression (Malenka and Bear, 2004). Calcium binds the protein calmodulin and activates enzymes such as CaMKII and calcineurin, which phosphorylate or dephosphorylate targets that alter receptor number and function. Calcium can also activate transcription factors, supporting longer lasting structural change. Spike-timing-dependent plasticity has been modelled with calcium-based rules, in which the sign of change depends on calcium level, although the biophysics is more complex than any single threshold model.

## Relation to Active Inference
The link is at the level of implementation. Active Inference casts learning as updating the parameters of a generative model, and Hebbian-like plasticity rules are used as candidate neural realisations of that update. Calcium dynamics are one biophysical mechanism for such rules, but the framework does not specify them, and no detailed mapping from calcium-based plasticity to free energy gradients is established.

## Related concepts
- [[neural_plasticity]] — the capacity of the nervous system to change
- [[synaptic_plasticity]] — change at the synapse that calcium helps trigger
- [[nmda_receptors]] — coincidence-detecting receptors that admit calcium
- [[ltp]] — potentiation associated with strong calcium rises
- [[ltd]] — depression associated with modest, prolonged rises

## References
- Berridge, M. J., Lipp, P., & Bootman, M. D. (2000). The versatility and universality of calcium signalling. Nature Reviews Molecular Cell Biology, 1, 11-21.
- Malenka, R. C., & Bear, M. F. (2004). LTP and LTD: An embarrassment of riches. Neuron, 44, 5-21.
