---
title: Transmitter Release
description: The calcium-triggered fusion of synaptic vesicles at the presynaptic terminal that releases neurotransmitter, its quantal nature, regulation, and modulation in synaptic plasticity.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - synaptic_transmission
  - neuroscience
  - presynaptic_function
semantic_relations:
  - type: relates
    links:
      - "[[synaptic_plasticity]]"
      - "[[voltage_channels]]"
      - "[[vesicle_traffic]]"
      - "[[neuromuscular_junction]]"
      - "[[neural_plasticity]]"
---

# Transmitter Release

## Overview
Transmitter release is the process by which a presynaptic neuron releases neurotransmitter into the synaptic cleft in response to an action potential. Neurotransmitter is stored in synaptic vesicles, membrane-bound organelles clustered near the active zone. Depolarisation opens voltage-gated calcium channels, calcium enters the terminal, and vesicles fuse with the plasma membrane to discharge their contents by exocytosis (Südhof, 2013); the calcium dependence and quantal character of release were established largely through Katz and colleagues' work on the neuromuscular junction (Katz, 1969).

## Sequence of events
1. An action potential invades the terminal and opens [[voltage_channels]] selective for calcium.
2. Calcium rises steeply in a small domain near the channels and binds sensors, notably synaptotagmins, that couple the signal to the fusion machinery.
3. SNARE proteins on vesicle and plasma membranes assemble and drive membrane fusion, assisted by regulators such as Munc18 and Munc13 (Südhof, 2013).
4. Transmitter diffuses across the cleft and binds postsynaptic receptors.
5. Vesicle membrane is retrieved by endocytosis and vesicles are refilled and recycled (see [[vesicle_traffic]]).

## Quantal nature and regulation
Work on the neuromuscular junction by Katz and colleagues showed that release occurs in multiples of a basic unit, the quantum, which corresponds to the content of a vesicle (see [[neuromuscular_junction]]). Release at central synapses is probabilistic: a vesicle is released after a given spike with a probability that varies between synapses and depends on calcium entry, vesicle pool size and recent activity. Short-term facilitation and depression reflect this dependence on recent history. Presynaptic receptors and neuromodulators alter release probability, and long-term changes in the presynaptic terminal contribute to some forms of lasting plasticity (see [[synaptic_plasticity]]).

## Relation to Active Inference
The relationship is one of implementation. In predictive-coding accounts, the synaptic gain applied to a signal is often associated with the precision of its prediction error, and the probabilistic, modulated character of release is a candidate physical basis for such gain control. That mapping is a hypothesis in the theoretical literature rather than a tested result, and the details of release machinery do not feature in the formal theory.

## Related concepts
- [[voltage_channels]] — channels whose opening triggers release
- [[vesicle_traffic]] — the vesicle cycle that supplies and retrieves release-ready vesicles
- [[neuromuscular_junction]] — the synapse where quantal release was established
- [[synaptic_plasticity]] — release probability changes as a mechanism of plasticity
- [[neuromodulation]] — regulation of release by modulators

## References
- Katz, B. (1969). The Release of Neural Transmitter Substances. Liverpool University Press.
- Südhof, T. C. (2013). Neurotransmitter release: the last millisecond in the life of a synaptic vesicle. Neuron, 80, 675-690.
- Kandel, E. R., et al. (Eds.) (2013). Principles of Neural Science (5th ed.). McGraw-Hill.
