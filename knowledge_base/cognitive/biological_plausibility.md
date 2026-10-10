---
title: Biological Plausibility
description: The criterion that a proposed computation or learning rule could be implemented by neurons and synapses, with the standard objections to backpropagation and the case for predictive-coding schemes.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - neuroscience
  - implementation
  - learning
semantic_relations:
  - type: relates
    links:
      - "[[predictive_coding]]"
      - "[[bayesian_brain]]"
      - "[[hebbian_learning]]"
---

# Biological Plausibility

## Overview
A model is biologically plausible to the extent that its operations could be carried out by known neural machinery: spiking cells, synapses with local plasticity, laminar cortical wiring, and neuromodulators, within realistic time and energy budgets. The criterion is graded. Models are often described at Marr's computational or algorithmic level, and plausibility concerns whether an implementation-level story exists.

## Common criteria
- **Locality**: a synaptic change should depend on quantities available at that synapse, such as pre- and postsynaptic activity and a broadcast modulatory signal (see [[hebbian_learning]], [[synaptic_plasticity]]).
- **No weight transport or separate phases**: error backpropagation, as usually written, needs the backward pathway to use the transpose of forward weights and to run in a distinct phase. Crick (1989) raised this as a reason to doubt that the brain implements it directly; Lillicrap and colleagues (2016) showed that random feedback weights can support learning, and Whittington and Bogacz (2019) reviewed several candidate schemes.
- **Sign and rate constraints**: neurons are either excitatory or inhibitory, and firing rates are non-negative.
- **Resource constraints**: energy and wiring costs (see [[neural_efficiency]], [[metabolic_efficiency]]).
- **Temporal realism**: continuous-time dynamics and conduction delays, not synchronous discrete updates.

## Role in Active Inference
Predictive coding is the main bridge. Message passing for free-energy minimisation can be expressed with local operations: error units compute differences between top-down predictions and bottom-up signals, and updates to state and weight estimates use only locally available quantities. Whittington and Bogacz (2017) showed that a predictive-coding network can approximate backpropagation with local Hebbian-type updates. Bastos and colleagues (2012) proposed a mapping of the scheme onto the canonical cortical microcircuit, with prediction errors carried by superficial pyramidal cells and predictions by deep pyramidal cells. These are proposals constrained by anatomy and electrophysiology; direct evidence for the specific cell-type assignments is partial, and some assumptions (for example symmetric connectivity, or one-to-one error units) are open to the same plausibility objections they were meant to answer. Plausibility is therefore a live research question for the framework, not a settled strength.

## Related concepts
- [[free_energy_minimization]] — the optimisation whose neural implementation is at issue
- [[neural_computation]] — implementation-level descriptions
- [[recurrent_networks]] — architectures where timing and locality matter
- [[neuromorphic_computing]] — hardware motivated by neural constraints

## References
- Crick, F. (1989). The recent excitement about neural networks. Nature, 337, 129-132.
- Bastos, A. M., et al. (2012). Canonical microcircuits for predictive coding. Neuron, 76, 695-711.
- Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support error backpropagation for deep learning. Nature Communications, 7, 13276.
- Whittington, J. C. R., & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. Neural Computation, 29, 1229-1262.
- Whittington, J. C. R., & Bogacz, R. (2019). Theories of error back-propagation in the brain. Trends in Cognitive Sciences, 23, 235-250.
