---
title: Neuromorphic Computing
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - cognitive
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/cognitive/computational_neuroscience]]"
      - "[[knowledge_base/systems/edge_computing]]"
      - "[[knowledge_base/cognitive/neural_efficiency]]"
      - "[[knowledge_base/cognitive/energy_optimization]]"
wikidata: Q137425629
---

# Neuromorphic Computing

## Overview

Neuromorphic computing builds hardware and software whose organisation follows that of nervous systems: many simple units that communicate by discrete spikes, memory placed next to computation, and activity driven by events instead of a global clock. Example platforms include IBM's TrueNorth, Intel's Loihi and the SpiNNaker machine developed at Manchester. The main motivation is energy efficiency for sparse, temporally structured workloads such as sensing and control. Running Active Inference on such hardware is a proposal at the research stage; the connection is a plausible fit of algorithm and substrate, not an established engineering result.

## Main Ideas

**Spiking networks.** Neurons integrate input and emit a spike when a threshold is crossed, so information is carried by spike timing and rate. Computation happens only where spikes occur, which allows power to scale with activity.

**Event-driven processing.** Sensors such as event cameras report changes in the scene instead of full frames. A system that updates only when its input changes pairs naturally with an inference scheme that reacts to prediction errors, since an error is also an event signalling that the model's expectation failed.

**Locality of computation.** Predictive coding and message-passing formulations of free-energy minimisation use local operations: each unit combines its own prediction with errors from neighbouring units. That locality is the property neuromorphic chips reward, and it is the main argument for the match. The neural process theories of Active Inference go further and assign these quantities to populations of spiking cells ([[knowledge_base/cognitive/neural_active_inference]], [[knowledge_base/cognitive/predictive_coding]]).

## Open Problems

Spike-based representations of probability distributions are not standardised. Learning rules that are both local and well matched to variational objectives remain under study. Choosing a discretisation for continuous-time inference on digital neuromorphic chips involves precision limits. The word "free energy" here is information-theoretic, so lower variational free energy does not mean lower electrical power; any energy saving must be measured on the hardware.

## In This Repository

Related pages are [[knowledge_base/cognitive/neural_efficiency]], [[knowledge_base/cognitive/energy_optimization]], [[knowledge_base/cognitive/metabolic_efficiency]], [[knowledge_base/cognitive/neural_synchrony]] and [[knowledge_base/systems/edge_computing]] for deployment on constrained devices. No spiking, event-driven or hardware-specific code exists in the repository. The agents in `code/Things` and `code/tools/src` are dense NumPy implementations that update on a fixed step schedule, so any neuromorphic mapping would be a re-implementation.

## Related Resources

- [[knowledge_base/cognitive/computational_neuroscience]]
- [[knowledge_base/systems/edge_computing]]
