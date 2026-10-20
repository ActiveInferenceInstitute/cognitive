---
external_mappings:
  - target: http://purl.obolibrary.org/obo/GO_0031594
    relation: broadMatch
    evidence: https://www.ebi.ac.uk/ols4/api/ontologies/go/terms?obo_id=GO:0031594
    rationale: >-
      The source defines the specialized synapse between a motor neuron and skeletal
      muscle fibre and describes cholinergic excitation. GO includes
      motor-axon/muscle-fibre junctions generally and explicitly includes inhibitory
      and non-cholinergic invertebrate junctions. GO therefore supplies a broader
      concept, not an exact match to this page's scope.
    reviewed_on: "2026-10-20"
title: Neuromuscular Junction
description: The chemical synapse between a motor neuron and a skeletal muscle fibre, how acetylcholine release produces contraction, its clinical disorders, and its place in motor control.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - synapse
  - acetylcholine
  - motor_system
semantic_relations:
  - type: relates
    links:
      - "[[motor_control]]"
      - "[[motor_cortex]]"
      - "[[ion_channels]]"
      - "[[cholinergic_system]]"
      - "[[synaptic_plasticity]]"
wikidata: Q776995
---

# Neuromuscular Junction

## Overview
The neuromuscular junction (NMJ) is the specialised synapse where the terminal of a motor neuron meets a skeletal muscle fibre. It is the final link in the chain from the nervous system to movement. Each muscle fibre is normally contacted by one motor neuron, while a single motor neuron branches to innervate many fibres, forming a motor unit.

## Transmission
When an action potential reaches the nerve terminal, voltage-gated calcium channels open and calcium entry triggers fusion of vesicles that release the neurotransmitter acetylcholine (ACh) into the synaptic cleft. ACh binds nicotinic receptors clustered on the folded postsynaptic membrane, the motor end plate. These receptors are ligand-gated cation channels (see [[ion_channels]]); their opening depolarises the membrane, producing an end-plate potential. The potential normally exceeds the threshold for voltage-gated sodium channels, so a muscle action potential spreads along the fibre and, through calcium release from the sarcoplasmic reticulum, causes contraction. The enzyme acetylcholinesterase degrades ACh in the cleft and ends the signal.

Neurotransmitter is released in discrete packets. Fatt and Katz (1952) recorded small spontaneous potentials at the end plate, the miniature end-plate potentials, which led to the quantal theory of transmitter release (Katz, 1966). The NMJ transmits with a large safety margin, so each nerve impulse reliably produces a muscle impulse, unlike many central synapses.

## Clinical relevance
Several conditions and agents act on this synapse. In myasthenia gravis, autoantibodies reduce functional ACh receptors and cause fatiguable weakness. Botulinum toxin blocks ACh release and causes paralysis. Curare blocks the receptors competitively. Such cases show that failure at one synapse can cause global loss of motor function.

## Relation to Active Inference
In motor formulations of Active Inference, descending signals are treated as predictions about proprioceptive input and not as direct commands, and spinal reflex arcs fulfil those predictions by moving the muscles (Adams et al., 2013; see [[motor_control]]). The NMJ is the peripheral interface through which these predictions are enacted. This is a modelling interpretation. The junction itself is described fully by classical electrophysiology.

## Related concepts
- [[motor_control]] — how movement is planned and executed
- [[motor_cortex]] — source of descending motor commands
- [[ion_channels]] — the receptor and voltage-gated channels in transmission
- [[cholinergic_system]] — acetylcholine signalling more broadly
- [[proprioception]] — sensory feedback from muscles

## References
- Fatt, P., & Katz, B. (1952). Spontaneous subthreshold activity at motor nerve endings. Journal of Physiology, 117, 109-128.
- Katz, B. (1966). Nerve, Muscle, and Synapse. McGraw-Hill.
- Adams, R. A., Shipp, S., & Friston, K. J. (2013). Predictions not commands: active inference in the motor system. Brain Structure and Function, 218, 611-643.
