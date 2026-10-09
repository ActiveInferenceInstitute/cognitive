---
title: Edge Computing
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - systems
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/systems/internet_of_things]]"
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/cognitive/neuromorphic_computing]]"
      - "[[knowledge_base/systems/fault_tolerance]]"
---

# Edge Computing

## Overview

Edge computing runs computation near where data is produced, on a gateway, phone, sensor node or microcontroller, rather than sending everything to a central cloud. The reasons are latency (a control loop cannot wait for a round trip), bandwidth, intermittent connectivity, energy and privacy. TinyML is the sub-field that fits machine learning models onto devices with kilobytes to a few megabytes of memory. Running Active Inference agents at the edge is a plausible engineering direction with few published deployments, so this page mostly discusses what the computation costs and what the trade-offs are.

## Constraints and Standard Techniques

Memory, compute and energy budgets are small, floating-point support may be missing, and a device may sleep most of the time. Common responses are quantisation (8-bit integer weights and activations), pruning, fixed-point arithmetic, duty cycling, and splitting work between device and server so that the device only does time-critical steps. Each trade accuracy or flexibility for cost: aggressive quantisation can distort small probabilities, which matters for inference that multiplies and normalises probabilities.

## What Active Inference Costs

For a discrete agent with $S$ states, $O$ observations and $A$ actions, a belief update is a matrix-vector product of order $S^2$ for the prediction plus a pointwise product with a likelihood row, so a single perceptual step is cheap. Planning dominates: scoring every action sequence of horizon $H$ means evaluating on the order of $A^H$ policies, each requiring state prediction and an expected-free-energy computation. On an embedded device this forces a short horizon, a cap on the number of policies, a small state space, or precomputed policy values. Continuous schemes replace this with gradient steps on a free energy, whose cost grows with the number of state variables and orders of motion but not exponentially with horizon.

Several features suit the setting: the update is dominated by small dense products, beliefs are compact vectors that can be sent between nodes, and precision gives a built-in way to downweight unreliable sensor readings. Distributed inference across nodes would be a message-passing problem, and message loss then becomes a fault-tolerance question ([[knowledge_base/systems/fault_tolerance]], [[knowledge_base/mathematics/message_passing]]).

## In This Repository

Related pages are [[knowledge_base/systems/internet_of_things]], [[knowledge_base/cognitive/neuromorphic_computing]], [[knowledge_base/cognitive/computational_efficiency]] and [[knowledge_base/cognitive/energy_optimization]]. The code is written for clarity on a workstation, not for devices: it uses NumPy with double-precision floats and has no quantised, fixed-point or embedded build. Two features are relevant to cost control. `InferenceConfig.policy_limit` (default 4096) caps the number of enumerated policies and raises an error if the cap would leave some first action unrepresented. And `cognitive-benchmark` reports seconds per call for matrix operations, dispatcher inference, the Simple POMDP step and continuous inference, which gives a baseline to compare against on target hardware.

## Related Resources

- [[knowledge_base/systems/internet_of_things]]
- [[knowledge_base/cognitive/active_inference]]
