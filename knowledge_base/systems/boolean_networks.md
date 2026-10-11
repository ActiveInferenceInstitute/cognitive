---
title: Boolean Networks
description: Discrete dynamical models in which nodes take binary values updated by logical rules, used to study gene regulation, attractors and the order-chaos transition, with a limited link to Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - boolean_networks
  - discrete_dynamics
  - gene_regulation
semantic_relations:
  - type: relates
    links:
      - "[[gene_regulatory_networks]]"
      - "[[cellular_automata]]"
      - "[[attractor_dynamics]]"
      - "[[dynamical_systems]]"
      - "[[critical_transitions]]"
---

# Boolean Networks

## Overview
A Boolean network consists of $N$ nodes, each in one of two states (on or off, 1 or 0), connected by a directed graph. Every node has a Boolean function of its input nodes that determines its next state. Because the state space is finite, with $2^N$ configurations, deterministic dynamics must eventually enter a repeating pattern. Boolean networks are a minimal model of systems in which components switch between active and inactive states, and they are among the simplest examples of high-dimensional nonlinear dynamics that can be studied exhaustively.

## Dynamics and attractors
Updating may be synchronous, with all nodes changing together, or asynchronous. Under synchronous updating the dynamics is deterministic: each configuration has one successor, so trajectories end in either a fixed point or a cycle. These terminal sets are the attractors, and the set of configurations leading to an attractor is its basin. In models of gene regulation, attractors are interpreted as stable cell types or cell-cycle states, and perturbations correspond to flipping node values.

## Random networks and criticality
Kauffman studied random Boolean networks in which each node receives $K$ inputs chosen at random, with random functions. Behaviour depends on connectivity and on how biased the functions are. For unbiased functions, networks with $K<2$ tend toward ordered dynamics with short cycles, networks with $K>2$ toward chaotic dynamics in which small perturbations spread, and $K=2$ lies at the boundary. Kauffman proposed that biological regulation operates near this critical regime, which balances stability with flexibility. This is a hypothesis about genetic networks that has been influential and tested against data to varying degrees.

## Relation to Active Inference
The link is limited. Boolean networks are not part of the Active Inference formalism, although discrete-state generative models and Boolean network models both describe systems with finite state spaces and attractors. Both can represent regulation as dynamics settling into stable configurations, and the criticality discussion parallels interest in systems poised between order and disorder. Any stronger connection, such as inferring network structure from data in a Bayesian manner, would be a separate modelling choice and not a feature of the framework.

## Related concepts
- [[gene_regulatory_networks]] — the main biological application
- [[cellular_automata]] — Boolean updating on a regular lattice
- [[attractor_dynamics]] — fixed points and cycles as stable outcomes
- [[critical_transitions]] — the order-chaos boundary and sudden shifts
- [[complex_systems_biology]] — the wider context for network models of cells
- [[dynamical_systems]] — the general mathematical setting

## References
- Kauffman, S. A. (1969). Metabolic stability and epigenesis in randomly constructed genetic nets. Journal of Theoretical Biology, 22, 437-467.
- Kauffman, S. A. (1993). The Origins of Order: Self-Organization and Selection in Evolution. Oxford University Press.
