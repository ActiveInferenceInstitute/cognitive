---
title: Distributed Control
description: "Distributed control coordinates many agents or components through local sensing and decisions without a central controller, as in ant colonies, swarms and networked systems, with a decentralised Active Inference reading."
type: system
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - control
  - decentralization
  - swarm
semantic_relations:
  - type: relates
    links:
      - "[[control_systems]]"
      - "[[swarm_intelligence]]"
      - "[[coordination]]"
      - "[[collective_behavior_ants]]"
---

# Distributed Control

## Overview

In distributed (decentralised) control, a system's behaviour is regulated by many controllers, each acting on local information and local authority, with no single unit that observes and commands everything. It contrasts with centralised control, where one controller computes all actions, and with hierarchical control, where authority is layered. Ant colonies, bird flocks, power grids and multi-robot teams are common examples.

## Principles

- **Local sensing and action**: each unit uses only what it can observe or receive from neighbours, so communication cost does not grow with system size in the way central coordination does.
- **Redundancy and graceful degradation**: because no unit is indispensable, loss of some units reduces performance gradually rather than causing failure (see [[robustness]]).
- **Consensus and agreement**: units that exchange values with neighbours can converge on a common value. In the standard linear averaging protocol on a connected graph, each node repeatedly moves toward the mean of its neighbours' states, and under conditions on the graph all states converge to the average; convergence speed depends on the graph's connectivity (its Laplacian spectrum).
- **Stigmergy and implicit communication**: units coordinate through marks left in a shared environment, as with pheromone trails ([[stigmergic_coordination]]), which removes the need for direct messaging.
- **Threshold-based task allocation**: each unit has a response threshold for a task stimulus, and population-level allocation emerges without assignments ([[task_allocation]]).

## Trade-offs

Distributed schemes tolerate faults and scale well, but they converge more slowly than a central optimiser with full information, can settle on suboptimal local equilibria, and are harder to analyse and guarantee. Designers choose interaction rules and network structure to achieve the desired global behaviour, a design problem known as the inverse problem of emergence. Verification is another cost, since global properties must be proved from local rules.

## Role in Active Inference

Active Inference scales naturally to decentralised settings, because each agent has its own generative model, its own observations and its own action selection; coordination arises when agents' predictions of each other, or of the shared environment, align. Multi-agent formulations in which every agent minimises its own free energy, with the other agents part of its environment, are a form of distributed control with no global objective written down. Whether a global quantity is thereby optimised depends on the model: in general each agent's local minimisation does not guarantee a collective optimum, so claims of "emergent optimality" need to be checked case by case. This is an active research area, not a settled method.

## Related concepts

- [[control_systems]] — centralised and feedback control background
- [[swarm_intelligence]] — decentralised problem solving in colonies
- [[interaction_rules]] — the local rules that determine global behaviour
- [[multi_agent_active_inference]] — decentralised agents with their own free energy
- [[fault_tolerance]] — resilience from redundancy
- [[collective_intelligence]] — group performance from decentralised contributions
