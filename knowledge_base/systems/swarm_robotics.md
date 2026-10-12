---
title: Swarm Robotics
description: "Swarm robotics designs large groups of simple robots that coordinate through local interaction and stigmergy, drawing on social insects, and relates to multi-agent Active Inference and decentralised control."
type: system
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - robotics
  - swarm
  - multi_agent
semantic_relations:
  - type: relates
    links:
      - "[[swarm_intelligence]]"
      - "[[social_insect_cognition]]"
      - "[[distributed_control]]"
      - "[[multi_agent_active_inference]]"
aliases:
  - robotic_swarms
wikidata: Q2446723
---

# Swarm Robotics

## Overview

Swarm robotics is an approach to multi-robot systems in which a large number of relatively simple, typically identical robots act on local sensing and local communication, with no central controller, so that useful collective behaviour emerges. It takes its design principles from social insects such as ants, bees and termites ([[social_insect_cognition]], [[swarm_intelligence]]). A commonly cited set of criteria says the robots are autonomous, numerous, homogeneous, limited in capability, and unable alone to complete the task.

## Design principles

- **Local interaction rules**: each robot reacts to nearby robots and the environment ([[interaction_rules]]). Behaviour-based controllers or finite-state machines are typical, because the hardware is small.
- **Scalability**: performance should remain acceptable as group size changes, without retuning, because the interaction depends only on neighbours.
- **Robustness**: with many replaceable units, losing some reduces performance gradually ([[robustness]]).
- **Flexibility**: the same robots can switch between collective tasks by changing behaviour.

## Collective tasks

- **Aggregation and dispersion**: robots cluster or spread using simple attraction or repulsion rules.
- **Collective transport and construction**: groups push or build objects larger than any member, sometimes using stigmergic cues in the construction itself.
- **Foraging and path formation**: ant-inspired trail laying, implemented with virtual or physical markers, since real chemical pheromones are impractical to deploy.
- **Pattern formation and shape forming**: robots arrange into target shapes using local position estimates.
- **Exploration, mapping and search**: coverage of an area with decentralised task allocation ([[task_allocation]]).
- **Collective decision-making**: consensus on one of several options through quorum or opinion-dynamics rules.

## Challenges

Predicting global behaviour from local rules is hard, and design is often by simulation, evolutionary robotics or hand-tuned rules followed by testing on the physical swarm. Sensing noise, limited communication and the gap between simulation and real hardware are standing difficulties. Formal guarantees exist for some tasks such as consensus but are rare for complex ones.

## Role in Active Inference

Active Inference has been applied to single robots for perception and control, and to multi-agent settings in simulation. A swarm in this framing is a set of agents, each with a generative model of its local surroundings and its neighbours, selecting actions by expected free energy. This could replace hand-written rules with preferences over local observations, but swarm-scale Active Inference on physical hardware is largely a research direction and not an established method. In this repo the Ant_Colony Thing holds design documentation for an ant-colony Active Inference model, with no executable code (only Simple_POMDP and Continuous_Generic run), and it is not a robotics platform.

## Related concepts

- [[swarm_intelligence]] — the biological and algorithmic basis
- [[stigmergic_coordination]] — environment-mediated coordination used in robot swarms
- [[collective_behavior_ants]] — ant models that inspire foraging algorithms
- [[distributed_control]] — control theory view of decentralised robot teams
- [[robotics]] — Active Inference applied to robotic agents
- [[ant_colony_optimization]] — optimisation methods derived from ant foraging
