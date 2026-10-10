---
title: Interaction Rules
description: "Interaction rules are the local rules governing how agents sense and respond to neighbours and environment, from which collective patterns arise, with examples from flocking and ant foraging and a link to Active Inference."
type: system
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - emergence
  - agent_based_modeling
  - collective_behavior
semantic_relations:
  - type: relates
    links:
      - "[[emergence]]"
      - "[[swarm_intelligence]]"
      - "[[collective_behavior_ants]]"
      - "[[cellular_automata]]"
---

# Interaction Rules

## Overview

Interaction rules specify how an individual agent updates its state or chooses an action as a function of what it perceives locally: neighbouring agents, environmental signals or its own recent history. In agent-based and complex-systems modelling, the collective pattern (a flock, a foraging trail, a traffic jam) is a consequence of the rule set together with the agents' spatial and network arrangement. Collective behaviour is "emergent" in the modelling sense that it is not stated in any rule.

## Types of rule

- **Metric and topological neighbourhoods**: agents react to others within a fixed distance, or to a fixed number of nearest neighbours. Empirical work on starlings suggested topological interaction, which keeps flocks cohesive under changes in density.
- **Attraction, alignment, repulsion**: the classic flocking rules of the Boids model (Reynolds, 1987) are separation, alignment and cohesion. Vicsek-type models use a single alignment rule with noise and show an order-disorder transition as noise or density changes.
- **Stimulus-response thresholds**: an agent engages in a task when a local stimulus exceeds a personal threshold.
- **Stigmergic rules**: agents deposit and follow markers. In ant foraging models, the choice probability for a path rises with pheromone concentration, and evaporation prevents lock-in, so shorter routes win through differential reinforcement.
- **Imitation and quorum rules**: agents copy others, with the probability depending on the number of agents doing so, which can produce threshold-like group decisions.
- **Game-theoretic rules**: strategies such as tit-for-tat in repeated interaction produce cooperation under suitable conditions.

## Properties of rule sets

Small rule changes can produce qualitatively different collective regimes, and parameters often have critical values at which order appears. Rules inferred from observation are typically underdetermined: different rule sets can give the same group-level statistics, so model validation needs individual-level data such as tracked trajectories. Simplicity is a modelling choice, not evidence that real animals use simple rules.

## Role in Active Inference

In an Active Inference model, an interaction rule is not specified directly; it follows from the agent's generative model and preferences. An agent that expects to observe neighbours at a certain distance or alignment, and acts to fulfil this expectation, behaves as though it followed an attraction-alignment rule. The advantage is that the "rule" adapts with learning and responds to uncertainty. The cost is that fitting such models to data is more demanding than fitting a fixed rule, and the equivalence between a given rule and a given generative model is a modelling assumption, not a theorem in general.

## Related concepts

- [[emergence]] — collective phenomena produced by local rules
- [[swarm_intelligence]] — rule-based problem solving in groups
- [[stigmergic_coordination]] — rules mediated by environmental marks
- [[distributed_control]] — engineering of local rules for global goals
- [[cellular_automata]] — the simplest formal rule-based systems
- [[ant_colony_optimization]] — pheromone rules as an optimisation algorithm
