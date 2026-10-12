---
title: Collective Intelligence
description: "Collective intelligence is group-level problem solving that exceeds what members achieve alone, covering aggregation, diversity, independence, stigmergy, and multi-agent Active Inference accounts of shared beliefs."
type: system
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - collective_behavior
  - swarm
  - coordination
semantic_relations:
  - type: relates
    links:
      - "[[swarm_intelligence]]"
      - "[[stigmergic_coordination]]"
      - "[[collective_learning]]"
      - "[[multi_agent_active_inference]]"
wikidata: Q432197
---

# Collective Intelligence

## Overview

Collective intelligence is the capacity of a group, whether ants, humans or software agents, to solve problems, make estimates or adapt in ways that exceed what its typical member can do alone. It is a property of how individual contributions are combined, not simply of how many contribute. The term spans insect colonies ([[swarm_intelligence]]), human crowds and organisations, and engineered multi-agent systems.

## Mechanisms

- **Statistical aggregation**: averaging independent, unbiased estimates cancels individual error. The Condorcet jury theorem gives the discrete version: if each of $n$ voters is independently correct with probability $p > 0.5$, majority accuracy rises with $n$. For $p=0.6$ the majority is right with probability about 0.753 for $n=11$ and about 0.979 for $n=101$; for $p=0.4$ and $n=11$ it falls to about 0.247, so the same mechanism amplifies a bias.
- **Diversity and independence**: aggregation helps when errors are uncorrelated. Social influence that makes members copy each other can reduce the effective sample size and produce herding.
- **Division of labour**: specialisation and [[task_allocation]] raise throughput, as in colonies where worker allocation responds to local demand.
- **Indirect coordination**: agents modify a shared environment and respond to its traces, as in pheromone trails ([[stigmergic_coordination]]), wikis or version-controlled repositories.
- **Feedback**: positive feedback amplifies good options (recruitment), and negative feedback limits runaway commitment (saturation, decay of trails).
- **Structure**: network topology, communication delay and decision rules (quorum thresholds, voting) set the speed-accuracy trade-off.

## Measurement and caveats

Studies of human groups have reported a general "c factor" for group performance across tasks, which depends on social sensitivity and turn-taking more than on the maximum individual ability; the finding has been debated and replicated with mixed results. Groups also fail through groupthink, information cascades and correlated bias, so collective intelligence is conditional on design, not automatic.

## Role in Active Inference

Multi-agent Active Inference models treat a collective as agents that each infer hidden states and, through a shared environment or communication, align their generative models. Work on shared narratives and "shared generative models" describes coordination as a reduction in collective free energy when agents predict one another. Theory-of-mind and stigmergy fit naturally, since an agent's actions are observations for the others. The strongest claims, for instance that a collective itself has a Markov blanket and minimises a joint free energy, are a research programme with simulations as support rather than a settled result.

## Related concepts

- [[swarm_intelligence]] — insect-inspired collective problem solving
- [[stigmergic_coordination]] — coordination through environmental traces
- [[collective_learning]] — how groups improve with experience
- [[distributed_control]] — control without a central authority
- [[theory_of_mind]] — modelling other agents' beliefs
- [[cooperation]] — conditions that sustain joint action
