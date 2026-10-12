---
title: Sociophysics
description: The use of models and methods from statistical physics to study collective social phenomena such as opinion formation, cultural dynamics and the spread of norms.
type: concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - statistical_physics
  - social_dynamics
  - opinion_dynamics
  - agent_based_models
semantic_relations:
  - type: relates
    links:
      - "[[statistical_physics]]"
      - "[[opinion_dynamics]]"
      - "[[phase_transitions]]"
      - "[[agent_based_modeling]]"
      - "[[network_models]]"
      - "[[social_systems]]"
      - "[[econophysics]]"
---

# Sociophysics

## Overview
Sociophysics treats a population of interacting individuals in the way statistical physics treats a population of interacting particles. Individual people are reduced to a few variables, such as an opinion or a cultural trait, and simple rules for how neighbours influence each other. The question is which large-scale patterns, such as consensus, polarisation or fragmentation, follow from the rules and which microscopic details do not matter. Castellano, Fortunato and Loreto (2009) review the field as the statistical physics of social dynamics, covering opinion, cultural and language dynamics, crowd behaviour, hierarchy formation, human dynamics and social spreading, and comparing model results with empirical data where possible. Galam (2008) reviews models he and co-authors introduced over 25 years, grouped into five classes: democratic voting in hierarchical systems, decision making, fragmentation versus coalitions, terrorism and opinion dynamics.

## Typical models
- **Voter-type models.** Each agent holds one of a few discrete opinions and copies an opinion from a neighbour. On a finite population, the process ends in consensus, and the time taken depends on the structure of the network.
- **Ising-like and majority-rule models.** Opinions are binary spins, and an agent aligns with the local majority or with a group chosen at random. Galam's voting models belong to this family.
- **Bounded-confidence and cultural models.** Agents interact only when their opinions or trait vectors are already close enough, which can lead to several coexisting clusters instead of global agreement.

Typical analyses use mean-field approximations, simulation and the theory of [[phase_transitions]], looking for a critical value of a parameter such as noise or connectivity at which the system switches between ordered and disordered states.

## Limits
The models are stylised. Their agents have no goals, memory or beliefs about other agents, and quantitative comparison with data is difficult. Galam (2008) discusses the perspectives and challenges involved in making sociophysics a predictive science. Similar-looking dynamics can come from different mechanisms, so agreement with a qualitative pattern is weak evidence for a particular model.

## Relation to Active Inference
Sociophysics and Active Inference address collective behaviour at opposite levels of detail. Sociophysics assumes minimal agents and studies the emergent statistics, whereas Active Inference gives each agent a generative model and a policy, and then asks what collective dynamics follow when agents infer each other's states. The two can be combined in agent-based simulations: a rule such as imitation of a neighbour can be seen as a simple stand-in for belief updating, but this is an interpretive reading and does not follow from the physics models themselves. No result in sociophysics is a consequence of the free energy principle.

## Related concepts
- [[opinion_dynamics]] — the central application area
- [[statistical_physics]] — source of the methods
- [[phase_transitions]] — transitions between consensus and fragmentation
- [[agent_based_modeling]] — simulation of the micro-rules
- [[econophysics]] — the corresponding programme for economic data

## References
- Castellano, C., Fortunato, S., & Loreto, V. (2009). Statistical physics of social dynamics. Reviews of Modern Physics, 81(2), 591-646. https://doi.org/10.1103/RevModPhys.81.591
- Galam, S. (2008). Sociophysics: a review of Galam models. International Journal of Modern Physics C, 19(3), 409-440. https://doi.org/10.1142/S0129183108012297
