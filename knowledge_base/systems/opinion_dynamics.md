---
title: Opinion Dynamics
description: Mathematical models of how opinions form, spread and change in groups through social influence, including DeGroot averaging and bounded-confidence models, and their relation to belief updating in Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - social_dynamics
  - consensus
  - agent_based_modeling
semantic_relations:
  - type: relates
    links:
      - "[[social_systems]]"
      - "[[network_theory]]"
      - "[[wisdom_of_crowds]]"
      - "[[social_learning]]"
      - "[[agent_based_modeling]]"
---

# Opinion Dynamics

## Overview
Opinion dynamics studies how the opinions of interacting individuals evolve over time. Each individual holds a state, which may be a binary choice or a number on a scale, and updates it in response to the states of others. Typical questions are whether a population reaches consensus, splits into clusters or polarises, and how network structure and the way people weigh other views shape the result.

## Models
- **DeGroot model.** Opinions form a vector $x(t)$ and each update is $x(t+1)=W x(t)$, where $W$ is a row-stochastic matrix of trust weights. If the influence network is strongly connected and aperiodic, opinions converge to a common value, a weighted average of the initial ones. DeGroot introduced the model in 1974.
- **Friedkin-Johnsen model.** Individuals remain partly anchored to their initial opinions, so consensus need not be reached.
- **Bounded confidence.** In the models of Deffuant and of Hegselmann and Krause, agents only influence one another when their opinions differ by less than a threshold. Depending on the threshold, the population reaches consensus or fragments into several clusters.
- **Voter and Ising-type models.** Discrete opinions are copied from neighbours or aligned by local interaction, borrowing methods from statistical physics.

Other work adds noise, agents who never change their view, heterogeneous networks, and mechanisms thought to explain echo chambers.

## Empirical connection
Models are typically validated by their qualitative behaviour. Measuring real opinion change at scale is difficult, and fitting simple update rules to survey or online data remains an open problem. Castellano, Fortunato and Loreto reviewed such models from the point of view of statistical physics.

## Relation to Active Inference
The link is an active but small research programme. Social influence can be modelled as agents who update beliefs by Bayesian inference using observations of other agents' statements as evidence, with the weight given depending on the precision they assign to the source. In such treatments, echo chambers and polarisation arise from agents that trust only like-minded sources. This offers a principled alternative to the fixed update rules above, although the models are largely theoretical and have not been compared systematically to data.

## Related concepts
- [[social_systems]] — the settings in which opinions spread
- [[network_theory]] — structure of who influences whom
- [[wisdom_of_crowds]] — when aggregated opinions are accurate
- [[social_learning]] — acquiring beliefs from others
- [[agent_based_modeling]] — simulation approach used for these models
- [[belief_updating]] — the Bayesian counterpart of opinion change

## References
- DeGroot, M. H. (1974). Reaching a consensus. Journal of the American Statistical Association, 69, 118-121.
- Hegselmann, R., & Krause, U. (2002). Opinion dynamics and bounded confidence models, analysis and simulation. Journal of Artificial Societies and Social Simulation, 5(3).
- Castellano, C., Fortunato, S., & Loreto, V. (2009). Statistical physics of social dynamics. Reviews of Modern Physics, 81, 591-646.
