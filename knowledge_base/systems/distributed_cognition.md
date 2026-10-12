---
title: Distributed Cognition
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - distributed_cognition
  - extended_mind
  - cognitive_science
  - collective_behavior
aliases:
  - distributed_systems
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/philosophy/4e_cognition]]"
      - "[[knowledge_base/cognitive/embodied_cognition]]"
      - "[[knowledge_base/cognitive/swarm_intelligence]]"
      - "[[knowledge_base/cognitive/stigmergic_coordination]]"
wikidata: Q7551870
---

# Distributed Cognition

## Overview
Distributed cognition is the view that cognitive processes are not confined to a single brain but are spread across multiple individuals, artifacts, and the environment. Edwin Hutchins developed it in the 1980s and 1990s, most fully in Cognition in the Wild (1995), based on ethnography of navigation on a US Navy ship.

## Mechanism
The unit of analysis is the functional system that performs a task, such as a navigation team with its charts, instruments, and procedures, instead of the individual mind. Computation is described as the propagation of representational states across media: a bearing is read from an alidade, spoken, plotted on a chart, and compared with a prior fix, with each step transforming the information. Three kinds of distribution are usually distinguished: across people in a social group, between internal structure and external structure (artifacts, tools, spatial layouts), and across time, where earlier products shape later activity. Artifacts do real cognitive work by changing task structure; a checklist or a well-designed interface turns a memory problem into a recognition problem.

The position is related to, but not identical with, the extended mind thesis of Clark and Chalmers (1998), which is a philosophical claim about when external resources count as parts of an individual's mind; distributed cognition is a method for explaining group-level performance. Both lie within the family of 4E approaches. The idea should not be confused with distributed computing, in which tasks are partitioned across processors. Critics question whether group-level cognitive states are explanatory beyond the sum of individual contributions, and where to draw the boundary of a cognitive system.

Social insects supply a clear case where group-level information processing is evident without individual comprehension: a colony allocating workers or choosing a nest site integrates local signals through interaction rates and persistent environmental marks.

## Role in Active Inference
Active Inference offers a formal handle on the boundary question: a system is individuated by a Markov blanket, and blankets can be nested or can include artifacts and other agents when the coupling is tight enough to be treated as one conditional-independence structure. Shared generative models, in which members predict one another and the shared environment, are one way to describe how a team functions as a unit. How well this fits human teams in practice is an open empirical question, and Markov-blanket individuation of social systems is debated. In this repository the clearest example is the Ant_Colony Thing, where the pheromone field carries the colony's working memory.

## Related concepts
- [[knowledge_base/philosophy/4e_cognition]] — embodied, embedded, enactive, extended cognition
- [[knowledge_base/cognitive/embodied_cognition]] — the role of body and action in cognition
- [[knowledge_base/cognitive/stigmergic_coordination]] — environment as memory for a group
- [[knowledge_base/cognitive/swarm_intelligence]] — group-level information processing in insects
- [[knowledge_base/mathematics/markov_blanket]] — the formal boundary concept
- [[knowledge_base/systems/collective_learning]] — how distributed systems improve over time

## References
- Hutchins, E. (1995). Cognition in the Wild.
- Clark, A., and Chalmers, D. (1998). The extended mind. Analysis.
- Hollan, J., Hutchins, E., and Kirsh, D. (2000). Distributed cognition: toward a new foundation for human-computer interaction research. ACM Transactions on Computer-Human Interaction.
