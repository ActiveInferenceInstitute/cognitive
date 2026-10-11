---
title: Bird Flocks
description: Coordinated groups of flying birds, the local interaction rules proposed to explain them, empirical findings on topological interaction in starlings, and links to swarm models and Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - flocking
  - collective_behavior
  - self_organization
semantic_relations:
  - type: relates
    links:
      - "[[swarm_intelligence]]"
      - "[[self_propelled_particles]]"
      - "[[collective_behavior]]"
      - "[[collective_intelligence]]"
      - "[[emergence]]"
---

# Bird Flocks

## Overview
A bird flock is a group of birds that fly together in coordinated motion, with no designated leader determining the movements of the whole. Dense flocks of starlings, which form murmurations, change shape and direction as if they were a single body. Flocking is a standard example of emergent collective behaviour, in which global patterns arise from interactions between individuals that each respond only to neighbours.

## Local rule models
Reynolds introduced the boids model in 1987 for computer animation. Each simulated bird follows three steering rules based on nearby flockmates: separation, which avoids crowding, alignment, which steers toward the average heading, and cohesion, which steers toward the average position. These rules are enough to generate realistic flock-like motion. Physicists formulated related models in terms of self-propelled particles with alignment (see [[self_propelled_particles]]), which show a transition from disorder to ordered collective motion.

## Empirical findings
Field studies in Rome used stereoscopic imaging to reconstruct the three-dimensional positions of individual starlings in large flocks. Ballerini and colleagues found that each bird interacts with a fixed number of nearest neighbours, about six to seven, regardless of how far away they are, rather than with all birds within a fixed distance. This topological interaction keeps flocks cohesive when density changes, for example when a flock is attacked by a predator. Later work found strong correlations over distances that exceed the interaction range, with information on turns propagating through the flock rapidly. Cavagna and Giardina reviewed this evidence in the framework of condensed matter and statistical physics.

## Relation to Active Inference
The connection is mainly by analogy and modelling. Flocking is a case of many agents coupled through shared observations, and simulations have represented each agent with a generative model of its neighbours' motion, so that alignment emerges from minimising prediction error. Such studies illustrate the multi-agent version of the framework. They show that the framework can reproduce flocking-like behaviour, not that birds actually perform inference of this kind, which would have to be tested against data.

## Related concepts
- [[swarm_intelligence]] — collective problem solving in decentralised groups
- [[self_propelled_particles]] — the physics models of alignment and flocking
- [[collective_behavior]] — group-level patterns arising from individual actions
- [[collective_intelligence]] — improved group performance through interaction
- [[swarm_robotics]] — engineered systems that use similar local rules
- [[emergence]] — higher-level order from lower-level interaction

## References
- Reynolds, C. W. (1987). Flocks, herds and schools: a distributed behavioral model. Computer Graphics, 21(4), 25-34.
- Ballerini, M., Cabibbo, N., Candelier, R., Cavagna, A., Cisbani, E., Giardina, I., et al. (2008). Interaction ruling animal collective behavior depends on topological rather than metric distance: evidence from a field study. Proceedings of the National Academy of Sciences, 105, 1232-1237.
- Cavagna, A., & Giardina, I. (2014). Bird flocks as condensed matter. Annual Review of Condensed Matter Physics, 5, 183-207.
